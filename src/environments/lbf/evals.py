from typing import Callable, Dict, Tuple

import chex
import jax
import jax.numpy as jnp
import pandas as pd
from tqdm import tqdm

from src.evaluate import evaluate, Policy
from src.utils import get_type_dists
from .lbf import LBFEnv, LBFEnvParams, LBFEnvState, make


def compute_episode_metrics(final_state: LBFEnvState) -> Dict[str, chex.Array]:
    agent_level = final_state.agent_levels[0]
    goal_levels = final_state.fruit_levels * final_state.goals_attempted
    solo_goals = jnp.where((goal_levels <= agent_level) & (goal_levels > 0), 1, 0)
    coop_goals = jnp.where(goal_levels > agent_level, 1, 0)
    return {"n_solo": solo_goals.sum(), "n_coop": coop_goals.sum()}


def run_evals(
    rng: chex.PRNGKey,
    policy: Policy,
    env: LBFEnv,
    default_env_params: LBFEnvParams,
    num_seeds=100,
) -> pd.DataFrame:
    eval_data = {"eval type": [], "return": [], "cooperativity": []}
    eval_npc_type_dists = get_type_dists(
        env.prefs_support, env.prefs_support[default_env_params.learner_agent_type]
    )

    for k in tqdm(eval_npc_type_dists.keys()):
        env_params = LBFEnvParams(
            max_steps_in_episode=default_env_params.max_steps_in_episode,
            learner_agent_type=default_env_params.learner_agent_type,
            npc_policy_params=default_env_params.npc_policy_params,
            normalise_reward=default_env_params.normalise_reward,
            npc_type_dist=eval_npc_type_dists[k],
        )
        _, returns, metrics = evaluate(
            policy,
            rng,
            env,
            env_params,
            compute_episode_metrics,
            num_seeds,
            max_steps_in_episode=env_params.max_steps_in_episode,
        )
        eval_data["eval type"].extend([k] * num_seeds)
        eval_data["return"].extend(returns.tolist())
        metrics = {k: int(v.sum()) for k, v in metrics.items()}
        cooperativity = metrics["n_coop"] / (metrics["n_coop"] + metrics["n_solo"])
        eval_data["cooperativity"].extend([cooperativity] * num_seeds)
    return pd.DataFrame(eval_data)


def compute_lbf_reference_returns(key: chex.PRNGKey, n_eps=100) -> pd.DataFrame:
    eval_data = {"eval type": [], "return": []}
    env, default_env_params = make()
    step_jitted = jax.jit(env.step)
    policy_jitted = jax.jit(env.reference_policy)

    def play_episode(key: chex.PRNGKey, env_params: LBFEnvParams) -> chex.Array:
        def cond_fun(args):
            _, _, _, done = args
            return ~done

        def body_fun(args):
            key, state, ret, done = args
            key, key_policy, key_step = jax.random.split(key, 3)
            action = policy_jitted(key_policy, state, 0)
            obs, state, reward, done, _ = step_jitted(
                key_step, state, action, env_params
            )
            return key, state, ret + reward, done

        _, state = env.reset(key, env_params)
        key, state, ret, done = jax.lax.while_loop(
            cond_fun, body_fun, (key, state, jnp.array(0.0), False)
        )

        return ret

    for eval_type in tqdm(eval_npc_type_dists.keys()):
        env_params = get_env_params(default_env_params, eval_type)
        keys = jax.random.split(key, n_eps)
        returns = jax.vmap(play_episode, in_axes=(0, None))(keys, env_params)
        eval_data["eval type"].extend([eval_type] * n_eps)
        eval_data["return"].extend(returns.tolist())

    return pd.DataFrame(eval_data)

import time
from typing import Callable, Tuple

import chex
import jax
import jax.numpy as jnp
from rejax.evaluate import evaluate
import pandas as pd
from tqdm import tqdm

from .lbf import LBFEnv, LBFEnvParams, make

eval_npc_type_dists = {
    "no overlap": jnp.array([0, 1, 0, 0, 0, 0, 0, 0]),
    "partial overlap": jnp.array([0, 0, 1, 1, 1, 1, 0, 0]) / 4,
    "full overlap": jnp.array([0, 0, 0, 0, 0, 0, 1, 1]) / 2,
    "overall": jnp.array([0, 1, 1, 1, 1, 1, 1, 1]) / 7
}

def get_env_params(default_params, eval_type):
    return LBFEnvParams(
        max_steps_in_episode=default_params.max_steps_in_episode,
        learner_agent_type=default_params.learner_agent_type,
        npc_policy_params=default_params.npc_policy_params,
        normalise_reward=default_params.normalise_reward,
        npc_type_dist=eval_npc_type_dists[eval_type]
    )

def run_evals(
    rng: chex.PRNGKey,
    policy: Callable[[chex.Array, chex.PRNGKey], chex.Array],
    env: LBFEnv,
    default_env_params: LBFEnvParams,
    num_seeds=100
) -> pd.DataFrame:
    eval_data = {"eval type": [], "return": []}
    for k in tqdm(eval_npc_type_dists.keys()):
        env_params = get_env_params(default_env_params, k)
        _, returns = evaluate(
            policy,
            rng,
            env,
            env_params,
            num_seeds,
            max_steps_in_episode=env_params.max_steps_in_episode
        )
        eval_data["eval type"].extend([k] * num_seeds)
        eval_data["return"].extend(returns.tolist())
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
            obs, state, reward, done, _ = step_jitted(key_step, state, action, env_params)
            return key, state, ret + reward, done

        _, state = env.reset(key, env_params)
        key, state, ret, done = jax.lax.while_loop(
            cond_fun, body_fun, (key, state, jnp.array(0.0), False)
        )

        return ret

    for eval_type in tqdm(eval_npc_type_dists.keys()):
        env_params = get_env_params(default_env_params, eval_type)
        keys = jax.random.split(key, n_eps)
        returns = jax.vmap(
            play_episode, in_axes=(0, None)
        )(keys, env_params)
        eval_data["eval type"].extend([eval_type] * n_eps)
        eval_data["return"].extend(returns.tolist())

    return pd.DataFrame(eval_data)

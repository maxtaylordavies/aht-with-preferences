from typing import Dict

import chex
import jax.numpy as jnp
import pandas as pd
from tqdm import tqdm

from src.evaluate import evaluate, Policy
from src.utils import get_type_dists
from .lbf import LBFEnv, LBFEnvParams, LBFEnvState

goal_metric_keys = ["n_total", "n_g_1", "n_g_2", "n_g_3", "n_g_4", "n_solo", "n_coop"]


def compute_episode_goal_stats(
    env: LBFEnv, final_state: LBFEnvState
) -> Dict[str, chex.Array]:
    attempted = final_state.goals_attempted
    types = final_state.fruit_types
    levels = final_state.fruit_levels

    learner_prefs = env.prefs_support[final_state.agent_types[0]][types]
    teammate_prefs = env.prefs_support[final_state.agent_types[1]][types]
    shared_prefs = learner_prefs * teammate_prefs

    learner_level = final_state.agent_levels[0]
    solo = jnp.where((levels <= learner_level), 1, 0)
    coop = 1 - solo

    # number of goals attempted outside G_learner
    n_g_1 = (attempted * (1 - learner_prefs)).sum()

    # number of goals attempted in G_learner and in G_solo
    n_g_2 = (attempted * learner_prefs * solo).sum()

    # number of goals attempted in G_learner and in G_teammates (and not in G_solo)
    n_g_3 = (attempted * shared_prefs * coop).sum()

    # number of goals attempted in G_learner and not in either G_solo or G_teammates
    n_g_4 = (attempted * learner_prefs * (1 - teammate_prefs) * coop).sum()

    # number of solo and cooperative goals attempted
    n_solo, n_coop = (attempted * solo).sum(), (attempted * coop).sum()
    n_total = n_solo + n_coop

    vars = locals()
    return {k: vars[k] for k in goal_metric_keys}


def run_evals(
    rng: chex.PRNGKey,
    policy: Policy,
    env: LBFEnv,
    default_env_params: LBFEnvParams,
    num_seeds=100,
) -> pd.DataFrame:
    eval_data = {"eval type": [], "return": []}
    eval_data = {**eval_data, **{k: [] for k in goal_metric_keys}}

    eval_npc_type_dists = get_type_dists(
        env.prefs_support, env.prefs_support[default_env_params.learner_agent_type]
    )

    for k in tqdm(eval_npc_type_dists.keys()):
        env_params = LBFEnvParams(
            max_steps_in_episode=default_env_params.max_steps_in_episode,
            learner_agent_type=default_env_params.learner_agent_type,
            npc_policy_params=default_env_params.npc_policy_params,
            normalise_rewards=default_env_params.normalise_rewards,
            npc_type_dist=eval_npc_type_dists[k],
        )
        _, returns, metrics = evaluate(
            policy,
            rng,
            env,
            env_params,
            compute_episode_goal_stats,
            num_seeds,
            max_steps_in_episode=env_params.max_steps_in_episode,
        )
        eval_data["eval type"].extend([k] * num_seeds)
        eval_data["return"].extend(returns.tolist())
        for k, v in metrics.items():
            eval_data[k].extend([int(v.sum())] * num_seeds)
    return pd.DataFrame(eval_data)

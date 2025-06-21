from typing import Any, Dict

import chex
import jax
import pandas as pd
from tqdm import tqdm

from src.evaluate import evaluate, Policy
from src.utils import get_type_dists
from .reaching import ReachingEnv, ReachingEnvState, ReachingEnvParams

goal_metric_keys = ["n_total", "n_g_1", "n_g_2", "n_g_3", "n_g_4", "n_solo", "n_coop"]


def compute_episode_goal_stats(
    env: ReachingEnv, final_state: ReachingEnvState, attempted: chex.Array
) -> Dict[str, chex.Array]:
    learner_prefs = env.prefs_support[final_state.agent_types[0]].astype(int)
    teammate_prefs = env.prefs_support[final_state.agent_types[1]].astype(int)
    shared_prefs = (learner_prefs * teammate_prefs).astype(int)

    # jax.debug.print(
    #     "Learner prefs: {learner_prefs}, Teammate prefs: {teammate_prefs}, Shared prefs: {shared_prefs}, Attempted: {attempted}",
    #     learner_prefs=learner_prefs,
    #     teammate_prefs=teammate_prefs,
    #     shared_prefs=shared_prefs,
    #     attempted=attempted,
    # )

    # number of goals attempted outside G_learner
    n_g_1 = (attempted[:4] * (1 - learner_prefs)).sum()

    # number of goals attempted in G_learner and in G_solo
    n_g_2 = attempted[4]

    # number of goals attempted in G_learner and in G_teammates (and not in G_solo)
    n_g_3 = (attempted[:4] * shared_prefs).sum()

    # # number of goals attempted in G_learner and not in either G_solo or G_teammates
    # n_g_4 = (attempted[:4] * learner_prefs * (1 - teammate_prefs)).sum()

    n_g_4 = attempted[:4].sum() - (n_g_1 + n_g_3)

    # number of solo and cooperative goals attempted
    n_solo, n_coop = attempted[4], attempted[:4].sum()
    n_total = n_solo + n_coop

    vars = locals()
    return {k: vars[k] for k in goal_metric_keys}


def run_evals(
    rng: chex.PRNGKey,
    policy: Policy,
    env: ReachingEnv,
    default_env_params: ReachingEnvParams,
    init_extra: Dict[str, Any] = {},
    num_seeds=100,
    normalise_return=False,
) -> pd.DataFrame:

    eval_data = {"eval type": [], "return": []}
    eval_data = {**eval_data, **{k: [] for k in goal_metric_keys}}

    eval_npc_type_dists = get_type_dists(
        env.prefs_support, env.prefs_support[default_env_params.learner_agent_type]
    )

    for k in tqdm(eval_npc_type_dists.keys()):
        env_params = ReachingEnvParams(
            max_steps_in_episode=default_env_params.max_steps_in_episode,
            learner_agent_type=default_env_params.learner_agent_type,
            npc_type_dist=eval_npc_type_dists[k],
        )
        _, returns, metrics = evaluate(
            policy,
            rng,
            env,
            env_params,
            compute_episode_goal_stats,
            init_extra,
            num_seeds,
            max_steps_in_episode=env_params.max_steps_in_episode,
            normalise_return=normalise_return,
        )
        eval_data["eval type"].extend([k] * num_seeds)
        eval_data["return"].extend(returns.tolist())
        for k, v in metrics.items():
            eval_data[k].extend(v.tolist())

    avg_return = sum(eval_data["return"]) / len(eval_data["return"])
    eval_data["eval type"].append("average")
    eval_data["return"].append(avg_return)
    for k in goal_metric_keys:
        eval_data[k].append(0.0)

    return pd.DataFrame(eval_data)

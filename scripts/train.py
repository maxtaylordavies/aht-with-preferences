import argparse
import time
from typing import Any, Optional, Tuple
import os

import jax
import pandas as pd

from src.algos.ppo import train_ppo

# from src.algos.mcts import train_mcts
from src.algos.liam import train_liam
from src.algos.new_5 import train_new_5
from src.algos.new_6 import train_new_6
from src.environments import make, get_eval_func
from src.utils import save_training_outputs

parser = argparse.ArgumentParser(
    description="Train and evaluate a single model on a single environment"
)
parser.add_argument(
    "--project-dir",
    type=str,
    default=".",
    help="path to project directory on local scratch disk",
)
parser.add_argument("--algo", type=str, default="oracle", help="algorithm name")
parser.add_argument("--env", type=str, default="reaching", help="environment name")
parser.add_argument("--seed", type=int, default=0, help="random seed")
parser.add_argument(
    "--eval-eps",
    type=int,
    default=1000,
    help="number of episodes per eval type",
)

args = parser.parse_args()
print(f"Running with arguments: {args}")

key = jax.random.PRNGKey(args.seed)
env, env_params = make(args.env)
run_evals = get_eval_func(args.env)

RunFnOutput = Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[Any]]


def run_oracle() -> RunFnOutput:
    policy = lambda key, obs, state, extra: (env.oracle_policy(key, state, 0), extra)
    eval_df = run_evals(
        key, policy, env, env_params, num_seeds=args.eval_eps, normalise_return=True
    )
    return None, eval_df, None


def run_ppo() -> RunFnOutput:
    print("Beginning training...")
    start_time = time.time()
    policy, train_df = train_ppo(key, env, env_params)
    elapsed = time.time() - start_time
    print(f"Finished training in {elapsed:.2f}s")

    print("Running evals...")
    start_time = time.time()
    eval_df = run_evals(
        key, policy, env, env_params, num_seeds=args.eval_eps, normalise_return=True
    )
    elapsed = time.time() - start_time
    print(f"Finished evals in {elapsed:.2f}s")

    return train_df, eval_df, None


# def run_mcts() -> RunFnOutput:
#     print("Beginning training...")
#     start_time = time.time()
#     policy, train_df = train_mcts(key, env, env_params)
#     elapsed = time.time() - start_time
#     print(f"Finished training in {elapsed:.2f}s")

#     print("Running evals...")
#     start_time = time.time()
#     eval_df = run_evals(key, policy, env, env_params, num_seeds=args.eval_eps)
#     elapsed = time.time() - start_time
#     print(f"Finished evals in {elapsed:.2f}s")

#     return train_df, eval_df, None


def run_liam() -> RunFnOutput:
    print("Beginning training...")
    start_time = time.time()
    train_df, policy, init_extra, trajectories = train_liam(key, env, env_params)
    elapsed = time.time() - start_time
    print(f"Finished training in {elapsed:.2f}s")

    print("Running evals...")
    start_time = time.time()
    eval_df = run_evals(
        key,
        policy,
        env,
        env_params,
        init_extra,
        num_seeds=args.eval_eps,
        normalise_return=True,
    )
    elapsed = time.time() - start_time
    print(f"Finished evals in {elapsed:.2f}s")

    return train_df, eval_df, trajectories


def run_new_5() -> RunFnOutput:
    print("Beginning training...")
    start_time = time.time()
    # bc_path = f"{args.project_dir}/bc_params/{args.env}"
    # bc_path = f"/home/s2227283/projects/aht-with-preferences/bc_params/{args.env}"
    bc_path = f"/Users/max/Code/aht-with-preferences/bc_params/{args.env}"
    train_df, policy, init_extra, trajectories = train_new_5(
        key, env, env_params, bc_path
    )
    elapsed = time.time() - start_time
    print(f"Finished training in {elapsed:.2f}s")

    print("Running evals...")
    start_time = time.time()
    eval_df = run_evals(
        key,
        policy,
        env,
        env_params,
        init_extra,
        num_seeds=args.eval_eps,
        normalise_return=True,
    )
    elapsed = time.time() - start_time
    print(f"Finished evals in {elapsed:.2f}s")

    return train_df, eval_df, trajectories


def run_new_6() -> RunFnOutput:
    print("Beginning training...")
    start_time = time.time()
    # bc_path = f"{args.project_dir}/bc_params/{args.env}"
    bc_path = f"/Users/max/Code/aht-with-preferences/bc_params/{args.env}"
    train_df, policy, init_extra, trajectories = train_new_6(
        key, env, env_params, bc_path
    )
    elapsed = time.time() - start_time
    print(f"Finished training in {elapsed:.2f}s")

    print("Running evals...")
    start_time = time.time()
    eval_df = run_evals(
        key,
        policy,
        env,
        env_params,
        init_extra,
        num_seeds=args.eval_eps,
        normalise_return=True,
    )
    elapsed = time.time() - start_time
    print(f"Finished evals in {elapsed:.2f}s")

    return train_df, eval_df, trajectories


funcs = {
    "oracle": run_oracle,
    "ppo": run_ppo,
    # "mcts": run_mcts,
    "liam": run_liam,
    "new_5": run_new_5,
    "new_6": run_new_6,
}
train_df, eval_df, trajectories = funcs[args.algo]()
save_training_outputs(
    args.project_dir,
    args.env,
    args.algo,
    args.seed,
    train_df=train_df,
    eval_df=eval_df,
    trajectories=trajectories,
)

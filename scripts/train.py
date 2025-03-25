import argparse
import time
from typing import Optional, Tuple

import jax
import pandas as pd

from src.algos.ppo import train_ppo
from src.algos.mcts import train_mcts
from src.environments import make, get_eval_func
from src.utils import save_dataframes


parser = argparse.ArgumentParser(
    description="Train and evaluate a single model on a single environment"
)
parser.add_argument("--algo", type=str, default="oracle", help="algorithm name")
parser.add_argument("--env", type=str, default="reaching", help="environment name")
parser.add_argument("--seed", type=int, default=0, help="random seed")
parser.add_argument(
    "--eval-eps",
    type=int,
    default=500,
    help="number of episodes per eval type",
)

args = parser.parse_args()
print(f"Running with arguments: {args}")

key = jax.random.PRNGKey(args.seed)
env, env_params = make(args.env)
run_evals = get_eval_func(args.env)


def run_oracle() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    policy = lambda key, obs, state: env.reference_policy(key, state, 0)
    eval_df = run_evals(key, policy, env, env_params, num_seeds=args.eval_eps)
    return None, eval_df


def run_ppo() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    print("Beginning training...")
    start_time = time.time()
    policy, train_df = train_ppo(key, env, env_params)
    elapsed = time.time() - start_time
    print(f"Finished training in {elapsed:.2f}s")

    print("Running evals...")
    start_time = time.time()
    eval_df = run_evals(key, policy, env, env_params, num_seeds=args.eval_eps)
    elapsed = time.time() - start_time
    print(f"Finished evals in {elapsed:.2f}s")

    return train_df, eval_df


def run_mcts() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    print("Beginning training...")
    start_time = time.time()
    policy, train_df = train_mcts(key, env, env_params)
    elapsed = time.time() - start_time
    print(f"Finished training in {elapsed:.2f}s")

    print("Running evals...")
    start_time = time.time()
    eval_df = run_evals(key, policy, env, env_params, num_seeds=args.eval_eps)
    elapsed = time.time() - start_time
    print(f"Finished evals in {elapsed:.2f}s")

    return train_df, eval_df


funcs = {
    "oracle": run_oracle,
    "ppo": run_ppo,
    "mcts": run_mcts,
}
train_df, eval_df = funcs[args.algo]()
save_dataframes(args.env, args.algo, args.seed, train_df=train_df, eval_df=eval_df)

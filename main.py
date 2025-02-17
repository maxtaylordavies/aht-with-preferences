import os
from typing import Tuple

import jax
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.environments.lbf import compute_lbf_reference_returns

SEED = 0
N_REF_EPS = 100

sns.set_style("darkgrid")

algos = ["ppo"]
envs = ["lbf"]

reference_fns = {
    "lbf": compute_lbf_reference_returns
}

for env in envs:
    train_dfs, eval_dfs = [], []

    for algo in algos:
        data_dir = os.path.join("data", env, algo)
        train_df: pd.DataFrame = pd.read_pickle(os.path.join(data_dir, "train.pkl"))
        eval_df: pd.DataFrame = pd.read_pickle(os.path.join(data_dir, "eval.pkl"))
        train_df["algo"] = algo
        eval_df["algo"] = algo
        train_dfs.append(train_df)
        eval_dfs.append(eval_df)

    reference_df = None
    if env in reference_fns:
        reference_df = reference_fns[env](
            jax.random.PRNGKey(SEED), N_REF_EPS
        )
        reference_df["algo"] = "reference"
        eval_dfs.append(reference_df)

    train_df = pd.concat(train_dfs, ignore_index=True)
    eval_df = pd.concat(eval_dfs, ignore_index=True)

    fig, ax = plt.subplots()
    sns.lineplot(train_df, x="timestep", y="return", hue="algo")
    if reference_df is not None:
        tmp = reference_df[reference_df["eval type"] == "overall"]["return"].mean()
        plt.axhline(tmp, color="red", linestyle="--", label="reference")
    plt.show()

    fig, ax = plt.subplots()
    sns.barplot(eval_df, x="eval type", y="return", hue="algo")
    plt.show()

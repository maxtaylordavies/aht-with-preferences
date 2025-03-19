import os
from typing import Tuple

import jax
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

SEED = 0

sns.set_style("darkgrid")

algos = ["ppo", "mcts", "reference"]
envs = ["reaching"]

# reference_fns = {"lbf": compute_lbf_reference_returns}

for env in envs:
    train_dfs, eval_dfs = [], []

    for algo in algos:
        data_dir = os.path.join("data", env, algo)
        if algo != "reference":
            train_df: pd.DataFrame = pd.read_pickle(os.path.join(data_dir, "train.pkl"))
            train_df["algo"] = algo
            train_dfs.append(train_df)
        eval_df: pd.DataFrame = pd.read_pickle(os.path.join(data_dir, "eval.pkl"))
        eval_df["algo"] = algo
        eval_dfs.append(eval_df)

    if len(train_dfs) > 0:
        train_df = pd.concat(train_dfs, ignore_index=True)
        fig, ax = plt.subplots()
        sns.lineplot(train_df, x="timestep", y="return", hue="algo")
        # if reference_df is not None:
        #     tmp = reference_df[reference_df["eval type"] == "overall"]["return"].mean()
        #     plt.axhline(tmp, color="red", linestyle="--", label="reference")
        plt.show()

    if len(eval_dfs) > 0:
        eval_df = pd.concat(eval_dfs, ignore_index=True)

        # plot return
        fig, ax = plt.subplots()
        sns.barplot(eval_df, x="eval type", y="return", hue="algo")
        ax.set(xlabel="Evaluation setting", ylabel="Return", title="Return")
        plt.show()

        # plot proportion of attempted goals in Glearner
        fig, ax = plt.subplots()
        sns.barplot(eval_df, x="eval type", y="prop_in_g_learner", hue="algo")
        ax.set(
            xlabel="Evaluation setting",
            ylabel="Proportion",
            title=r"Proportion of attempted goals $\in G^\text{learner}$",
        )
        plt.show()

        # plot proportion of attempted goals that were feasible
        fig, ax = plt.subplots()
        sns.barplot(eval_df, x="eval type", y="prop_feasible", hue="algo")
        ax.set(
            xlabel="Evaluation setting",
            ylabel="Proportion",
            title=r"Proportion of attempted goals $\in$ feasible region",
        )
        plt.show()

        # plot proportion of attempted goals that were cooperative
        fig, ax = plt.subplots()
        sns.barplot(eval_df, x="eval type", y="prop_coop", hue="algo")
        ax.set(
            xlabel="Evaluation setting",
            ylabel="Proportion",
            title=r"Proportion of attempted goals $\notin G^\text{solo}$",
        )
        plt.show()

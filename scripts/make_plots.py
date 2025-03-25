import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

algos = ["ppo", "mcts", "oracle"]
envs = ["lbf"]

colors = ["#D93783", "#F77277", "#EC7924", "#DCBB28", "#33B171", "#45D1D8", "#2A56A2"]


def pointplot(data, x, y, hue, palette=None, fig_ax=None, despine=True, **kwargs):
    if fig_ax is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = fig_ax
    if palette is None:
        palette = sns.color_palette(colors[::-1])
    sns.pointplot(
        data=data,
        x=x,
        y=y,
        hue=hue,
        palette=palette,
        # marker="_",
        linestyle="none",
        dodge=0.4,
        markersize=8,
        # markeredgewidth=6,
        err_kws={"alpha": 0.7, "solid_capstyle": "butt"},
        ax=ax,
        legend=False,
        **kwargs,
    )
    sns.stripplot(
        data=data,
        x=x,
        y=y,
        hue=hue,
        palette=palette,
        alpha=0.3,
        dodge=True,
        ax=ax,
        legend=False,
        **kwargs,
    )
    if despine:
        sns.despine(fig, ax, bottom=True, left=True)
    return fig, ax


def save_fig(fig, filename, extensions=["svg", "png"]):
    fig.tight_layout()
    for ext in extensions:
        fig.savefig(
            os.path.join("plots", f"{filename}.{ext}"), dpi=300, bbox_inches="tight"
        )
    plt.close(fig)


for env in envs:
    train_dfs, eval_dfs = [], []
    for algo in algos:
        data_dir = os.path.join("data", env, algo)
        for seed_str in os.listdir(data_dir):
            if seed_str.startswith("."):
                continue
            if algo != "oracle":
                train_df = pd.read_pickle(os.path.join(data_dir, seed_str, "train.pkl"))
                train_df[["algo", "run_seed"]] = algo, int(seed_str)
                train_dfs.append(train_df)
            eval_df = pd.read_pickle(os.path.join(data_dir, seed_str, "eval.pkl"))
            eval_df[["algo", "run_seed"]] = algo, int(seed_str)
            print(f"{algo}, {seed_str}: {eval_df['return'].max():.2f}")
            eval_dfs.append(eval_df)

    train_df, eval_df = None, None
    if len(train_dfs) > 0:
        train_df = pd.concat(train_dfs, ignore_index=True)
    if len(eval_dfs) > 0:
        eval_df = pd.concat(eval_dfs, ignore_index=True)

    if train_df is not None:
        fig, ax = plt.subplots()
        sns.lineplot(
            train_df,
            x="timestep",
            y="return",
            hue="algo",
            palette=sns.color_palette(colors[::-1]),
        )
        if eval_df is not None:
            tmp = float(
                eval_df[
                    (eval_df["algo"] == "oracle") & (eval_df["eval type"] == "overall")
                ]["return"].mean()
            )
            plt.axhline(tmp, color="red", linestyle="--", label="oracle policy")
        ax.set(xlabel="Timestep", ylabel="Return", title="Mean return during training")
        sns.despine(fig, ax, bottom=True, left=True)
        save_fig(fig, "learning-curves")

    if eval_df is not None:
        # plot return
        fig, ax = pointplot(eval_df, "eval type", "return", "algo")
        ax.set(xticklabels=[x.capitalize() for x in eval_df["eval type"].unique()])
        ax.set(xlabel="", ylabel="Return", title="Mean episode return")
        save_fig(fig, "eval-returns")

        # eval_df has columns eval_type, algo, n_g_1, n_g_2, n_g_3, n_g_4, and some others
        # we want to create a new df which has columns eval_type, algo, g, n
        # where each row's n is one of n_g_1, n_g_2, n_g_3, n_g_4
        goal_df = (
            eval_df.melt(
                id_vars=["eval type", "algo", "run_seed"],
                value_vars=["n_g_1", "n_g_2", "n_g_3", "n_g_4"],
                var_name="goal_set",
                value_name="proportion",
            )
            .groupby(["eval type", "algo", "goal_set", "run_seed"])
            .mean()
            .reset_index()
        )

        # normalise the goal attempt distributions
        goal_df["proportion"] = goal_df["proportion"] / goal_df.groupby(
            ["eval type", "algo", "run_seed"]
        )["proportion"].transform("sum")

        # plot the goal attempt distributions
        eval_types = ["no overlap", "partial overlap", "full overlap"]
        fig, axs = plt.subplots(1, len(eval_types), figsize=(12, 4), sharey=True)
        for i, eval_type in enumerate(eval_types):
            pointplot(
                goal_df[goal_df["eval type"] == eval_type],
                "algo",
                "proportion",
                "goal_set",
                palette={
                    "n_g_1": "salmon",
                    "n_g_2": "mediumaquamarine",
                    "n_g_3": "mediumseagreen",
                    "n_g_4": "indianred",
                },
                fig_ax=(fig, axs[i]),
                order=algos,
            )
            axs[i].set(
                xlabel="",
                ylabel="Proportion",
                title=eval_type.capitalize(),
                ylim=(-0.05, 1.05),
            )
        fig.suptitle("Goal attempt distributions")
        save_fig(fig, "goal-distributions")

        # plot proportion of attempted goals that were worthwhile
        eval_df["prop_worthwhile"] = (eval_df["n_g_2"] + eval_df["n_g_3"]) / eval_df[
            "n_total"
        ]
        fig, ax = pointplot(eval_df, "eval type", "prop_worthwhile", "algo")
        ax.set(
            xlabel="",
            ylabel="Proportion",
            title=r"Proportion of attempted goals $\in$ 'worthwhile' region",
        )
        save_fig(fig, "worthwhile")

        # plot proportion of attempted goals that were cooperative
        eval_df["prop_coop"] = eval_df["n_coop"] / eval_df["n_total"]
        fig, ax = pointplot(eval_df, "eval type", "prop_coop", "algo")
        ax.set(
            xlabel="",
            ylabel="Proportion",
            title=r"Proportion of attempted goals $\notin G^\text{solo}$",
        )
        save_fig(fig, "cooperativity")

        # finally, we want to compute and plot the difference in cooperation proportion
        # between the 'full overlap' and 'no overlap' evaluation settings
        pivot_df = (
            eval_df.groupby(["algo", "eval type", "run_seed"])["prop_coop"]
            .mean()
            .reset_index()
            .pivot(index=["algo", "run_seed"], columns="eval type", values="prop_coop")
        )

        diff_df = pd.DataFrame(
            {
                "algo": pivot_df.index.get_level_values("algo"),
                "run_seed": pivot_df.index.get_level_values("run_seed"),
                "diff": pivot_df["full overlap"] - pivot_df["no overlap"],
            }
        ).reset_index(drop=True)

        fig, ax = pointplot(diff_df, "algo", "diff", hue=None, order=algos)
        ax.set(
            xlabel="",
            ylabel="Difference",
            title=r"Difference in cooperativity between 'full overlap' and 'no overlap' settings",
        )
        save_fig(fig, "cooperativity-difference")

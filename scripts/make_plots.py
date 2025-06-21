from collections import defaultdict
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import load_training_outputs

sns.set_style("whitegrid")
colors = ["xkcd:green blue", "xkcd:periwinkle", "xkcd:salmon", "xkcd:pumpkin", "black"]


def pointplot(
    data, x, y, hue, palette=None, fig_ax=None, despine=True, legend=False, **kwargs
):
    if fig_ax is None:
        fig, ax = plt.subplots(figsize=(6, 3.5))
    else:
        fig, ax = fig_ax
    if palette is None:
        palette = sns.color_palette(colors)
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
        legend=legend,
        **kwargs,
    )
    if despine:
        sns.despine(fig, ax, bottom=True, left=True)
    return fig, ax


def save_fig(fig, env_name, filename, extensions=["png"]):
    fig.tight_layout()
    dir = os.path.join("plots", env_name)
    if "/" in filename:
        dir = os.path.join(dir, *filename.split("/")[:-1])
        filename = filename.split("/")[-1]
    os.makedirs(dir, exist_ok=True)
    for ext in extensions:
        fig.savefig(
            os.path.join(dir, f"{filename}.{ext}"),
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def extra_new_6_plots(train_df, eval_df, trajectories, env_name):
    df = train_df[train_df["algo"] == "new_6"]

    fig, ax = plt.subplots()
    sns.lineplot(data=df, x="timestep", y="obs_err", ax=ax)
    ax.set(
        xlabel="Timestep",
        ylabel="Error",
        title="Teammate observation reconstruction error",
    )
    save_fig(fig, env_name, "new_6/obs_error")

    fig, ax = plt.subplots()
    sns.lineplot(data=df, x="timestep", y="action_acc", ax=ax)
    ax.set(
        xlabel="Timestep",
        ylabel="Accuracy",
        title="Teammate action prediction accuracy",
    )
    save_fig(fig, env_name, "new_6/action_acc")

    fig, ax = plt.subplots()
    sns.lineplot(data=df, x="timestep", y="type_acc", ax=ax)
    ax.set(
        xlabel="Timestep",
        ylabel="Accuracy",
        title="Teammate type reconstruction accuracy",
    )
    save_fig(fig, env_name, "new_6/type_acc")

    fig, ax = plt.subplots()
    sns.lineplot(data=df, x="timestep", y="setting_acc", ax=ax)
    ax.set(
        xlabel="Timestep",
        ylabel="Accuracy",
        title="Setting reconstruction accuracy",
    )
    save_fig(fig, env_name, "new_6/setting_acc")


def extra_liam_plots(train_df, eval_df, trajectories, env_name):
    df = train_df[train_df["algo"] == "liam"]

    fig, ax = plt.subplots()
    sns.lineplot(data=df, x="timestep", y="obs_err", ax=ax)
    ax.set(
        xlabel="Timestep",
        ylabel="Error",
        title="Teammate observation reconstruction error",
    )
    save_fig(fig, env_name, "liam/obs_error")

    fig, ax = plt.subplots()
    sns.lineplot(data=df, x="timestep", y="action_acc", ax=ax)
    ax.set(
        xlabel="Timestep",
        ylabel="Accuracy",
        title="Teammate action prediction accuracy",
    )
    save_fig(fig, env_name, "liam/action_acc")

    # fig, ax = plt.subplots()
    # sns.lineplot(data=df, x="timestep", y="type_acc", ax=ax)
    # ax.set(
    #     xlabel="Timestep",
    #     ylabel="Accuracy",
    #     title="Teammate type reconstruction accuracy",
    # )
    # save_fig(fig, env_name, "liam/type_acc")

    # fig, ax = plt.subplots()
    # sns.lineplot(data=df, x="timestep", y="setting_acc", ax=ax)
    # ax.set(
    #     xlabel="Timestep",
    #     ylabel="Accuracy",
    #     title="Setting reconstruction accuracy",
    # )
    # save_fig(fig, env_name, "liam/setting_acc")

    # env, _ = make(env_name)
    # if env_name == "reaching":
    #     timestep_mask = trajectories[0].timestep == 19
    #     embeddings = trajectories[0].embedding[timestep_mask]
    #     low_dim = tsnex.transform(embeddings, n_components=2)

    #     teammate_obses = trajectories[0].teammate_obs[timestep_mask]
    #     nearest_goals = jax.vmap(env.get_nearest_goal, in_axes=(0, None))(
    #         teammate_obses, 1
    #     )
    #     teammate_types = jax.vmap(env.get_type)(teammate_obses)
    #     settings = jax.vmap(env.get_setting)(teammate_obses)

    #     # do scatter plot
    #     fig, ax = plt.subplots()
    #     sns.scatterplot(
    #         x=low_dim[:, 0],
    #         y=low_dim[:, 1],
    #         hue=nearest_goals,
    #         alpha=0.7,
    #         ax=ax,
    #         palette=["red", "green", "blue", "yellow"],
    #         # legend=False,
    #     )
    #     legend = ax.legend()
    #     legend.set_title("Nearest goal")
    #     labels = [str(loc) for loc in env.goal_locs]
    #     for t, l in zip(legend.texts, labels):
    #         t.set_text(l)
    #     ax.set_title("End-of-episode embeddings coloured by teammate's nearest goal")
    #     save_fig(fig, env_name, "liam/embeddings-by-goal")

    #     x_min, x_max = 1.1 * low_dim[:, 0].min(), 1.1 * low_dim[:, 0].max()
    #     y_min, y_max = 1.1 * low_dim[:, 1].min(), 1.1 * low_dim[:, 1].max()
    #     fig, axs = plt.subplots(5, 3, figsize=(9, 12))
    #     for i in range(15):
    #         subset = low_dim[teammate_types == i + 1]
    #         if subset.shape[0] == 0:
    #             continue
    #         sns.scatterplot(
    #             x=subset[:, 0],
    #             y=subset[:, 1],
    #             alpha=0.7,
    #             ax=axs[i // 3, i % 3],
    #         )
    #         axs[i // 3, i % 3].set(
    #             title=str(env.prefs_support[i + 1]),
    #             xlim=(x_min, x_max),
    #             ylim=(y_min, y_max),
    #         )
    #         sns.despine(fig, axs[i // 3, i % 3])
    #     fig.suptitle(
    #         "End-of-episode embeddings separated by teammate's goal preferences"
    #     )
    #     save_fig(fig, env_name, "liam/embeddings-by-type")

    #     # do scatter plot
    #     fig, ax = plt.subplots()
    #     sns.scatterplot(
    #         x=low_dim[:, 0],
    #         y=low_dim[:, 1],
    #         hue=settings,
    #         alpha=0.7,
    #         ax=ax,
    #         palette=["red", "blue"],
    #     )
    #     ax.set(title="End-of-episode embeddings coloured by setting")
    #     save_fig(fig, env_name, "liam/embeddings-by-setting")
    # elif env_name == "lbf":
    #     timestep_mask = trajectories[0].timestep == 49
    #     embeddings = trajectories[0].embedding[timestep_mask]
    #     low_dim = tsnex.transform(embeddings, n_components=2)

    #     teammate_obses = trajectories[0].teammate_obs[timestep_mask]
    #     teammate_types = jax.vmap(env.get_type)(teammate_obses)
    #     settings = jax.vmap(env.get_setting)(teammate_obses)

    #     x_min, x_max = 1.1 * low_dim[:, 0].min(), 1.1 * low_dim[:, 0].max()
    #     y_min, y_max = 1.1 * low_dim[:, 1].min(), 1.1 * low_dim[:, 1].max()
    #     fig, axs = plt.subplots(4, 2, figsize=(9, 12))
    #     for i in range(7):
    #         subset = low_dim[teammate_types == i + 1]
    #         if subset.shape[0] == 0:
    #             continue
    #         sns.scatterplot(
    #             x=subset[:, 0],
    #             y=subset[:, 1],
    #             alpha=0.7,
    #             ax=axs[i // 2, i % 2],
    #         )
    #         axs[i // 2, i % 2].set(
    #             title=str(env.prefs_support[i + 1]),
    #             xlim=(x_min, x_max),
    #             ylim=(y_min, y_max),
    #         )
    #         sns.despine(fig, axs[i // 2, i % 2])
    #     fig.suptitle(
    #         "End-of-episode embeddings separated by teammate's fruit preferences"
    #     )
    #     save_fig(fig, env_name, "liam/embeddings-by-type")

    #     # do scatter plot
    #     fig, ax = plt.subplots()
    #     sns.scatterplot(
    #         x=low_dim[:, 0],
    #         y=low_dim[:, 1],
    #         hue=settings,
    #         alpha=0.7,
    #         ax=ax,
    #         palette=["red", "blue", "green"],
    #     )
    #     ax.set(title="End-of-episode embeddings coloured by setting")
    #     save_fig(fig, env_name, "liam/embeddings-by-setting")


algos = ["ppo", "liam", "new_5", "new_6", "oracle"]
algo_labels = {
    "ppo": "PPO",
    "liam": "LIAM",
    "new_5": "H1",
    "new_6": "H2",
    "oracle": "Oracle",
}
envs = ["reaching", "lbf"]
env_labels = {"reaching": "cooperative reaching", "lbf": "level-based foraging"}
extra_plot_fns = {"new_6": extra_new_6_plots, "liam": extra_liam_plots}

for env in envs:
    train_dfs, eval_dfs, trajectories = [], [], defaultdict(list)
    for algo in algos:
        data_dir = os.path.join("data", env, algo)
        for seed_str in os.listdir(data_dir):
            if seed_str.startswith("."):
                continue

            train_df, eval_df, trajs = load_training_outputs(
                ".", env, algo, int(seed_str)
            )
            skip = False
            if train_df is not None:
                train_df[["algo", "run_seed"]] = algo, int(seed_str)
                final_return_avg = train_df["return"].tail(10).values.mean()
                if final_return_avg <= 0.0:
                    print(
                        f"Skipping {algo} with seed {seed_str} on {env} due to nonpositive final return: {final_return_avg:.3f}"
                    )
                    skip = True
                else:
                    train_dfs.append(train_df)
            if eval_df is not None and not skip:
                eval_df[["algo", "run_seed"]] = algo, int(seed_str)
                eval_dfs.append(eval_df)
            if trajs is not None and not skip:
                trajectories[algo].append(trajs)

    train_df, eval_df = None, None
    if len(train_dfs) > 0:
        train_df = pd.concat(train_dfs, ignore_index=True)
    if len(eval_dfs) > 0:
        eval_df = pd.concat(eval_dfs, ignore_index=True)

    timesteps = train_df["timestep"].unique()
    train_df["timestep"] = train_df["timestep"].astype(int)
    train_df["timestep"] = train_df["timestep"] - (train_df["timestep"] % 500000)
    # for t in timesteps:
    #     print(t, int(t) - (int(t) % 500000))

    if train_df is not None:
        fig, ax = plt.subplots()
        sns.lineplot(
            train_df,
            x="timestep",
            y="return",
            hue="algo",
            palette=sns.color_palette(colors),
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
        save_fig(fig, env, "learning-curves")

    if eval_df is not None:
        eval_df = eval_df[eval_df["eval type"] != "overall"]

        if "new_5" in algos and "new_6" in algos:
            new_5_avg = eval_df[
                (eval_df["algo"] == "new_5") & (eval_df["eval type"] == "average")
            ]["return"].mean()
            new_6_avg = eval_df[
                (eval_df["algo"] == "new_6") & (eval_df["eval type"] == "average")
            ]["return"].mean()
            print(
                f"Difference in average return between new_5 and new_6: {new_6_avg - new_5_avg:.3f}"
            )

        # plot return
        fig, ax = pointplot(eval_df, "eval type", "return", "algo")
        ax.set(xticklabels=[x.capitalize() for x in eval_df["eval type"].unique()])
        ax.set(
            xlabel="",
            ylabel="",
            title=f"Mean episode return: {env_labels[env]}",
            ylim=(0.15, 1.05),
        )
        save_fig(fig, env, "eval-returns")

        # eval_df has columns eval_type, algo, n_g_1, n_g_2, n_g_3, n_g_4, and some others
        # we want to create a new df which has columns eval_type, algo, g, n
        # where each row's n is one of n_g_1, n_g_2, n_g_3, n_g_4
        eval_df = eval_df[eval_df["eval type"] != "average"]

        goal_df = (
            eval_df.melt(
                id_vars=["eval type", "algo", "run_seed"],
                value_vars=["n_g_1", "n_g_2", "n_g_3", "n_g_4"],
                var_name="goal_set",
                value_name="proportion",
            )
            .groupby(["eval type", "algo", "goal_set", "run_seed"])
            .sum()
            .reset_index()
        )

        # normalise the goal attempt distributions
        goal_df["proportion"] = goal_df["proportion"] / (
            goal_df.groupby(["eval type", "algo", "run_seed"])["proportion"].transform(
                "sum"
            )
            + 1e-8
        )

        # plot the goal attempt distributions
        eval_types = ["no overlap", "partial overlap", "full overlap"]
        goal_sets = ["n_g_1", "n_g_4", "n_g_2", "n_g_3"]
        fig, axs = plt.subplots(1, len(eval_types), figsize=(12, 3), sharey=True)
        for i, eval_type in enumerate(eval_types):
            df_ = goal_df[goal_df["eval type"] == eval_type][
                ["algo", "goal_set", "proportion"]
            ]
            grouped = df_.groupby(["algo", "goal_set"]).mean()
            prevs = np.array([0.0] * len(algos))
            for j, gs in enumerate(goal_sets):
                vals = np.array(
                    [grouped.loc[(algo, gs), "proportion"] for algo in algos]
                )
                axs[i].bar(
                    algos,
                    vals,
                    bottom=prevs.copy(),
                    color=[
                        "xkcd:coral",
                        "xkcd:dark pink",
                        "xkcd:greenish",
                        "xkcd:blue green",
                    ][j],
                    alpha=0.9,
                )
                prevs += vals.copy()
            labels = [algo_labels[algo] for algo in algos]
            axs[i].set(title=eval_type.capitalize(), xticklabels=labels)
            sns.despine(ax=axs[i], bottom=True, left=True)
        save_fig(fig, env, "goal-distributions")

        # plot proportion of attempted goals that were worthwhile
        worthwhile_df = []
        for algo in algos:
            for seed in goal_df["run_seed"].unique():
                for eval_type in eval_types:
                    try:
                        tmp = goal_df[
                            (goal_df["algo"] == algo)
                            & (goal_df["run_seed"] == seed)
                            & (goal_df["eval type"] == eval_type)
                        ]

                        prop_n_g_2 = tmp[tmp["goal_set"] == "n_g_2"][
                            "proportion"
                        ].values[0]
                        prop_n_g_3 = tmp[tmp["goal_set"] == "n_g_3"][
                            "proportion"
                        ].values[0]

                        worthwhile_df.append(
                            {
                                "algo": algo,
                                "eval type": eval_type,
                                "run_seed": seed,
                                "prop_worthwhile": prop_n_g_2 + prop_n_g_3,
                            }
                        )
                    except IndexError:
                        continue
        worthwhile_df = pd.DataFrame(worthwhile_df)

        fig, ax = pointplot(worthwhile_df, "eval type", "prop_worthwhile", "algo")
        ax.set(
            xlabel="",
            ylabel="",
            title=r"Proportion of attempted goals $\in$ 'worthwhile' region",
        )
        save_fig(fig, env, "worthwhile")

        # finally, we want to compute and plot the difference in cooperation proportion
        # between the 'full overlap' and 'no overlap' evaluation settings
        coop_df = []
        for algo in algos:
            for seed in eval_df["run_seed"].unique():
                props = []
                for eval_type in ["no overlap", "full overlap"]:
                    tmp = eval_df[
                        (eval_df["algo"] == algo)
                        & (eval_df["run_seed"] == seed)
                        & (eval_df["eval type"] == eval_type)
                    ]
                    if tmp.empty:
                        continue
                    n_coop = tmp["n_coop"].values.sum()
                    n_total = tmp["n_total"].values.sum()
                    props.append(n_coop / n_total)
                if len(props) == 2:
                    coop_df.append(
                        {
                            "algo": algo,
                            "run_seed": seed,
                            "eval type": eval_type,
                            "diff": props[1] - props[0],
                        }
                    )
        coop_df = pd.DataFrame(coop_df)

        fig, ax = pointplot(coop_df, "algo", "diff", hue=None, order=algos)
        ax.set(
            xlabel="",
            ylabel="",
            title=r"Difference in cooperativity between 'full overlap' and 'no overlap' settings",
        )
        save_fig(fig, env, "cooperativity-difference")

    for algo in algos:
        if algo in extra_plot_fns:
            try:
                extra_plot_fns[algo](train_df, eval_df, trajectories[algo], env)
            except:
                pass

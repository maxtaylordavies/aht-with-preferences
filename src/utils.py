import os
import time
from typing import Any, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
import pandas as pd
import pickle


# Copied from: https://stackoverflow.com/a/23689767
class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def replace(self, **kwargs):
        return dotdict({**self, **kwargs})


def to_one_hot(x: int, d: int) -> chex.Array:
    """
    Convert an integer to a one-hot vector of dimension d
    """
    v = jnp.zeros(d)
    return v.at[x].set(1.0)


def from_one_hot(v: chex.Array) -> chex.Array:
    """
    Convert a one-hot vector to an integer
    """
    return jnp.argmax(v)


def visualise_episode(key, env, env_params, policy_fn, sleep=0.5):
    obs, state = env.reset(key, env_params)
    for t in range(env_params.max_steps_in_episode):
        env.render(state)
        time.sleep(sleep)
        key, key_policy, key_step = jax.random.split(key, 3)
        action = policy_fn(key_policy, obs, state)
        obs, state, reward, done, info = env.step(key_step, state, action, env_params)
        if done:
            break


def save_training_outputs(
    project_dir: str,
    env_name: str,
    algo_name: str,
    seed: int,
    train_df: Optional[pd.DataFrame],
    eval_df: Optional[pd.DataFrame],
    trajectories: Optional[Any],
):
    out_dir = os.path.join(project_dir, "data", env_name, algo_name, str(seed))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if train_df is not None:
        train_df.to_pickle(os.path.join(out_dir, "train.pkl"))
    if eval_df is not None:
        eval_df.to_pickle(os.path.join(out_dir, "eval.pkl"))
    if trajectories is not None:
        with open(os.path.join(out_dir, "trajectories.pkl"), "wb") as f:
            pickle.dump(trajectories, f)


def load_training_outputs(
    project_dir: str, env_name: str, algo_name: str, seed: int
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[Any]]:
    out_dir = os.path.join(project_dir, "data", env_name, algo_name, str(seed))
    train_df, eval_df, trajectories = None, None, None
    if os.path.exists(out_dir):
        if os.path.exists(os.path.join(out_dir, "train.pkl")):
            train_df = pd.read_pickle(os.path.join(out_dir, "train.pkl"))
        if os.path.exists(os.path.join(out_dir, "eval.pkl")):
            eval_df = pd.read_pickle(os.path.join(out_dir, "eval.pkl"))
        if os.path.exists(os.path.join(out_dir, "trajectories.pkl")):
            with open(os.path.join(out_dir, "trajectories.pkl"), "rb") as f:
                trajectories = pickle.load(f)
    return train_df, eval_df, trajectories


def get_type_dists(
    prefs_support: chex.Array, learner_prefs: chex.Array
) -> dict[str, chex.Array]:
    dists = {
        k: jnp.zeros(prefs_support.shape[0])
        for k in ["no overlap", "partial overlap", "full overlap"]
    }

    for i in range(prefs_support.shape[0]):
        tmp = (prefs_support[i] * learner_prefs).sum()
        if tmp == 0:
            dists["no overlap"] = dists["no overlap"].at[i].set(1.0)
        elif tmp == learner_prefs.sum():
            dists["full overlap"] = dists["full overlap"].at[i].set(1.0)
        else:
            dists["partial overlap"] = dists["partial overlap"].at[i].set(1.0)

    dists["overall"] = (
        dists["no overlap"] + dists["partial overlap"] + dists["full overlap"]
    )

    for k, v in dists.items():
        # v = v.at[0].set(0.0)
        dists[k] = v / v.sum()

    print(dists)

    return dists

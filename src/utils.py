import os
import time

import chex
import jax
import jax.numpy as jnp
import pandas as pd

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

def save_returns_data(env_name: str, algo_name: str, train_df: pd.DataFrame, eval_df: pd.DataFrame):
    out_dir = os.path.join("data", env_name, algo_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    train_df.to_pickle(os.path.join(out_dir, "train.pkl"))
    eval_df.to_pickle(os.path.join(out_dir, "eval.pkl"))

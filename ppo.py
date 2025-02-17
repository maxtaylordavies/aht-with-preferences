import time
from itertools import product

import chex
import jax
import jax.numpy as jnp
from rejax import PPO
import pandas as pd
from tqdm import tqdm

from src.environments.lbf import (
    LBFEnv,
    LBFEnvParams,
    NPCPolicyParams,
    make,
    run_evals
)
from src.utils import save_returns_data

rng = jax.random.PRNGKey(0)
env, default_env_params = make()

algo = PPO.create(
    env=env,
    env_params=default_env_params,
    total_timesteps=2e6,
    eval_freq=5000,
    num_envs=16,
    num_steps=128,
    num_epochs=1,
    num_minibatches=8,
    max_grad_norm=0.5,
    learning_rate=2.5e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_eps=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
)

print("Beginning training...")
start_time = time.time()
train_fn = jax.jit(algo.train)
train_state, evaluation = jax.block_until_ready(train_fn(rng))

elapsed = time.time() - start_time
sps = algo.total_timesteps / elapsed
print(f"Finished training in {elapsed:.2f}s ({sps:.2f} steps/s)")

_, ep_returns = evaluation
mean_returns = ep_returns.mean(axis=1)
train_df = pd.DataFrame({
    "timestep": jnp.linspace(0, algo.total_timesteps, len(mean_returns)),
    "return": mean_returns,
})

# run evals
print("Running evals...")
start_time = time.time()
policy = jax.jit(algo.make_act(train_state))
eval_df = run_evals(rng, policy, env, default_env_params)

elapsed = time.time() - start_time
print(f"Finished evals in {elapsed:.2f}s")

# save data
save_returns_data("lbf", "ppo", train_df, eval_df)

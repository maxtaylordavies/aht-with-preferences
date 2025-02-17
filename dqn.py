import time
from itertools import product

import chex
import jax
import jax.numpy as jnp
from rejax import DQN
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




# agent_kwargs:
# activation: swish
# num_envs: 128
# num_epochs: 128
# buffer_size: 131_072
# fill_buffer: 8_192
# batch_size: 1_024
# learning_rate: 0.0003
# max_grad_norm: 1
# total_timesteps: 1_048_576
# eval_freq: 131_072
# gamma: 0.9
# eps_start: 1.0
# eps_end: 0.05
# exploration_fraction: 0.1
# target_update_freq: 4096
# ddqn: true
# normalize_observations: true

algo = DQN.create(
    env=env,
    env_params=default_env_params,
    # total_timesteps=1_048_576,
    total_timesteps=262144,
    # eval_freq=131_072,
    eval_freq=2e4,
    num_envs=16,
    # num_epochs=128,
    num_epochs=128,
    # buffer_size=131_072,
    buffer_size=32768,
    # fill_buffer=8_192,
    fill_buffer=2048,
    # batch_size=1_024,
    batch_size=256,
    learning_rate=0.0003,
    max_grad_norm=1.0,
    gamma=0.9,
    eps_start=1.0,
    eps_end=0.05,
    exploration_fraction=0.1,
    target_update_freq=4096,
    ddqn=True,
    normalize_observations=True,
    # gae_lambda=0.95,
    # clip_eps=0.2,
    # ent_coef=0.01,
    # vf_coef=0.5,
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
save_returns_data("lbf", "dqn", train_df, eval_df)

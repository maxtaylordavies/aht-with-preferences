import time

import jax
import jax.numpy as jnp
from rejax import PPO
import pandas as pd

from src.environments.reaching import make, run_evals
from src.evaluate import Policy
from src.utils import save_dataframes

rng = jax.random.PRNGKey(0)
env, default_env_params = make()

algo = PPO.create(
    env=env,
    env_params=default_env_params,
    total_timesteps=1e5,
    eval_freq=1000,
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
train_df = pd.DataFrame(
    {
        "timestep": jnp.linspace(0, algo.total_timesteps, len(mean_returns)),
        "return": mean_returns,
    }
)

# run evals
print("Running evals...")
start_time = time.time()
_policy = algo.make_act(train_state)
policy: Policy = jax.jit(lambda key, obs, state: _policy(obs, key))
eval_df = run_evals(rng, policy, env, default_env_params, num_seeds=500)

elapsed = time.time() - start_time
print(f"Finished evals in {elapsed:.2f}s")

# save data
save_dataframes(env.name, "ppo", train_df, eval_df)

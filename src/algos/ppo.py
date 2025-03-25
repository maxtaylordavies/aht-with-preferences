from typing import Tuple

import chex
from gymnax.environments.environment import Environment, EnvParams
import jax
import jax.numpy as jnp
from rejax import PPO
import pandas as pd

from src.evaluate import Policy


def train_ppo(
    key: chex.PRNGKey, env: Environment, env_params: EnvParams
) -> Tuple[Policy, pd.DataFrame]:
    algo = PPO.create(
        env=env,
        env_params=env_params,
        total_timesteps=1e6,
        eval_freq=5e4,
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

    train_fn = jax.jit(algo.train)
    train_state, evaluation = jax.block_until_ready(train_fn(key))

    _, ep_returns = evaluation
    mean_returns = ep_returns.mean(axis=1)
    train_df = pd.DataFrame(
        {
            "timestep": jnp.linspace(0, algo.total_timesteps, len(mean_returns)),
            "return": mean_returns,
        }
    )

    _policy = algo.make_act(train_state)
    policy: Policy = jax.jit(lambda key, obs, state: _policy(obs, key))

    return policy, train_df

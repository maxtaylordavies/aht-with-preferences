from collections import Counter
import time

import jax
import jax.numpy as jnp
from orbax.checkpoint import PyTreeCheckpointer
from rejax.networks import DiscretePolicy
from flax import linen as nn
from tqdm import tqdm

from src.environments import make
from src.algos.new_5 import NEW5

rng = jax.random.PRNGKey(0)
env, env_params = make("lbf")
n_actions = env.action_space().n

# checkpointer = PyTreeCheckpointer()
# ckpt = checkpointer.restore("/Users/max/Code/aht-with-preferences/new_5_params/lbf")

# agent = NEW5.create(
#     env=env,
#     env_params=env_params,
#     total_timesteps=int(5e6),
#     eval_freq=5e5,
#     num_envs=16,
#     num_steps=1024,
#     num_epochs=2,
#     num_minibatches=8,
#     max_grad_norm=0.5,
#     learning_rate=3e-4,
#     gamma=0.99,
#     gae_lambda=0.95,
#     clip_eps=0.2,
#     ent_coef=0.01,
#     vf_coef=0.5,
#     normalize_observations=False,
#     normalize_rewards=False,
#     bc_net_lr=5e-3,
#     goal_actor_params=ckpt["goal_actor_params"],
#     goal_critic_params=ckpt["goal_critic_params"],
#     bc_net_params=ckpt["bc_net_params"],
# )
# ts = agent.init_state(rng)
# policy, _ = agent.make_policy(ts)
# policy = jax.jit(policy)

# goal_counts = {g: 0 for g in range(6)}

# fruit_names = ["apple", "orange", "pear"]
# goal_attempt_counts = {}
# for fn in fruit_names:
#     for lvl in [1, 2]:
#         goal_attempt_counts[f"{fn}_{lvl}"] = 0

# fruit_counts = {fn: 0 for fn in fruit_names}

obs, env_state = env.reset(rng, env_params)

print(env_state)

print(obs)
print(obs.shape)

# obs_idxs = jnp.array([[i * 6, (i * 6) + 1] for i in range(6)]).flatten()
# print(obs[obs_idxs])

# for t in tqdm(range(30000)):
#     # env.render(env_state)
#     # time.sleep(0.1)

#     rng, rng_act, rng_step = jax.random.split(rng, 3)

#     action, extra = policy(rng_act, obs, env_state, {})
#     # goal_counts[int(extra["goal"])] += 1

#     obs, new_env_state, _, done, info = env.step(
#         rng_step, env_state, action, env_params
#     )
#     if done:
#         types, levels = env_state.fruit_types, env_state.fruit_levels
#         for i, g in enumerate(info["attempted"]):
#             if g == 1:
#                 goal_key = f"{fruit_names[types[i]]}_{levels[i]}"
#                 goal_attempt_counts[goal_key] += 1

#         for ft in types:
#             fruit_counts[fruit_names[ft]] += 1

#     env_state = new_env_state

# print("Goals attempted:", goal_attempt_counts)
# print("Fruit counts:", fruit_counts)

# n_g_1 = goal_attempt_counts["pear_1"] + goal_attempt_counts["pear_2"]
# n_g_2 = goal_attempt_counts["apple_1"] + goal_attempt_counts["orange_1"]
# n_g_4 = goal_attempt_counts["apple_2"] + goal_attempt_counts["orange_2"]
# total = n_g_1 + n_g_2 + n_g_4
# print(f"n_g_1: {n_g_1 / total}, n_g_2: {n_g_2 / total}, n_g_4: {n_g_4 / total}")

# print("Goal counts:", goal_counts)

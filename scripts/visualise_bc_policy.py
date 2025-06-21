from collections import Counter
import time

import jax
import jax.numpy as jnp
from orbax.checkpoint import PyTreeCheckpointer
from rejax.networks import DiscretePolicy
from flax import linen as nn
from tqdm import tqdm

from src.environments import make

rng = jax.random.PRNGKey(2)
env, env_params = make("lbf")
n_actions = env.action_space().n

checkpointer = PyTreeCheckpointer()
ckpt = checkpointer.restore("/Users/max/Code/aht-with-preferences/bc_params/lbf")
policy_params = ckpt["params"]

policy_net = DiscretePolicy(
    n_actions,
    hidden_layer_sizes=(64, 64),
    activation=nn.swish,
)

obs_idxs = jnp.array([[i * 4, (i * 4) + 1] for i in range(6)]).flatten()

goal = 1
goals_attempted = []
obs, env_state = env.reset(rng, env_params)
for t in tqdm(range(3000)):
    # env.render(env_state)
    # time.sleep(0.1)

    rng, rng_act, rng_step = jax.random.split(rng, 3)

    z = jax.nn.one_hot(goal, 5)
    input = jnp.concatenate([obs[obs_idxs] / 7.0, z, jnp.array([0.0])])
    input = jnp.expand_dims(input, axis=0)  # Add batch dimension

    action, _, _ = policy_net.apply(policy_params, input, rng)
    action = action[0]
    # action = 0

    obs, env_state, _, done, info = env.step(rng_step, env_state, action, env_params)
    if done:
        for i, g in enumerate(info["attempted"]):
            if g == 1:
                goals_attempted.append(i)

print("Goals attempted:", Counter(goals_attempted))

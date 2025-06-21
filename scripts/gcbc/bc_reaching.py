from collections import Counter, defaultdict
import functools
import os
import shutil

import chex
import jax
from flax import linen as nn
from flax.training.train_state import TrainState
from flax import struct
from jax import numpy as jnp
import numpy as np
import optax
import orbax.checkpoint
from tqdm import tqdm
import matplotlib.pyplot as plt
from rejax.networks import DiscretePolicy
import matplotlib.pyplot as plt
import seaborn as sns

from src.environments.reaching import make


class Trajectory(struct.PyTreeNode):
    timestep: chex.Array
    obs: chex.Array
    action: chex.Array
    # in_goal: chex.Array
    done: chex.Array


class Minibatch(struct.PyTreeNode):
    obs: chex.Array
    action: chex.Array
    # goal: chex.Array


rng = jax.random.PRNGKey(2)
n_trajs, traj_len = int(1e4), 20
n_envs, n_steps = 20, n_trajs * traj_len

env, env_params = make()
obs_dim = env.observation_space(env_params).shape[0]
n_actions = env.action_space(env_params).n
n_goals = 6

vmapped_reset = jax.vmap(env.reset, in_axes=(0, None))
vmapped_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))

expert_policy = lambda key, state, idx: env.heuristic_policy(key, state, idx)[0]
vmapped_policy = jax.vmap(expert_policy, in_axes=(0, 0, None))


def pass_through(x: jnp.ndarray, fn) -> jnp.ndarray:
    # computes (x - x) + fn(x)
    # By Sterbenz lemma, and ieee754 zero arithmatic, this is exactly fn(x).
    return x - jax.lax.stop_gradient(x) + jax.lax.stop_gradient(fn(x))


def gumbel_softmax(rng, logits, tau=1.0, hard=True):
    gumbel_noise = -jnp.log(
        -jnp.log(jax.random.uniform(rng, logits.shape) + 1e-20) + 1e-20
    )
    y = nn.softmax((logits + gumbel_noise) / tau)

    if hard:
        # Straight-through estimator
        fn = lambda y: jax.nn.one_hot(jnp.argmax(y, axis=-1), logits.shape[-1])
        y = pass_through(y, fn)

    return y


class DiscreteEncoder(nn.Module):
    hidden_dim: int = 64
    output_dim: int = 64

    @nn.compact
    def __call__(self, x, tau, rng):
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)
        logits = nn.Dense(features=self.output_dim)(x)
        probs = jax.nn.softmax(logits, axis=-1)
        samples = gumbel_softmax(rng, logits, tau=tau, hard=True)
        return samples, probs, logits


policy_net = DiscretePolicy(
    n_actions,
    hidden_layer_sizes=(64, 64),
    activation=nn.swish,
)


def create_train_state(rng, hidden_dim, learning_rate, n_envs):
    """Creates initial training state."""
    trajectory_dim = 2 * traj_len

    encoder = DiscreteEncoder(hidden_dim=hidden_dim, output_dim=n_goals)

    rng, rng_encoder, rng_policy = jax.random.split(rng, 3)

    init_enc_input = jnp.empty([1, trajectory_dim])
    init_policy_input = jnp.empty([1, 2 + n_goals])
    init_tau = 1.0

    encoder_params = encoder.init(rng_encoder, init_enc_input, init_tau, rng_encoder)
    policy_params = policy_net.init(rng_policy, init_policy_input, rng_policy)

    tx = optax.chain(
        optax.clip_by_global_norm(1.0), optax.adam(learning_rate=learning_rate)
    )

    encoder_ts = TrainState.create(apply_fn=encoder.apply, params=encoder_params, tx=tx)
    policy_ts = TrainState.create(apply_fn=(), params=policy_params, tx=tx)

    obs, env_state = vmapped_reset(jax.random.split(rng, n_envs), env_params)

    ts = {
        "encoder_ts": encoder_ts,
        "policy_ts": policy_ts,
        "rng": rng,
        "tau": init_tau,
        "env_state": env_state,
        "last_obs": obs,
        "last_timestep": jnp.zeros(n_envs, dtype=jnp.int32),
        "global_step": 0,
        "last_done": jnp.zeros(n_envs, dtype=bool),
    }

    cls_name = "CustomTrainState"
    state = {k: struct.field(pytree_node=True) for k in ts.keys()}
    state_hints = {k: type(v) for k, v in ts.items()}
    d = {**state, "__annotations__": state_hints}
    clz = type(cls_name, (struct.PyTreeNode,), d)
    return clz(**ts)


def embed(encode_fn, params, traj, tau, rng):
    z, probs, logits = encode_fn(params, traj, tau, rng)
    z = z.squeeze()
    return z, probs, logits


def get_policy_inputs(obss, goal_idxs):
    goal_vecs = jax.nn.one_hot(goal_idxs, n_goals)
    return jnp.concatenate([obss, goal_vecs], axis=-1)


@functools.partial(jax.jit, static_argnums=(1, 2))
def collect_trajectories(ts, n_envs, n_steps):
    def env_step(ts, unused):
        # Get keys for sampling action and stepping environment
        rng, new_rng = jax.random.split(ts.rng)
        ts = ts.replace(rng=rng)
        rng_action, rng_step = jax.random.split(new_rng, 2)
        rng_actions = jax.random.split(rng_action, n_envs)
        rng_steps = jax.random.split(rng_step, n_envs)

        action = vmapped_policy(rng_actions, ts.env_state, 0)

        # Step environment
        t = vmapped_step(rng_steps, ts.env_state, action, env_params)
        next_obs, env_state, _, done, info = t
        timestep = jnp.where(done, 0, ts.last_timestep + 1)

        # Return updated runner state and transition
        transition = Trajectory(
            ts.last_timestep,
            ts.last_obs,
            action,
            done,
        )
        ts = ts.replace(
            env_state=env_state,
            last_obs=next_obs,
            last_done=done,
            last_timestep=timestep,
            global_step=ts.global_step + n_envs,
        )
        return ts, transition

    ts, trajectories = jax.lax.scan(env_step, ts, None, n_steps)
    trajectories = jax.tree_util.tree_map(lambda x: x.swapaxes(0, 1), trajectories)
    return ts, trajectories


def normalize(x):
    return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def get_enc_input(traj):
    enc_input = traj.obs[:, :2].reshape(-1)
    return jnp.expand_dims(enc_input, axis=0)


@jax.jit
def update(ts, mb):
    def loss_fn(enc_params, policy_params):
        def traj_loss_fn(traj):
            enc_input = get_enc_input(traj)

            z, _, _ = ts.encoder_ts.apply_fn(enc_params, enc_input, ts.tau, ts.rng)
            z = z.squeeze()
            # z = jnp.zeros_like(z)  # For simplicity, we use a zero vector here

            def timestep_loss(t):
                policy_input = jnp.concatenate([traj.obs[t, :2], z])
                policy_input = jnp.expand_dims(
                    policy_input, axis=0
                )  # Add batch dimension
                true_action = jnp.expand_dims(traj.action[t], axis=0)
                log_probs, _ = policy_net.apply(
                    policy_params, policy_input, true_action, method="log_prob_entropy"
                )
                return -log_probs.mean()

            losses = jax.vmap(timestep_loss)(jnp.arange(10))
            return losses.mean()

        losses = jax.vmap(traj_loss_fn)(mb)
        return losses.mean()

    loss_val, grads = jax.value_and_grad(loss_fn, argnums=(0, 1))(
        ts.encoder_ts.params, ts.policy_ts.params
    )

    new_encoder_ts = ts.encoder_ts.apply_gradients(grads=grads[0])
    new_policy_ts = ts.policy_ts.apply_gradients(grads=grads[1])

    new_ts = ts.replace(
        encoder_ts=new_encoder_ts,
        policy_ts=new_policy_ts,
        global_step=ts.global_step + 1,
    )

    return new_ts, loss_val


@jax.jit
def evaluate(ts, mb):
    def traj_eval(traj):
        enc_input = get_enc_input(traj)

        z, _, _ = ts.encoder_ts.apply_fn(
            ts.encoder_ts.params, enc_input, ts.tau, ts.rng
        )
        z = z.squeeze()
        # z = jnp.zeros_like(z)

        def timestep_eval(t):
            policy_input = jnp.concatenate([traj.obs[t, :2], z])
            policy_input = jnp.expand_dims(policy_input, axis=0)  # Add batch dimension
            action, _, _ = policy_net.apply(ts.policy_ts.params, policy_input, ts.rng)
            return (action.squeeze() == traj.action[t]).astype(jnp.float32)

        results = jax.vmap(timestep_eval)(jnp.arange(10))
        return results.mean()

    return jax.vmap(traj_eval)(mb).mean()


def make_train_test_split(batch, frac=0.8):
    n_train = int(frac * batch.obs.shape[0])
    train_batch = jax.tree_util.tree_map(lambda x: x[:n_train], batch)
    test_batch = jax.tree_util.tree_map(lambda x: x[n_train:], batch)
    return train_batch, test_batch


ts = create_train_state(rng, 128, 5e-3, n_envs)
ts, batch = collect_trajectories(ts, n_envs, n_steps)

batch = jax.tree_util.tree_map(
    lambda x: x.reshape(n_envs * n_trajs, traj_len, -1).squeeze(), batch
)

train_batch, test_batch = make_train_test_split(batch)

print("Training batch shape:", train_batch.obs.shape)
print("Test batch shape:", test_batch.obs.shape)

# actions_list = train_batch.action.reshape(-1).tolist()
# action_counts = Counter(actions_list)
# print("Action counts in training batch:")
# for action, count in action_counts.items():
#     print(f"Action {action}: {count} times")


def sample_minibatch(rng, batch, batch_size):
    indices = jax.random.randint(
        rng, shape=(batch_size,), minval=0, maxval=batch.obs.shape[0]
    )
    return Minibatch(
        obs=batch.obs[indices] / 8.0,
        action=batch.action[indices],
    )


losses, train_accs, test_accs = [], [], []

for i in tqdm(range(int(2.5e3))):
    rng, new_rng = jax.random.split(ts.rng)
    ts = ts.replace(rng=new_rng)

    mb = sample_minibatch(rng, train_batch, 256)

    ts, loss_val = update(ts, mb)
    losses.append(float(loss_val))
    if i % 200 == 0:
        train_acc = evaluate(ts, mb)
        test_mb = sample_minibatch(rng, test_batch, 256)
        test_acc = evaluate(ts, test_mb)
        train_accs.append(float(train_acc))
        test_accs.append(float(test_acc))
        tqdm.write(
            f"Step {i}, Loss: {loss_val:.4f}, Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}"
        )

z_dict = defaultdict(list)
test_mb = sample_minibatch(rng, test_batch, 1000)
for i in tqdm(range(1000)):
    traj = jax.tree_util.tree_map(lambda x: x[i], test_mb)
    enc_input = get_enc_input(traj)
    z, _, _ = ts.encoder_ts.apply_fn(ts.encoder_ts.params, enc_input, ts.tau, ts.rng)
    z = int(z.squeeze().argmax())

    goal = traj.obs[-1, :2] * 8.0  # Scale back to original range
    goal_str = f"{int(goal[0])}_{int(goal[1])}"
    z_dict[goal_str].append(z)

for goal_str, z_values in z_dict.items():
    counts = Counter(z_values)
    print(f"Goal {goal_str}: {dict(counts)}")

# fig, ax = plt.subplots(1, 2, figsize=(12, 6))
# sns.lineplot(x=range(len(losses)), y=losses, ax=ax[0])
# sns.lineplot(x=range(len(test_accs)), y=test_accs, ax=ax[1])
# ax[0].set_title("Loss")
# ax[1].set_title("Accuracy")
# plt.tight_layout()
# plt.show()

# Save the model parameters
# fp = "/home/s2227283/projects/aht-with-preferences/bc_params/reaching"
fp = "/Users/max/Code/aht-with-preferences/bc_params/reaching"
if os.path.exists(fp):
    shutil.rmtree(fp)

checkpointer = orbax.checkpoint.PyTreeCheckpointer()
ckpt = {"params": ts.policy_ts.params}
checkpointer.save(fp, ckpt)

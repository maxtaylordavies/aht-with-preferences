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

from src.environments.lbf import make


class Trajectory(struct.PyTreeNode):
    timestep: chex.Array
    obs: chex.Array
    action: chex.Array
    goal: chex.Array
    done: chex.Array


class Batch(struct.PyTreeNode):
    # timestep: chex.Array
    obs: chex.Array
    action: chex.Array
    goal: chex.Array
    mask: chex.Array


rng = jax.random.PRNGKey(0)
n_envs, n_steps, traj_len = 30, int(1e5), 7
n_iter = int(2e4)
# n_envs, n_steps = 20, n_trajs * traj_len

env, env_params = make()
# obs_dim = env.observation_space(env_params).shape[0]
obs_dim = 12
n_actions = env.action_space(env_params).n
n_goals = 5

obs_idxs = jnp.array([[i * 6, (i * 6) + 1] for i in range(6)]).flatten()

vmapped_reset = jax.vmap(env.reset, in_axes=(0, None))
vmapped_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))
vmapped_policy = jax.vmap(env.oracle_policy, in_axes=(0, 0, None))


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
        # return logits, None, None
        probs = jax.nn.softmax(logits, axis=-1)
        # return probs, logits, None
        samples = gumbel_softmax(rng, logits, tau=tau, hard=True)
        return samples, probs, logits


class ObsDecoder(nn.Module):
    hidden_dim: int = 64
    output_dim: int = 12

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)
        logits = nn.Dense(features=self.output_dim)(x)
        return logits


policy_net = DiscretePolicy(
    n_actions,
    hidden_layer_sizes=(64, 64),
    activation=nn.swish,
)


def create_train_state(rng, hidden_dim, enc_lr, policy_lr, n_envs):
    """Creates initial training state."""
    # trajectory_dim = obs_dim + (2 * (traj_len - 1)) + 1
    trajectory_dim = 24

    encoder = DiscreteEncoder(hidden_dim=hidden_dim, output_dim=n_goals)
    aux_decoder = ObsDecoder(hidden_dim=32, output_dim=n_goals)

    rng, rng_encoder, rng_policy = jax.random.split(rng, 3)

    init_enc_input = jnp.empty([1, trajectory_dim])
    init_policy_input = jnp.empty([1, obs_dim + n_goals + 1])
    init_aux_dec_input = jnp.empty([1, obs_dim + n_goals])
    init_tau = 1.0

    encoder_params = encoder.init(rng_encoder, init_enc_input, init_tau, rng_encoder)
    policy_params = policy_net.init(rng_policy, init_policy_input, rng_policy)
    aux_decoder_params = aux_decoder.init(rng_policy, init_aux_dec_input)

    # enc_schedule = optax.linear_schedule(
    #     init_value=enc_lr, end_value=0.0, transition_steps=n_iter // 2
    # )
    # policy_schedule = optax.linear_schedule(
    #     init_value=policy_lr, end_value=0.0, transition_steps=n_iter
    # )

    enc_tx = optax.chain(
        optax.clip_by_global_norm(1.0), optax.adam(learning_rate=enc_lr)
    )
    policy_tx = optax.chain(
        optax.clip_by_global_norm(1.0), optax.adam(learning_rate=policy_lr)
    )

    encoder_ts = TrainState.create(
        apply_fn=encoder.apply, params=encoder_params, tx=enc_tx
    )
    policy_ts = TrainState.create(apply_fn=(), params=policy_params, tx=policy_tx)
    aux_decoder_ts = TrainState.create(
        apply_fn=aux_decoder.apply, params=aux_decoder_params, tx=enc_tx
    )

    obs, env_state = vmapped_reset(jax.random.split(rng, n_envs), env_params)

    ts = {
        "encoder_ts": encoder_ts,
        "policy_ts": policy_ts,
        "aux_decoder_ts": aux_decoder_ts,
        "rng": rng,
        "tau": init_tau,
        "env_state": env_state,
        "last_obs": obs,
        "last_teammate_obs": jnp.zeros((n_envs, 12)),
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

        teammate_obs, teammate_action, teammate_goal = (
            info["teammate_obs"],
            info["teammate_action"],
            info["teammate_current_goal"],
        )

        # Return updated runner state and transition
        transition = Trajectory(
            ts.last_timestep,
            ts.last_teammate_obs,
            teammate_action,
            teammate_goal,
            done,
        )
        ts = ts.replace(
            env_state=env_state,
            last_obs=next_obs,
            last_teammate_obs=teammate_obs[:, obs_idxs],
            last_done=done,
            last_timestep=timestep,
            global_step=ts.global_step + n_envs,
        )
        return ts, transition

    ts, trajectories = jax.lax.scan(env_step, ts, None, n_steps)
    trajectories = jax.tree_util.tree_map(lambda x: x.swapaxes(0, 1), trajectories)
    return ts, trajectories


@jax.jit
def get_enc_input(traj):
    # enc_input = jax.nn.one_hot(traj.goal[0], 6)  # One-hot encoding of the goal
    enc_input = [
        # traj.goal[0].reshape(
        #     1,
        # ),
        traj.obs[0],
    ]
    for i in range(1, traj_len):
        enc_input.append(traj.obs[i, -2:])
    enc_input = jnp.concatenate(enc_input)
    return jnp.expand_dims(enc_input, axis=0)


@jax.jit
def get_closest_goal(last_obs):
    agent_loc = last_obs[-2:]
    fruit_locs = last_obs[:-2].reshape(5, 2)
    distances = jnp.abs(fruit_locs - agent_loc).sum(axis=1)
    distances = jnp.where(fruit_locs[:, 0] < 0, 1000.0, distances)
    return distances.argmin()


@jax.jit
def update(ts, mb):
    def loss_fn(enc_params, policy_params, aux_dec_params):
        def traj_loss_fn(traj):
            enc_input = get_enc_input(traj)

            z_full, _, logits = ts.encoder_ts.apply_fn(
                enc_params, enc_input, ts.tau, ts.rng
            )
            z_full = z_full.squeeze()
            z = jax.lax.stop_gradient(z_full)
            # z = jax.nn.one_hot(z.argmax(), n_goals)

            # === Action loss (policy only) ===
            def timestep_loss(t):
                policy_input = jnp.concatenate([traj.obs[t], z, jnp.array([0.0])])
                policy_input = jnp.expand_dims(
                    policy_input, axis=0
                )  # Add batch dimension
                true_action = jnp.expand_dims(traj.action[t], axis=0)
                log_probs, _ = policy_net.apply(
                    policy_params, policy_input, true_action, method="log_prob_entropy"
                )
                return -log_probs[0]

            losses = jax.vmap(timestep_loss)(jnp.arange(traj_len))
            action_loss = (losses * traj.mask).sum() / traj.mask.sum()

            # === Auxiliary loss (encoder + obs decoder) ===
            closest_goal_hat = ts.aux_decoder_ts.apply_fn(
                aux_dec_params, jnp.concatenate([traj.obs[0], z_full])
            )
            true_closest_goal = jax.nn.one_hot(get_closest_goal(traj.obs[-1]), n_goals)
            aux_loss = jnp.sum(jnp.square(closest_goal_hat - true_closest_goal))

            return action_loss, aux_loss, logits

        action_losses, aux_losses, logits = jax.vmap(traj_loss_fn)(mb)

        # apply an entropy loss to the logits
        entropy_loss = jnp.mean(
            jax.nn.log_softmax(logits, axis=-1) * jax.nn.softmax(logits, axis=-1)
        )

        return action_losses.mean() + aux_losses.mean() + (0.1 * entropy_loss)

    loss_val, grads = jax.value_and_grad(
        lambda enc, pol, obs: loss_fn(enc, pol, obs), argnums=(0, 1, 2), has_aux=False
    )(ts.encoder_ts.params, ts.policy_ts.params, ts.aux_decoder_ts.params)

    enc_grads, policy_grads, aux_dec_grads = grads

    new_encoder_ts = ts.encoder_ts.apply_gradients(grads=enc_grads)
    new_policy_ts = ts.policy_ts.apply_gradients(grads=policy_grads)
    new_aux_decoder_ts = ts.aux_decoder_ts.apply_gradients(grads=aux_dec_grads)

    new_tau = jnp.clip(ts.tau * jnp.exp(-1e-3), 0.1, 1.0)

    new_ts = ts.replace(
        encoder_ts=new_encoder_ts,
        policy_ts=new_policy_ts,
        aux_decoder_ts=new_aux_decoder_ts,
        tau=new_tau,
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
        # z = jax.nn.one_hot(z.argmax(), n_goals)
        # z = jnp.zeros_like(z)

        def timestep_eval(t):
            policy_input = jnp.concatenate([traj.obs[t], z, jnp.array([0.0])])
            policy_input = jnp.expand_dims(policy_input, axis=0)  # Add batch dimension
            action, _, _ = policy_net.apply(ts.policy_ts.params, policy_input, ts.rng)
            return (action.squeeze() == traj.action[t]).astype(jnp.float32)

        results = jax.vmap(timestep_eval)(jnp.arange(traj_len))
        return (results * traj.mask).sum() / traj.mask.sum()

    return jax.vmap(traj_eval)(mb).mean()


def make_train_test_split(batch, frac=0.8):
    n_train = int(frac * batch.obs.shape[0])
    train_batch = jax.tree_util.tree_map(lambda x: x[:n_train], batch)
    test_batch = jax.tree_util.tree_map(lambda x: x[n_train:], batch)
    return train_batch, test_batch


ts = create_train_state(rng, 256, 1e-3, 1e-2, n_envs)

# ts, batch = collect_trajectories(ts, n_envs, n_steps)
# # batch = jax.tree_util.tree_map(lambda x: x[:, 1:], batch)
# batch = jax.tree_util.tree_map(
#     lambda x: x.reshape(n_envs * (n_steps), -1).squeeze(), batch
# )

# # filter out where timestep == 0
# idxs = jnp.where(batch.timestep != 0)[0]
# batch = jax.tree_util.tree_map(lambda x: x[idxs], batch)

# # filter out goals where goal == 5
# idxs = jnp.where(batch.goal != 5)[0]
# batch = jax.tree_util.tree_map(lambda x: x[idxs], batch)

# # obs = batch.obs[:, idxs]
# obs = jnp.where(batch.obs == -1.0, -1.0, batch.obs / 7.0)
# batch = batch.replace(obs=obs)

# print(batch.obs.shape, batch.action.shape, batch.goal.shape)

# fragments, curr_frag = [], []
# for i in tqdm(range(len(batch.goal))):
#     start_new = (i > 0) and ((batch.goal[i] != batch.goal[i - 1]) or batch.done[i - 1])
#     if not start_new:
#         curr_frag.append(
#             (batch.timestep[i], batch.obs[i], batch.goal[i], batch.action[i])
#         )
#     else:
#         fragments.append(curr_frag)
#         curr_frag = [(batch.timestep[i], batch.obs[i], batch.goal[i], batch.action[i])]

# # fragments = [frag for frag in fragments if len(frag) >= 3 and len(frag) <= traj_len]
# fragments = [
#     frag for frag in fragments if len(frag) >= traj_len and len(frag) <= traj_len + 3
# ]
# print(len(fragments), "fragments found")
# print(Counter([len(frag) for frag in fragments]))


# def pad_fragment(frag):
#     padded_timesteps = jnp.zeros((traj_len,))
#     padded_obss = jnp.zeros((traj_len, obs_dim))
#     padded_actions = jnp.zeros((traj_len,))
#     padded_goals = jnp.zeros((traj_len,))
#     mask = jnp.zeros((traj_len,))

#     for i, (timestep, obs, goal, action) in enumerate(frag):
#         padded_timesteps = padded_timesteps.at[i].set(timestep)
#         padded_obss = padded_obss.at[i].set(obs)
#         padded_actions = padded_actions.at[i].set(action)
#         padded_goals = padded_goals.at[i].set(goal)
#         mask = mask.at[i].set(1.0)

#     return padded_timesteps, padded_obss, padded_actions, padded_goals, mask


# timesteps, obss, actions, goals, masks = zip(
#     *[pad_fragment(frag) for frag in tqdm(fragments)]
# )

# # batch = Batch(
# #     timestep=jnp.stack(timesteps),
# #     obs=jnp.stack(obss),
# #     action=jnp.stack(actions),
# #     goal=jnp.stack(goals),
# #     mask=jnp.stack(masks),
# # )

# # save arrays to disk
# jnp.savez(
#     "lbf_data.npz",
#     obs=jnp.stack(obss),
#     action=jnp.stack(actions),
#     goal=jnp.stack(goals),
#     mask=jnp.stack(masks),
# )

# load arrays from disk
data = jnp.load("lbf_data.npz")
batch = Batch(
    obs=data["obs"],
    action=data["action"],
    goal=data["goal"],
    mask=data["mask"],
)

obs = jnp.where(batch.obs == -1.0, -1.0 / 7.0, batch.obs)
# obs *= 7.0
batch = batch.replace(obs=obs)

print(batch.obs[0])

# find indices of trajectories where the last action == 5
# i.e. find values i such that batch.actions[i, -1] == 5
idxs = jnp.where(batch.action[:, -1] == 5)[0]
batch = jax.tree_util.tree_map(lambda x: x[idxs], batch)

train_batch, test_batch = make_train_test_split(batch)

print(
    f"Train batch shape: {train_batch.obs.shape}, Test batch shape: {test_batch.obs.shape}"
)

count = 0
for i in tqdm(range(1000)):
    goal = int(train_batch.goal[i, 0])
    last_obs = train_batch.obs[i, -1]
    agent_loc = last_obs[-2:]
    fruit_locs = last_obs[:-2].reshape(5, 2)
    # get manhattan distance to each fruit
    distances = jnp.abs(fruit_locs - agent_loc).sum(axis=1)
    # set distances where fruit_locs == -1.0 to 1000
    distances = jnp.where(fruit_locs[:, 0] < 0, 1000.0, distances)
    if distances.argmin() == goal:
        count += 1
print(f"Accuracy: {count / 1000:.2f}")

# actions_list = batch.action.flatten().tolist()
# print("Action counts:", Counter(actions_list))


def sample_minibatch(rng, batch, batch_size):
    indices = jax.random.randint(
        rng, shape=(batch_size,), minval=0, maxval=batch.obs.shape[0]
    )
    return Batch(
        obs=batch.obs[indices],
        action=batch.action[indices],
        goal=batch.goal[indices],
        mask=batch.mask[indices],
    )


def print_z_dists(ts, test_batch):
    z_dict = defaultdict(list)
    test_mb = sample_minibatch(rng, test_batch, 1000)
    for i in tqdm(range(1000)):
        traj = jax.tree_util.tree_map(lambda x: x[i], test_mb)
        enc_input = get_enc_input(traj)
        z, _, _ = ts.encoder_ts.apply_fn(
            ts.encoder_ts.params, enc_input, ts.tau, ts.rng
        )
        z_dict[int(traj.goal[0])].append(z.squeeze())

    avgs = {
        goal: jnp.mean(jnp.array(z_values), axis=0) for goal, z_values in z_dict.items()
    }
    for goal in range(n_goals):
        print(
            f"Goal {goal} average z: {avgs[goal]} ({avgs[goal].argmax()}: {avgs[goal].max()})"
        )


losses, train_accs, test_accs = [], [], []

for i in tqdm(range(int(n_iter))):
    rng, new_rng = jax.random.split(ts.rng)
    ts = ts.replace(rng=new_rng)

    mb = sample_minibatch(rng, train_batch, 1024)

    ts, loss_val = update(ts, mb)
    losses.append(float(loss_val))
    if i % 1000 == 0:
        test_mb = sample_minibatch(rng, test_batch, 1024)
        train_acc = evaluate(ts, mb)
        test_acc = evaluate(ts, test_mb)
        train_accs.append(float(train_acc))
        test_accs.append(float(test_acc))
        tqdm.write(
            f"Step {i}, Loss: {loss_val:.4f}, Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}"
        )

        if test_acc >= 0.95:
            print("High test accuracy reached, stopping training.")
            break

    if i % 5e3 == 0:
        print_z_dists(ts, test_batch)


fig, ax = plt.subplots(1, 3, figsize=(15, 5))
sns.lineplot(x=range(len(losses)), y=losses, ax=ax[0])
sns.lineplot(x=range(len(train_accs)), y=train_accs, ax=ax[1])
sns.lineplot(x=range(len(test_accs)), y=test_accs, ax=ax[2])
ax[0].set_title("Loss")
ax[1].set_title("Train Accuracy")
ax[2].set_title("Test Accuracy")
plt.tight_layout()
plt.show()
# fig.savefig("bc_curve.png")
# plt.close(fig)

# # Save the model parameters
# # fp = "/Users/max/Code/aht-with-preferences/bc_params/lbf"
# fp = "/home/s2227283/projects/aht-with-preferences/bc_params/lbf"
# if os.path.exists(fp):
#     shutil.rmtree(fp)

# checkpointer = orbax.checkpoint.PyTreeCheckpointer()
# ckpt = {"params": ts.policy_ts.params}
# checkpointer.save(fp, ckpt)

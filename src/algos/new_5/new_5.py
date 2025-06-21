from typing import Any, Dict, Tuple

import chex
import jax
import optax
from flax import linen as nn
from flax import struct
from flax.training.train_state import TrainState
from jax import numpy as jnp
from orbax.checkpoint import PyTreeCheckpointer
import numpy as np
from rejax.algos.algorithm import Algorithm, register_init
from rejax.networks import DiscretePolicy, VNetwork
from gymnax.environments.environment import Environment, EnvParams
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.algos.mixins import (
    NormalizeObservationsMixin,
    NormalizeRewardsMixin,
    OnPolicyMixin,
)
from src.evaluate import evaluate, Policy

N_GOALS = 6


class Trajectory(struct.PyTreeNode):
    obs: chex.Array
    goal: chex.Array
    action: chex.Array
    log_prob: chex.Array
    reward: chex.Array
    value: chex.Array
    done: chex.Array
    teammate_obs: chex.Array
    teammate_goal: chex.Array
    teammate_action: chex.Array


class AdvantageMinibatch(struct.PyTreeNode):
    trajectories: Trajectory
    advantages: chex.Array
    targets: chex.Array


class NEW5(OnPolicyMixin, NormalizeObservationsMixin, NormalizeRewardsMixin, Algorithm):
    goal_actor: nn.Module = struct.field(pytree_node=False, default=None)
    goal_critic: nn.Module = struct.field(pytree_node=False, default=None)
    bc_net: nn.Module = struct.field(pytree_node=False, default=None)
    num_epochs: int = struct.field(pytree_node=False, default=8)
    gae_lambda: chex.Scalar = struct.field(pytree_node=True, default=0.95)
    clip_eps: chex.Scalar = struct.field(pytree_node=True, default=0.2)
    vf_coef: chex.Scalar = struct.field(pytree_node=True, default=0.5)
    ent_coef: chex.Scalar = struct.field(pytree_node=True, default=0.01)
    bc_net_lr: chex.Scalar = struct.field(pytree_node=True, default=1e-3)
    goal_actor_params: Any = struct.field(pytree_node=False, default=None)
    goal_critic_params: Any = struct.field(pytree_node=False, default=None)
    bc_net_params: Any = struct.field(pytree_node=False, default=None)

    @classmethod
    def create(cls, **config):
        env, env_params = cls.create_env(config)
        agent = cls.create_agent(config, env, env_params)

        def eval_callback(algo, ts, rng, trajectories):
            policy, init_extra = algo.make_policy(ts)
            max_steps = algo.env_params.max_steps_in_episode
            _, returns, _ = evaluate(
                policy,
                rng,
                env,
                env_params,
                init_extra=init_extra,
                max_steps_in_episode=max_steps,
            )
            jax.debug.print("iter: {} mean return: {}", ts.global_step, returns.mean())
            return returns

        return cls(
            env=env,
            env_params=env_params,
            eval_callback=eval_callback,
            **agent,
            **config,
        )

    @classmethod
    def create_agent(cls, config, env, env_params):
        # action_space = env.action_space(env_params)

        agent_kwargs = config.pop("agent_kwargs", {})
        activation = agent_kwargs.pop("activation", "swish")
        agent_kwargs["activation"] = getattr(nn, activation)

        hidden_layer_sizes = agent_kwargs.pop("hidden_layer_sizes", (64, 64))
        agent_kwargs["hidden_layer_sizes"] = tuple(hidden_layer_sizes)

        goal_actor = DiscretePolicy(N_GOALS, **agent_kwargs)
        goal_critic = VNetwork(**agent_kwargs)
        bc_net = DiscretePolicy(env.action_space(env_params).n, (64, 64), nn.swish)

        return {
            "goal_actor": goal_actor,
            "goal_critic": goal_critic,
            "bc_net": bc_net,
        }

    @register_init
    def initialize_network_params(self, rng):
        rng, rng_actor, rng_critic, rng_goal_net = jax.random.split(rng, 4)

        obs_dim = self.env.observation_space(self.env_params).shape[0]
        init_obs = jnp.empty([1, obs_dim], dtype=jnp.float32)

        goal_actor_params, goal_critic_params, bc_net_params = (
            self.goal_actor_params,
            self.goal_critic_params,
            self.bc_net_params,
        )
        if goal_actor_params is None:
            goal_actor_params = self.goal_actor.init(rng_actor, init_obs, rng_actor)
        if goal_critic_params is None:
            goal_critic_params = self.goal_critic.init(rng_critic, init_obs)
        if bc_net_params is None:
            init_bc_input = jnp.empty([1, obs_dim + N_GOALS], dtype=jnp.float32)
            bc_net_params = self.bc_net.init(rng_goal_net, init_bc_input, rng_goal_net)

        tx1 = optax.chain(
            optax.clip(self.max_grad_norm),
            optax.adam(learning_rate=self.learning_rate),
        )

        tx2 = optax.chain(
            optax.clip(self.max_grad_norm),
            optax.adam(learning_rate=self.bc_net_lr),
        )

        goal_actor_ts = TrainState.create(apply_fn=(), params=goal_actor_params, tx=tx1)
        goal_critic_ts = TrainState.create(
            apply_fn=(), params=goal_critic_params, tx=tx1
        )
        bc_net_ts = TrainState.create(apply_fn=(), params=bc_net_params, tx=tx2)

        return {
            "goal_actor_ts": goal_actor_ts,
            "goal_critic_ts": goal_critic_ts,
            "bc_net_ts": bc_net_ts,
        }

    def get_inputs(self, obss, goal_idxs):
        goal_vecs = jax.nn.one_hot(goal_idxs, N_GOALS)
        # obs_idxs = jnp.array([[i * 6, (i * 6) + 1] for i in range(6)]).flatten()
        # obs = obss[..., obs_idxs] / 7.0
        obs = obss[..., :2] / 8.0
        return jnp.concatenate([obs, goal_vecs], axis=-1)

    def get_low_level_actions(self, key, params, obss, goal_idxs, states):
        # def get_action(state, goal_idx):
        #     return self.env.load_or_move_towards(
        #         key, state.agent_locs[0], state.fruit_locs[goal_idx], state
        #     )

        # vmapped_get_action = jax.vmap(get_action, in_axes=(0, 0), out_axes=0)
        # actions = vmapped_get_action(states, goal_idxs)
        # actions = jnp.where(goal_idxs == N_GOALS - 1, 0, actions)
        # return actions

        inputs = self.get_inputs(obss, goal_idxs)
        actions = self.bc_net.apply(params, inputs, key, method="act")
        # actions = jnp.where(goal_idxs == N_GOALS - 1, 0, actions)
        actions = jnp.where(goal_idxs == 1, 0, actions)
        return actions

    def train(self, rng=None, train_state=None):
        if train_state is None and rng is None:
            raise ValueError("Either train_state or rng must be provided")

        ts = train_state or self.init_state(rng)
        best_ts = jax.tree_util.tree_map(lambda x: jnp.copy(x), ts)
        best_eval = -jnp.inf

        if not self.skip_initial_evaluation:
            ts, trajectories = self.collect_trajectories(ts, self.num_steps)
            initial_evaluation = self.eval_callback(self, ts, ts.rng, trajectories)

        def eval_iteration(carry, unused):
            ts, best_ts, best_eval = carry
            ts, trajectories = self.collect_trajectories(ts, self.num_steps)

            # Run a few training iterations
            iteration_steps = self.num_envs * self.num_steps
            num_iterations = np.ceil(self.eval_freq / iteration_steps).astype(int)
            ts, trajectories = jax.lax.fori_loop(
                0,
                num_iterations,
                lambda _, x: self.train_iteration(x[0]),
                (ts, trajectories),
            )

            # Run evaluation
            evaluation = self.eval_callback(self, ts, ts.rng, trajectories)
            mean_return = evaluation[0].mean()

            ts_copy = jax.tree_util.tree_map(lambda x: jnp.copy(x), ts)
            best_ts = jax.tree_util.tree_map(
                lambda x, y: jnp.where(mean_return > best_eval, x, y), ts_copy, best_ts
            )
            best_eval = jnp.maximum(mean_return, best_eval)

            return (ts, best_ts, best_eval), evaluation

        num_evals = np.ceil(self.total_timesteps / self.eval_freq).astype(int)
        out_carry, evaluation = jax.lax.scan(
            eval_iteration, (ts, best_ts, best_eval), None, num_evals
        )
        ts, best_ts, best_eval = out_carry

        if not self.skip_initial_evaluation:
            evaluation = jax.tree_util.tree_map(
                lambda i, ev: jnp.concatenate((jnp.expand_dims(i, 0), ev)),
                initial_evaluation,
                evaluation,
            )

        # ts, trajectories = self.collect_trajectories(ts, 10000)

        return ts, evaluation, None
        # return ts, evaluation, trajectories

    def train_iteration(self, ts):
        ts, trajectories = self.collect_trajectories(ts, self.num_steps)
        last_val = self.goal_critic.apply(ts.goal_critic_ts.params, ts.last_obs)
        last_val = jnp.where(ts.last_done, 0, last_val)
        advantages, targets = self.calculate_gae(trajectories, last_val)

        def update_epoch(ts, unused):
            rng, minibatch_rng = jax.random.split(ts.rng)
            ts = ts.replace(rng=rng)
            batch = AdvantageMinibatch(trajectories, advantages, targets)
            minibatches = self.shuffle_and_split(batch, minibatch_rng)
            ts, _ = jax.lax.scan(
                lambda ts, mbs: (
                    self.update(ts, mbs),
                    None,
                ),
                ts,
                minibatches,
            )
            return ts, None

        ts, _ = jax.lax.scan(update_epoch, ts, None, self.num_epochs)
        return ts, trajectories

    def collect_trajectories(self, ts, num_steps):
        def env_step(ts, unused):
            # Get keys for sampling action and stepping environment
            rng, new_rng = jax.random.split(ts.rng)
            ts = ts.replace(rng=rng)
            rng_steps, rng_action = jax.random.split(new_rng, 2)
            rng_steps = jax.random.split(rng_steps, self.num_envs)

            goal, log_prob = self.goal_actor.apply(
                ts.goal_actor_ts.params,
                ts.last_obs,
                rng_action,
                method="action_log_prob",
            )
            action = self.get_low_level_actions(
                rng_action, ts.bc_net_ts.params, ts.last_obs, goal, ts.env_state
            )
            value = self.goal_critic.apply(ts.goal_critic_ts.params, ts.last_obs)

            # Step environment
            t = self.vmap_step(rng_steps, ts.env_state, action, self.env_params)
            next_obs, env_state, reward, done, info = t

            teammate_obs, teammate_goal, teammate_action = (
                info["teammate_obs"],
                info["teammate_current_goal"],
                info["teammate_action"],
            )
            # teammate_goal = jnp.where(teammate_action == 0, 5, teammate_goal)

            if self.normalize_observations:
                obs_rms_state, next_obs = self.update_and_normalize_obs(
                    ts.obs_rms_state, next_obs
                )
                teammate_obs = self.normalize_obs(obs_rms_state, teammate_obs)
                ts = ts.replace(obs_rms_state=obs_rms_state)
            if self.normalize_rewards:
                rew_rms_state, reward = self.update_and_normalize_rew(
                    ts.rew_rms_state, reward, done
                )
                ts = ts.replace(rew_rms_state=rew_rms_state)

            # Return updated runner state and transition
            transition = Trajectory(
                ts.last_obs,
                goal,
                action,
                log_prob,
                reward,
                value,
                done,
                teammate_obs,
                teammate_goal,
                teammate_action,
            )
            ts = ts.replace(
                env_state=env_state,
                last_obs=next_obs,
                last_done=done,
                global_step=ts.global_step + self.num_envs,
            )
            return ts, transition

        ts, trajectories = jax.lax.scan(env_step, ts, None, num_steps)
        return ts, trajectories

    def calculate_gae(self, trajectories, last_val):
        def get_advantages(advantage_and_next_value, transition):
            advantage, next_value = advantage_and_next_value
            delta = (
                transition.reward.squeeze()  # For gymnax envs that return shape (1, )
                + self.gamma * next_value * (1 - transition.done)
                - transition.value
            )
            advantage = (
                delta + self.gamma * self.gae_lambda * (1 - transition.done) * advantage
            )
            return (advantage, transition.value), advantage

        _, advantages = jax.lax.scan(
            get_advantages,
            (jnp.zeros_like(last_val), last_val),
            trajectories,
            reverse=True,
        )
        return advantages, advantages + trajectories.value

    def update_goal_actor(self, ts, batch):
        def loss_fn(params):
            log_prob, entropy = self.goal_actor.apply(
                params,
                batch.trajectories.obs,
                batch.trajectories.goal,
                method="log_prob_entropy",
            )
            entropy = entropy.mean()

            # Calculate actor loss
            ratio = jnp.exp(log_prob - batch.trajectories.log_prob)
            advantages = (batch.advantages - batch.advantages.mean()) / (
                batch.advantages.std() + 1e-8
            )
            clipped_ratio = jnp.clip(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
            pi_loss1 = ratio * advantages
            pi_loss2 = clipped_ratio * advantages
            pi_loss = -jnp.minimum(pi_loss1, pi_loss2).mean()
            return pi_loss - self.ent_coef * entropy

        grads = jax.grad(loss_fn)(ts.goal_actor_ts.params)
        return ts.replace(goal_actor_ts=ts.goal_actor_ts.apply_gradients(grads=grads))

    def update_goal_critic(self, ts, batch):
        def loss_fn(params):
            value = self.goal_critic.apply(params, batch.trajectories.obs)
            value_pred_clipped = batch.trajectories.value + (
                value - batch.trajectories.value
            ).clip(-self.clip_eps, self.clip_eps)
            value_losses = jnp.square(value - batch.targets)
            value_losses_clipped = jnp.square(value_pred_clipped - batch.targets)
            value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
            return self.vf_coef * value_loss

        grads = jax.grad(loss_fn)(ts.goal_critic_ts.params)
        return ts.replace(goal_critic_ts=ts.goal_critic_ts.apply_gradients(grads=grads))

    def update_bc_net(self, ts, batch):
        def loss_fn(params):
            inputs = self.get_inputs(
                batch.trajectories.teammate_obs, batch.trajectories.teammate_goal
            )
            log_probs, entropy = self.bc_net.apply(
                params,
                inputs,
                batch.trajectories.teammate_action,
                method="log_prob_entropy",
            )
            # log_probs = jnp.where(
            #     batch.trajectories.teammate_action == 0, 0.0, log_probs
            # )
            return -log_probs.mean()

        grads = jax.grad(loss_fn)(ts.bc_net_ts.params)
        return ts.replace(bc_net_ts=ts.bc_net_ts.apply_gradients(grads=grads))

    def update(self, ts, batch):
        ts = self.update_goal_actor(ts, batch)
        ts = self.update_goal_critic(ts, batch)
        # ts = self.update_bc_net(ts, batch)
        return ts

    def make_policy(self, ts):
        def policy(rng, obs, state, extra):
            # if self.normalize_observations:
            #     obs = self.normalize_obs(ts.obs_rms_state, obs)

            obs = jnp.expand_dims(obs, axis=0)

            # use tree_map to call expand_dims on each element of state
            state = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=0), state)

            goal, _ = self.goal_actor.apply(
                ts.goal_actor_ts.params, obs, rng, method="action_log_prob"
            )

            action = self.get_low_level_actions(
                rng, ts.bc_net_ts.params, obs, goal, state
            )
            action = action.squeeze(axis=0)

            # return action, {**extra, "goal": goal[0]}
            return action, extra

        return policy, {}

    def eval_bc_net(self, rng, ts, trajectories):
        obs = trajectories.teammate_obs
        obs = obs.reshape(-1, obs.shape[-1])

        if self.normalize_observations:
            obs = self.normalize_obs(ts.obs_rms_state, obs)

        goal_idxs = trajectories.teammate_goal.reshape(-1)
        inputs = self.get_inputs(obs, goal_idxs)
        actions = self.bc_net.apply(
            ts.bc_net_ts.params,
            inputs,
            rng,
            method="act",
        )

        return jnp.mean(actions == trajectories.teammate_action.reshape(-1))


def train_new_5(
    key: chex.PRNGKey, env: Environment, env_params: EnvParams, bc_params_path: str
) -> Tuple[pd.DataFrame, Policy, Dict[str, Any], Trajectory]:
    checkpointer = PyTreeCheckpointer()
    bc_net_ckpt = checkpointer.restore(bc_params_path)
    bc_net_params = bc_net_ckpt["params"]

    algo = NEW5.create(
        env=env,
        env_params=env_params,
        total_timesteps=int(1e7),
        eval_freq=int(5e5),
        num_envs=16,
        num_steps=1024,
        num_epochs=2,
        num_minibatches=8,
        max_grad_norm=0.5,
        learning_rate=5e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        normalize_observations=False,
        normalize_rewards=False,
        bc_net_lr=5e-3,
        bc_net_params=bc_net_params,
    )

    train_fn = jax.jit(algo.train)
    train_state, evaluation, trajectories = jax.block_until_ready(train_fn(key))

    # # save parameters to file
    # checkpointer = PyTreeCheckpointer()
    # checkpointer.save(
    #     "/Users/max/Code/aht-with-preferences/new_5_params/lbf",
    #     {
    #         "goal_actor_params": train_state.goal_actor_ts.params,
    #         "goal_critic_params": train_state.goal_critic_ts.params,
    #         "bc_net_params": train_state.bc_net_ts.params,
    #     },
    # )

    ep_returns = evaluation
    mean_returns = ep_returns.mean(axis=1)

    train_df = pd.DataFrame(
        {
            "timestep": jnp.linspace(0, algo.total_timesteps, len(mean_returns)),
            "return": mean_returns,
            # "bc_acc": bc_accs,
        }
    )

    print(f"Avg of last 5 returns: {mean_returns[-5:].mean():.2f}")

    fig, ax = plt.subplots()
    sns.lineplot(data=train_df, x="timestep", y="return", ax=ax)
    plt.show()

    # fig, ax = plt.subplots()
    # sns.lineplot(data=train_df, x="timestep", y="bc_acc", ax=ax)
    # plt.show()

    policy, init_extra = algo.make_policy(train_state)
    policy = jax.jit(policy)

    return train_df, policy, init_extra, trajectories

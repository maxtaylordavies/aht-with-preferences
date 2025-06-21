import os
import shutil
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
from src.algos.models import LSTMEncoder, ActionDecoder, ScannedLSTM

N_GOALS = 6


class Trajectory(struct.PyTreeNode):
    timestep: chex.Array
    obs: chex.Array
    goal: chex.Array
    action: chex.Array
    log_prob: chex.Array
    reward: chex.Array
    value: chex.Array
    done: chex.Array
    hidden: chex.Array
    teammate_obs: chex.Array
    teammate_action: chex.Array


class AdvantageMinibatch(struct.PyTreeNode):
    trajectories: Trajectory
    advantages: chex.Array
    targets: chex.Array


class NEW6(OnPolicyMixin, NormalizeObservationsMixin, NormalizeRewardsMixin, Algorithm):
    encoder: nn.Module = struct.field(pytree_node=False, default=None)
    decoder: nn.Module = struct.field(pytree_node=False, default=None)
    goal_actor: nn.Module = struct.field(pytree_node=False, default=None)
    goal_critic: nn.Module = struct.field(pytree_node=False, default=None)
    bc_net: nn.Module = struct.field(pytree_node=False, default=None)
    num_epochs: int = struct.field(pytree_node=False, default=8)
    gae_lambda: chex.Scalar = struct.field(pytree_node=True, default=0.95)
    clip_eps: chex.Scalar = struct.field(pytree_node=True, default=0.2)
    vf_coef: chex.Scalar = struct.field(pytree_node=True, default=0.5)
    ent_coef: chex.Scalar = struct.field(pytree_node=True, default=0.01)
    encoder_lr: chex.Scalar = struct.field(pytree_node=True, default=1e-2)
    encoder_params: Any = struct.field(pytree_node=False, default=None)
    decoder_params: Any = struct.field(pytree_node=False, default=None)
    goal_actor_params: Any = struct.field(pytree_node=False, default=None)
    goal_critic_params: Any = struct.field(pytree_node=False, default=None)
    bc_net_params: Any = struct.field(pytree_node=False, default=None)
    encoder_hidden_size: int = struct.field(pytree_node=True, default=64)
    encoder_output_size: int = struct.field(pytree_node=True, default=20)
    action_space_n: int = struct.field(pytree_node=True, default=6)

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
            action_acc = algo.eval_reconstruction(ts, trajectories)
            jax.debug.print(
                "iter: {}, return: {:.3f}, action acc: {:.3f}",
                ts.global_step,
                returns.mean(),
                action_acc,
            )
            return returns, action_acc

        return cls(
            env=env,
            env_params=env_params,
            eval_callback=eval_callback,
            **agent,
            **config,
        )

    @classmethod
    def create_agent(cls, config, env, env_params):
        obs_space = env.observation_space(env_params)
        action_space = env.action_space(env_params)

        agent_kwargs = config.pop("agent_kwargs", {})
        activation = agent_kwargs.pop("activation", "swish")
        agent_kwargs["activation"] = getattr(nn, activation)

        hidden_layer_sizes = agent_kwargs.pop("hidden_layer_sizes", (64, 64))
        agent_kwargs["hidden_layer_sizes"] = tuple(hidden_layer_sizes)

        encoder = LSTMEncoder(output_size=config["encoder_output_size"])
        decoder = ActionDecoder(output_size=action_space.n)
        goal_actor = DiscretePolicy(N_GOALS, **agent_kwargs)
        goal_critic = VNetwork(**agent_kwargs)
        bc_net = DiscretePolicy(env.action_space(env_params).n, (64, 64), nn.swish)

        return {
            "encoder": encoder,
            "decoder": decoder,
            "goal_actor": goal_actor,
            "goal_critic": goal_critic,
            "bc_net": bc_net,
        }

    @register_init
    def initialize_network_params(self, rng):
        rng, rng_encoder, rng_decoder, rng_actor, rng_critic, rng_bc_net = (
            jax.random.split(rng, 6)
        )

        obs_dim = self.env.observation_space(self.env_params).shape[0]
        init_obs = jnp.empty([1, obs_dim], dtype=jnp.float32)
        init_action_one_hot = jnp.empty([1, self.action_space_n])
        init_dones = jnp.empty([1, 1])  # shape [batch_size, sequence_length]
        init_hidden = ScannedLSTM.initialize_carry(1, self.encoder_hidden_size)

        tmp = jnp.concatenate([init_obs, init_action_one_hot], axis=-1)
        tmp = jnp.expand_dims(tmp, axis=0)
        init_enc_input = (tmp, init_dones)
        init_z = jnp.empty([1, self.encoder_output_size])
        init_ac_input = jnp.concatenate([init_obs, init_z], axis=-1)

        enc_params, dec_params, goal_actor_params, goal_critic_params, bc_net_params = (
            self.encoder_params,
            self.decoder_params,
            self.goal_actor_params,
            self.goal_critic_params,
            self.bc_net_params,
        )
        if enc_params is None:
            enc_params = self.encoder.init(rng_encoder, init_hidden, init_enc_input)
        if dec_params is None:
            dec_params = self.decoder.init(rng_decoder, init_z)
        if goal_actor_params is None:
            goal_actor_params = self.goal_actor.init(
                rng_actor, init_ac_input, rng_actor
            )
        if goal_critic_params is None:
            goal_critic_params = self.goal_critic.init(rng_critic, init_ac_input)
        if bc_net_params is None:
            init_bc_input = jnp.empty([1, obs_dim + N_GOALS], dtype=jnp.float32)
            bc_net_params = self.bc_net.init(rng_bc_net, init_bc_input, rng_bc_net)

        ac_tx = optax.chain(
            optax.clip_by_global_norm(self.max_grad_norm),
            optax.adam(learning_rate=self.learning_rate),
        )

        enc_schedule = optax.linear_schedule(
            init_value=self.encoder_lr,
            end_value=0.0,
            transition_steps=self.total_timesteps // 5,
        )
        enc_tx = optax.chain(
            optax.clip_by_global_norm(self.max_grad_norm),
            optax.adam(learning_rate=enc_schedule),
        )

        encoder_ts = TrainState.create(apply_fn=(), params=enc_params, tx=enc_tx)
        decoder_ts = TrainState.create(apply_fn=(), params=dec_params, tx=enc_tx)
        goal_actor_ts = TrainState.create(
            apply_fn=(), params=goal_actor_params, tx=ac_tx
        )
        goal_critic_ts = TrainState.create(
            apply_fn=(), params=goal_critic_params, tx=ac_tx
        )
        bc_net_ts = TrainState.create(apply_fn=(), params=bc_net_params, tx=ac_tx)

        return {
            "encoder_ts": encoder_ts,
            "decoder_ts": decoder_ts,
            "goal_actor_ts": goal_actor_ts,
            "goal_critic_ts": goal_critic_ts,
            "bc_net_ts": bc_net_ts,
        }

    @register_init
    def initialize_additional_inputs(self, rng):
        init_hidden = ScannedLSTM.initialize_carry(
            self.num_envs, self.encoder_hidden_size
        )
        return {
            "last_action": jnp.zeros(self.num_envs, dtype=jnp.int32),
            "last_hidden": init_hidden,
            "last_timestep": jnp.zeros(self.num_envs, dtype=jnp.int32),
        }

    def embed(self, params, obs, action, done, hidden):
        action_one_hot = jax.nn.one_hot(action, self.action_space_n)
        inputs = (
            jnp.expand_dims(jnp.concatenate([obs, action_one_hot], axis=-1), axis=1),
            jnp.expand_dims(done, axis=1),
        )
        _hidden, z = self.encoder.apply(params, hidden, inputs)
        z = z.squeeze()
        return _hidden, z

    def get_bc_inputs(self, obss, goal_idxs):
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

        inputs = self.get_bc_inputs(obss, goal_idxs)
        actions = self.bc_net.apply(params, inputs, key, method="act")
        actions = jnp.where(goal_idxs == N_GOALS - 1, 0, actions)
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

        ts, trajectories = self.collect_trajectories(ts, 1000)
        return ts, evaluation, trajectories

    def train_iteration(self, ts):
        ts, trajectories = self.collect_trajectories(ts, self.num_steps)

        _, z = self.embed(
            ts.encoder_ts.params,
            ts.last_obs,
            ts.last_action,
            ts.last_done,
            ts.last_hidden,
        )
        input = jnp.concatenate([ts.last_obs, z], axis=-1)

        last_val = self.goal_critic.apply(ts.goal_critic_ts.params, input)
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

            # Sample goal and action
            hidden, zs = self.embed(
                ts.encoder_ts.params,
                ts.last_obs,
                ts.last_action,
                ts.last_done,
                ts.last_hidden,
            )
            inputs = jnp.concatenate([ts.last_obs, zs], axis=-1)
            goal, log_prob = self.goal_actor.apply(
                ts.goal_actor_ts.params,
                inputs,
                rng_action,
                method="action_log_prob",
            )
            action = self.get_low_level_actions(
                rng_action, ts.bc_net_ts.params, ts.last_obs, goal, ts.env_state
            )
            value = self.goal_critic.apply(ts.goal_critic_ts.params, inputs)

            # Step environment
            t = self.vmap_step(rng_steps, ts.env_state, action, self.env_params)
            next_obs, env_state, reward, done, info = t
            teammate_obs, teammate_action = (
                info["teammate_obs"],
                info["teammate_action"],
            )

            if self.normalize_observations:
                obs_rms_state, next_obs = self.update_and_normalize_obs(
                    ts.obs_rms_state, next_obs
                )
                ts = ts.replace(obs_rms_state=obs_rms_state)
                teammate_obs = self.normalize_obs(obs_rms_state, teammate_obs)
            if self.normalize_rewards:
                rew_rms_state, reward = self.update_and_normalize_rew(
                    ts.rew_rms_state, reward, done
                )
                ts = ts.replace(rew_rms_state=rew_rms_state)

            timestep = jnp.where(done, 0, ts.last_timestep + 1)

            # Return updated runner state and transition
            transition = Trajectory(
                timestep,
                ts.last_obs,
                goal,
                action,
                log_prob,
                reward,
                value,
                done,
                hidden,
                teammate_obs,
                teammate_action,
            )
            ts = ts.replace(
                env_state=env_state,
                last_obs=next_obs,
                last_action=action,
                last_done=done,
                last_hidden=hidden,
                last_timestep=timestep,
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
        _, zs = self.embed(
            ts.encoder_ts.params,
            batch.trajectories.obs,
            batch.trajectories.action,
            batch.trajectories.done,
            batch.trajectories.hidden,
        )
        inputs = jnp.concatenate([batch.trajectories.obs, zs], axis=-1)

        def loss_fn(params):
            log_prob, entropy = self.goal_actor.apply(
                params,
                inputs,
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
        _, zs = self.embed(
            ts.encoder_ts.params,
            batch.trajectories.obs,
            batch.trajectories.action,
            batch.trajectories.done,
            batch.trajectories.hidden,
        )
        inputs = jnp.concatenate([batch.trajectories.obs, zs], axis=-1)

        def loss_fn(params):
            value = self.goal_critic.apply(params, inputs)
            value_pred_clipped = batch.trajectories.value + (
                value - batch.trajectories.value
            ).clip(-self.clip_eps, self.clip_eps)
            value_losses = jnp.square(value - batch.targets)
            value_losses_clipped = jnp.square(value_pred_clipped - batch.targets)
            value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
            return self.vf_coef * value_loss

        grads = jax.grad(loss_fn)(ts.goal_critic_ts.params)
        return ts.replace(goal_critic_ts=ts.goal_critic_ts.apply_gradients(grads=grads))

    def update_encoder_decoder(self, ts, batch):
        def loss_fn(enc_params, dec_params):
            _, zs = self.embed(
                enc_params,
                batch.trajectories.obs,
                batch.trajectories.action,
                batch.trajectories.done,
                batch.trajectories.hidden,
            )
            probs = self.decoder.apply(dec_params, zs)
            rec_loss = -jnp.log(
                (
                    probs
                    * jax.nn.one_hot(
                        batch.trajectories.teammate_action, self.action_space_n
                    )
                ).sum(axis=-1)
            )
            return ((1 - batch.trajectories.done) * rec_loss).mean()

        encoder_loss_fn = lambda enc_params: loss_fn(enc_params, ts.decoder_ts.params)
        decoder_loss_fn = lambda dec_params: loss_fn(ts.encoder_ts.params, dec_params)

        encoder_grads = jax.grad(encoder_loss_fn)(ts.encoder_ts.params)
        decoder_grads = jax.grad(decoder_loss_fn)(ts.decoder_ts.params)

        return ts.replace(
            encoder_ts=ts.encoder_ts.apply_gradients(grads=encoder_grads),
            decoder_ts=ts.decoder_ts.apply_gradients(grads=decoder_grads),
        )

    def update(self, ts, batch):
        ts = self.update_goal_actor(ts, batch)
        ts = self.update_goal_critic(ts, batch)
        ts = self.update_encoder_decoder(ts, batch)
        return ts

    def make_policy(self, ts):
        def policy(rng, obs, state, extra):
            if self.normalize_observations:
                obs = self.normalize_obs(ts.obs_rms_state, obs)

            obs = jnp.expand_dims(obs, axis=0)
            state = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=0), state)

            last_action, hidden, last_done = (
                extra["last_action"],
                extra["hidden"],
                extra["done"],
            )

            hidden, z = self.embed(
                ts.encoder_ts.params,
                obs,
                last_action,
                last_done,
                hidden,
            )
            z = jnp.expand_dims(z, axis=0)
            inputs = jnp.concatenate([obs, z], axis=-1)
            goal, _ = self.goal_actor.apply(
                ts.goal_actor_ts.params, inputs, rng, method="action_log_prob"
            )
            action = self.get_low_level_actions(
                rng, ts.bc_net_ts.params, obs, goal, state
            )
            action = action.squeeze(axis=0)

            return action, {
                "last_action": jnp.expand_dims(action, axis=0),
                "hidden": hidden,
                "done": last_done,
            }

        init_extra = {
            "last_action": jnp.zeros(1, dtype=jnp.int32),
            "hidden": ScannedLSTM.initialize_carry(1, self.encoder_hidden_size),
            "done": jnp.zeros(1, dtype=jnp.bool_),
        }
        return policy, init_extra

    def eval_reconstruction(self, ts, trajectories):
        def eval_timestep(transition):
            _, zs = self.embed(
                ts.encoder_ts.params,
                transition.obs,
                transition.action,
                transition.done,
                transition.hidden,
            )
            pi_hat = self.decoder.apply(ts.decoder_ts.params, zs)
            return (pi_hat.argmax(axis=-1) == transition.teammate_action).astype(
                jnp.float32
            )

        action_accs = jax.vmap(eval_timestep)(trajectories)
        return action_accs.mean()


def train_new_6(
    key: chex.PRNGKey, env: Environment, env_params: EnvParams, bc_params_path: str
) -> Tuple[pd.DataFrame, Policy, Dict[str, Any], Trajectory]:
    checkpointer = PyTreeCheckpointer()
    bc_net_ckpt = checkpointer.restore(bc_params_path)
    bc_net_params = bc_net_ckpt["params"]

    algo = NEW6.create(
        env=env,
        env_params=env_params,
        total_timesteps=int(1e7),
        eval_freq=int(5e5),
        num_envs=16,
        num_steps=1024,
        num_epochs=2,
        num_minibatches=8,
        max_grad_norm=0.5,
        learning_rate=1e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        normalize_observations=False,
        normalize_rewards=False,
        bc_net_params=bc_net_params,
        action_space_n=env.action_space(env_params).n,
        encoder_output_size=10,
    )

    train_fn = jax.jit(algo.train)
    train_state, evaluation, trajectories = jax.block_until_ready(train_fn(key))

    # # save parameters to file
    # fp = "/Users/max/Code/aht-with-preferences/new_6_params/lbf"
    # if os.path.exists(fp):
    #     shutil.rmtree(fp)
    # checkpointer = PyTreeCheckpointer()
    # checkpointer.save(
    #     fp,
    #     {
    #         "goal_actor_params": train_state.goal_actor_ts.params,
    #         "goal_critic_params": train_state.goal_critic_ts.params,
    #         "bc_net_params": train_state.bc_net_ts.params,
    #     },
    # )

    ep_returns, action_accs = evaluation
    mean_returns = ep_returns.mean(axis=1)

    train_df = pd.DataFrame(
        {
            "timestep": jnp.linspace(0, algo.total_timesteps, len(mean_returns)),
            "return": mean_returns,
            "action_acc": action_accs,
        }
    )

    print(f"Avg of last 5 returns: {mean_returns[-5:].mean():.2f}")

    # fig, ax = plt.subplots()
    # sns.lineplot(data=train_df, x="timestep", y="return", ax=ax)
    # plt.show()

    policy, init_extra = algo.make_policy(train_state)
    policy = jax.jit(policy)

    return train_df, policy, init_extra, trajectories

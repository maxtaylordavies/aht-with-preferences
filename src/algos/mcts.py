import functools

from typing import Tuple

import chex
import mctx
import haiku as hk
import jax as jax
import jax.numpy as jnp
import optax
from gymnax.environments.environment import Environment, EnvParams
import pandas as pd
import numpy as np
from tqdm import tqdm

from src.evaluate import Policy
from src.utils import dotdict

config = dotdict(
    {
        "num_hidden_layers": 2,
        "num_hidden_units": 128,
        "V_alpha": 0.0001,
        "pi_alpha": 0.00004,
        "b1_adam": 0.9,
        "b2_adam": 0.99,
        "eps_adam": 1e-5,
        "wd_adam": 1e-6,
        "discount": 0.99,
        "use_mixed_value": True,
        "value_scale": 0.1,
        "value_target": "maxq",
        "target_update_frequency": 100,
        "batch_size": 128,
        "avg_return_smoothing": 0.5,
        "num_simulations": 10,
        "total_steps": 1e6,
        "eval_frequency": 5e4,
        "activation": "relu",
    }
)

activation_dict = {
    "relu": jax.nn.relu,
    "silu": jax.nn.silu,
    "elu": jax.nn.elu,
    "swish": jax.nn.swish,
}

# run_state contains all information to be maintained and updated in interaction_loop
run_state_names = [
    "env_states",
    "V_opt_state",
    "V_target_params",
    "pi_opt_state",
    "pi_params",
    "V_params",
    "opt_t",
    "avg_return",
    "episode_return",
    "num_episodes",
    "key",
]


class VFunction(hk.Module):
    def __init__(self, config, name=None):
        super().__init__(name=name)
        self.num_hidden_units = config.num_hidden_units
        self.num_hidden_layers = config.num_hidden_layers
        self.activation_function = activation_dict[config.activation]

    def __call__(self, obs):
        x = jnp.ravel(obs)
        for i in range(self.num_hidden_layers):
            x = self.activation_function(hk.Linear(self.num_hidden_units)(x))
        V = hk.Linear(1)(x)[0]
        return V


class PiFunction(hk.Module):
    def __init__(self, config, num_actions, name=None):
        super().__init__(name=name)
        self.num_hidden_units = config.num_hidden_units
        self.num_hidden_layers = config.num_hidden_layers
        self.activation_function = activation_dict[config.activation]
        self.num_actions = num_actions

    def __call__(self, obs):
        x = jnp.ravel(obs)
        for i in range(self.num_hidden_layers):
            x = self.activation_function(hk.Linear(self.num_hidden_units)(x))
        pi_logit = hk.Linear(self.num_actions)(x)
        return pi_logit


# this assumes the agent has access to the exact environment dynamics
def get_recurrent_fn(
    env: Environment, env_params: EnvParams, v_func, pi_func, batch_size, config
):
    batch_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))
    batch_pi_func = jax.vmap(pi_func, in_axes=(None, 0))
    batch_v_func = jax.vmap(v_func, in_axes=(None, 0))

    def recurrent_fn(params, key, actions, env_states):
        V_params, pi_params = params["V"], params["pi"]
        key, subkey = jax.random.split(key)
        subkeys = jax.random.split(subkey, num=batch_size)
        obs, env_states, rewards, terminals, _ = batch_step(
            subkeys, env_states, actions, env_params
        )
        V = batch_v_func(V_params, obs.astype(float))
        pi_logit = batch_pi_func(pi_params, obs.astype(float))
        return (
            mctx.RecurrentFnOutput(
                reward=rewards,
                discount=(1.0 - terminals) * config.discount,
                prior_logits=pi_logit,
                value=V,
            ),
            env_states,
        )

    return recurrent_fn


def get_init_fn(env: Environment, env_params: EnvParams, config):
    reset = lambda key, params: env.reset(key, env_params)[1]  # only return state
    batch_reset = jax.vmap(reset, in_axes=(0, None))

    def init_fn(key):
        dummy_obs, dummy_state = env.reset(key, env_params)
        key, subkey = jax.random.split(key)
        subkeys = jax.random.split(subkey, num=config.batch_size)
        env_states = batch_reset(subkeys, env_params)

        V_net = hk.without_apply_rng(
            hk.transform(lambda obs: VFunction(config)(obs.astype(float)))
        )
        key, subkey = jax.random.split(key)
        V_params = V_net.init(subkey, dummy_obs)
        v_func = V_net.apply

        V_target_params = V_params

        pi_net = hk.without_apply_rng(
            hk.transform(
                lambda obs: PiFunction(config, env.num_actions)(obs.astype(float))
            )
        )
        key, subkey = jax.random.split(key)
        pi_params = pi_net.init(subkey, dummy_obs)
        pi_func = pi_net.apply

        V_optim = optax.adamw(
            learning_rate=config.V_alpha,
            eps=config.eps_adam,
            b1=config.b1_adam,
            b2=config.b2_adam,
            weight_decay=config.wd_adam,
        )
        V_opt_state = V_optim.init(V_params)

        pi_optim = optax.adamw(
            learning_rate=config.pi_alpha,
            eps=config.eps_adam,
            b1=config.b1_adam,
            b2=config.b2_adam,
            weight_decay=config.wd_adam,
        )
        pi_opt_state = pi_optim.init(pi_params)

        return (
            env_states,
            v_func,
            pi_func,
            V_opt_state,
            pi_opt_state,
            V_optim,
            pi_optim,
            V_params,
            V_target_params,
            pi_params,
        )

    return init_fn


def get_AC_loss(pi_func, v_func):
    def AC_loss(pi_params, V_params, pi_target, V_target, obs):
        pi_logits = pi_func(pi_params, obs.astype(float))
        V = v_func(V_params, obs.astype(float))
        pi_loss = jnp.sum(
            pi_target * (jnp.log(pi_target) - jax.nn.log_softmax(pi_logits))
        )
        V_loss = (V_target - V) ** 2
        return jnp.sum(pi_loss + V_loss)

    return AC_loss


def get_interaction_loop_fn(
    env: Environment,
    env_params: EnvParams,
    v_func,
    pi_func,
    V_optim,
    pi_optim,
    recurrent_fn,
    num_actions,
    iterations,
    config,
):
    batch_loss = lambda *x: jnp.mean(
        jax.vmap(get_AC_loss(pi_func, v_func), in_axes=(None, None, 0, 0, 0))(*x)
    )
    loss_grad = jax.grad(batch_loss, argnums=(0, 1))
    batch_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))
    batch_obs = jax.vmap(env.get_obs)
    batch_v_func = jax.vmap(v_func, in_axes=(None, 0))
    batch_pi_func = jax.vmap(pi_func, in_axes=(None, 0))

    reset = lambda key, params: env.reset(key, env_params)[1]  # only return state
    batch_reset = jax.vmap(reset, in_axes=(0, None))

    def interaction_loop_fn(S):
        def loop_function(S, data):
            obs = batch_obs(S["env_states"])
            pi_logits = batch_pi_func(S["pi_params"], obs.astype(float))
            V = batch_v_func(S["V_target_params"], obs.astype(float))

            root = mctx.RootFnOutput(
                prior_logits=pi_logits, value=V, embedding=S["env_states"]
            )

            S["key"], subkey = jax.random.split(S["key"])
            policy_output = mctx.gumbel_muzero_policy(
                params={
                    "V": S["V_target_params"],
                    "pi": S["pi_params"],  # Use current parameters directly
                },
                rng_key=subkey,
                root=root,
                recurrent_fn=recurrent_fn,
                num_simulations=config.num_simulations,
                max_num_considered_actions=num_actions,
                max_depth=env_params.max_steps_in_episode,
                qtransform=functools.partial(
                    mctx.qtransform_completed_by_mix_value,
                    use_mixed_value=config.use_mixed_value,
                    value_scale=config.value_scale,
                ),
            )

            # tree search derived targets for policy and value function
            search_policy = policy_output.action_weights
            if config.value_target == "maxq":
                search_value = policy_output.search_tree.qvalues(
                    jnp.full(config.batch_size, policy_output.search_tree.ROOT_INDEX)
                )[jnp.arange(config.batch_size), policy_output.action]
            elif config.value_target == "nodev":
                search_value = policy_output.search_tree.node_values[
                    :, policy_output.search_tree.ROOT_INDEX
                ]
            else:
                raise ValueError("Unknown value target.")

            # compute loss gradient compared to tree search targets and update parameters
            pi_grads, V_grads = loss_grad(
                S["pi_params"],  # Use current parameters directly
                S["V_params"],  # Use current parameters directly
                search_policy,
                search_value,
                obs,
            )

            # Apply updates using optax
            pi_updates, S["pi_opt_state"] = pi_optim.update(
                pi_grads, S["pi_opt_state"], params=S["pi_params"]
            )
            V_updates, S["V_opt_state"] = V_optim.update(
                V_grads, S["V_opt_state"], params=S["V_params"]
            )
            S["pi_params"] = optax.apply_updates(S["pi_params"], pi_updates)
            S["V_params"] = optax.apply_updates(S["V_params"], V_updates)
            S["opt_t"] += 1

            # Update target params after a particular number of parameter updates
            S["V_target_params"] = jax.tree_util.tree_map(
                lambda x, y: jnp.where(
                    S["opt_t"] % config.target_update_frequency == 0, x, y
                ),
                S["V_params"],  # Use current parameters directly
                S["V_target_params"],
            )

            # always take action recommended by tree search
            actions = policy_output.action

            # step the environment
            S["key"], subkey = jax.random.split(S["key"])
            subkeys = jax.random.split(subkey, num=config.batch_size)
            obs, S["env_states"], reward, terminal, _ = batch_step(
                subkeys, S["env_states"], actions, env_params
            )

            # reset environment if terminated
            S["env_states"] = jax.tree_util.tree_map(
                lambda x, y: jnp.where(
                    jnp.reshape(
                        terminal, [terminal.shape[0]] + [1] * (len(x.shape) - 1)
                    ),
                    x,
                    y,
                ),
                batch_reset(subkeys, env_params),
                S["env_states"],
            )

            # update statistics for computing average return
            S["episode_return"] += reward
            S["avg_return"] = jnp.where(
                terminal,
                S["avg_return"] * config.avg_return_smoothing
                + S["episode_return"] * (1 - config.avg_return_smoothing),
                S["avg_return"],
            )
            S["episode_return"] = jnp.where(terminal, 0, S["episode_return"])
            S["num_episodes"] = jnp.where(
                terminal, S["num_episodes"] + 1, S["num_episodes"]
            )
            return S, None

        S["key"], subkey = jax.random.split(S["key"])
        S, _ = jax.lax.scan(loop_function, S, None, length=iterations)
        return S

    return interaction_loop_fn


def get_trained_policy(
    pi_func, v_func, pi_params, V_params, recurrent_fn, env_params, config, num_actions
) -> Policy:
    def policy(key, obs, state):
        _state = jax.tree_util.tree_map(lambda x: x.reshape((1,) + x.shape), state)
        pi_logits = pi_func(pi_params, obs.astype(float)).reshape((1, -1))
        v = v_func(V_params, obs.astype(float)).reshape((1,))
        root = mctx.RootFnOutput(prior_logits=pi_logits, value=v, embedding=_state)
        policy_output = mctx.gumbel_muzero_policy(
            params={
                "V": V_params,
                "pi": pi_params,
            },
            rng_key=key,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=config.num_simulations,
            max_num_considered_actions=num_actions,
            max_depth=env_params.max_steps_in_episode,
            qtransform=functools.partial(
                mctx.qtransform_completed_by_mix_value,
                use_mixed_value=config.use_mixed_value,
                value_scale=config.value_scale,
            ),
        )
        return policy_output.action[0]

    return policy


def train_mcts(
    key: chex.PRNGKey, env: Environment, env_params: EnvParams
) -> Tuple[Policy, pd.DataFrame]:
    eval_freq_batch = config.eval_frequency // config.batch_size
    opt_t, time_step = 0, 0
    avg_return = jnp.zeros(config.batch_size)
    episode_return = jnp.zeros(config.batch_size)
    num_episodes = jnp.zeros(config.batch_size)
    num_actions = env.num_actions

    key, subkey = jax.random.split(key)
    (
        env_states,
        v_func,
        pi_func,
        V_opt_state,
        pi_opt_state,
        V_optim,
        pi_optim,
        V_params,
        V_target_params,
        pi_params,
    ) = get_init_fn(env, env_params, config)(subkey)
    recurrent_fn = get_recurrent_fn(
        env, env_params, v_func, pi_func, config.batch_size, config
    )
    interaction_loop_fn = get_interaction_loop_fn(
        env,
        env_params,
        v_func,
        pi_func,
        V_optim,
        pi_optim,
        recurrent_fn,
        num_actions,
        eval_freq_batch,
        config,
    )

    var_dict = locals()
    run_state = {name: var_dict[name] for name in run_state_names}

    times, avg_returns = [], []
    pbar = tqdm(total=config.total_steps)
    num_batches, delta_t = (
        int(config.total_steps // config.batch_size),
        int(eval_freq_batch * config.batch_size),
    )
    for i in range(int(num_batches // eval_freq_batch)):
        # perform a number of iterations of agent environment interaction including learning updates
        run_state = interaction_loop_fn(run_state)

        # avg_return is debiased, and only includes batch elements with at least one completed episode so that it is more meaningful in early episodes
        valid_avg_returns = run_state["avg_return"][run_state["num_episodes"] > 0]
        valid_num_episodes = run_state["num_episodes"][run_state["num_episodes"] > 0]
        avg_return = jnp.mean(
            valid_avg_returns / (1 - config.avg_return_smoothing**valid_num_episodes)
        )

        tqdm.write(f"Running Avg Return (t={time_step}): {avg_return}")

        avg_returns += [avg_return]
        time_step += delta_t
        times += [time_step]
        pbar.update(delta_t)

    train_df = pd.DataFrame(
        {"timestep": np.array(times), "return": np.array(avg_returns)}
    )

    pi_params, V_params = run_state["pi_params"], run_state["V_params"]
    recurrent_fn = get_recurrent_fn(env, env_params, v_func, pi_func, 1, config)
    policy = get_trained_policy(
        pi_func,
        v_func,
        pi_params,
        V_params,
        recurrent_fn,
        env_params,
        config,
        num_actions,
    )

    return policy, train_df

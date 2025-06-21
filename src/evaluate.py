from functools import partial
from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
from gymnax.environments import environment

# (key, obs, state, extra) -> action, extra
Policy = Callable[
    [chex.PRNGKey, chex.Array, chex.Array, Dict[str, Any]],
    Tuple[chex.Array, Dict[str, Any]],
]

# (env, final episode state) -> metrics
MetricFn = Callable[
    [environment.Environment, environment.EnvState, chex.Array],
    Dict[str, chex.Array],
]

dummy_metric_fn = lambda env, state, attempted: {}


class EvalState(NamedTuple):
    rng: chex.PRNGKey
    env_state: environment.EnvState
    last_obs: chex.Array
    prev_env_state: environment.EnvState
    extra: Dict[str, Any]
    done: bool = False
    return_: float = 0.0
    length: int = 0
    attempted: chex.Array = jnp.zeros(
        5, dtype=jnp.int32
    )  # Assuming 5 goals, adjust as needed


def evaluate_single(
    act: Policy,  # (key, obs, state, extra) -> action, extra
    env: environment.Environment,
    env_params: environment.EnvParams,
    rng: chex.PRNGKey,
    max_steps_in_episode: int,
    metric_fn: MetricFn = dummy_metric_fn,
    init_extra: Dict[str, Any] = {},
    normalise_return: bool = False,
):
    def step(state):
        rng, rng_act, rng_step = jax.random.split(state.rng, 3)
        action, extra = act(rng_act, state.last_obs, state.env_state, state.extra)
        obs, env_state, reward, done, info = env.step(
            rng_step, state.env_state, action, env_params
        )

        # extra["done"] = jnp.expand_dims(done, axis=0)
        return EvalState(
            rng=rng,
            env_state=env_state,
            prev_env_state=state.env_state,
            last_obs=obs,
            done=done,
            extra=extra,
            return_=state.return_ + reward.squeeze(),
            length=state.length + 1,
            attempted=info["attempted"],
        )

    rng_reset, rng_eval = jax.random.split(rng)
    obs, env_state = env.reset(rng_reset, env_params)
    # init_extra["first_obs"] = obs
    state = EvalState(
        rng_eval,
        env_state,
        obs,
        prev_env_state=env_state,
        extra=init_extra,
    )
    state = jax.lax.while_loop(
        lambda s: jnp.logical_and(
            s.length < max_steps_in_episode, jnp.logical_not(s.done)
        ),
        step,
        state,
    )

    metrics = metric_fn(env, state.prev_env_state, state.attempted)
    ret = jnp.where(
        normalise_return,
        state.return_ / state.prev_env_state.max_available_reward,
        state.return_,
    )
    return state.length, ret, metrics


@partial(jax.jit, static_argnames=("act", "env", "num_seeds", "metric_fn"))
def evaluate(
    act: Policy,
    rng: chex.PRNGKey,
    env: environment.Environment,
    env_params: Any,
    metric_fn: MetricFn = dummy_metric_fn,
    init_extra: Dict[str, Any] = {},
    num_seeds: int = 128,
    max_steps_in_episode: Optional[int] = None,
    normalise_return: bool = False,
) -> Tuple[chex.Array, chex.Array, Dict[str, chex.Array]]:
    """Evaluate a policy given by `act` on `num_seeds` environments.

    Args:
        act (Callable[[chex.Array, chex.PRNGKey], chex.Array]): A policy represented as
        a function of type (obs, rng) -> action.
        rng (chex.PRNGKey): Initial seed, will be split into `num_seeds` seeds for
        parallel evaluation.
        env (environment.Environment): The environment to evaluate on.
        env_params (Any): The parameters of the environment.
        metric_fn (MetricFn): A function to compute additional episode-wise metrics
        num_seeds (int): Number of initializations of the environment.

    Returns:
        Tuple[chex.Array, chex.Array, Dict[str, chex.Array]]: Tuple of episode lengths, cumulative rewards
        and any additional metrics as computed by metric_fn.
    """
    if max_steps_in_episode is None:
        max_steps_in_episode = env_params.max_steps_in_episode
    seeds = jax.random.split(rng, num_seeds)
    vmap_collect = jax.vmap(
        evaluate_single, in_axes=(None, None, None, 0, None, None, None, None)
    )
    return vmap_collect(
        act,
        env,
        env_params,
        seeds,
        max_steps_in_episode,
        metric_fn,
        init_extra,
        normalise_return,
    )

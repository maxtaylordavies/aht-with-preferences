from functools import partial
from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
from gymnax.environments import environment

# (key, obs, state) -> action
Policy = Callable[[chex.PRNGKey, chex.Array, chex.Array], chex.Array]

# (env, final episode state) -> metrics
MetricFn = Callable[
    [environment.Environment, environment.EnvState],
    Dict[str, chex.Array],
]


class EvalState(NamedTuple):
    rng: chex.PRNGKey
    env_state: environment.EnvState
    last_obs: chex.Array
    prev_env_state: environment.EnvState
    done: bool = False
    return_: float = 0.0
    length: int = 0


def evaluate_single(
    act: Policy,  # (key, obs, state) -> action
    env: environment.Environment,
    env_params: environment.EnvParams,
    rng: chex.PRNGKey,
    max_steps_in_episode: int,
    metric_fn: MetricFn,
):
    def step(state):
        rng, rng_act, rng_step = jax.random.split(state.rng, 3)
        action = act(rng_act, state.last_obs, state.env_state)
        obs, env_state, reward, done, _ = env.step(
            rng_step, state.env_state, action, env_params
        )
        return EvalState(
            rng=rng,
            env_state=env_state,
            prev_env_state=state.env_state,
            last_obs=obs,
            done=done,
            return_=state.return_ + reward.squeeze(),
            length=state.length + 1,
        )

    rng_reset, rng_eval = jax.random.split(rng)
    obs, env_state = env.reset(rng_reset, env_params)
    state = EvalState(rng_eval, env_state, obs, prev_env_state=env_state)
    state = jax.lax.while_loop(
        lambda s: jnp.logical_and(
            s.length < max_steps_in_episode, jnp.logical_not(s.done)
        ),
        step,
        state,
    )
    metrics = metric_fn(env, state.prev_env_state)
    return state.length, state.return_, metrics


@partial(jax.jit, static_argnames=("act", "env", "num_seeds", "metric_fn"))
def evaluate(
    act: Callable[[chex.Array, chex.PRNGKey], chex.Array],
    rng: chex.PRNGKey,
    env: environment.Environment,
    env_params: Any,
    metric_fn: MetricFn,
    num_seeds: int = 128,
    max_steps_in_episode: Optional[int] = None,
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
    vmap_collect = jax.vmap(evaluate_single, in_axes=(None, None, None, 0, None, None))
    return vmap_collect(act, env, env_params, seeds, max_steps_in_episode, metric_fn)

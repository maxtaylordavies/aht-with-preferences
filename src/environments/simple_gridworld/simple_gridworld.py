from enum import Enum
from typing import Optional, Tuple, Union, Dict, Any

import chex
from flax import struct
import jax
import jax.numpy as jnp
import gymnax.environments.environment as environment
from gymnax.environments import spaces


class Action(Enum):
    NONE = 0
    NORTH = 1
    SOUTH = 2
    WEST = 3
    EAST = 4


ACTION_TO_DIRECTION = jnp.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]])


@struct.dataclass
class GridworldEnvState(environment.EnvState):
    agent_loc: chex.Array
    goal_loc: chex.Array
    time: int


@struct.dataclass
class GridworldEnvParams(environment.EnvParams):
    max_steps_in_episode: int = 20


class SimpleGridworldEnv(environment.Environment):
    def __init__(
        self,
        grid_size: int,
        penalty=0.01,
    ):
        super().__init__()

        self.grid_size = grid_size
        self.penalty = penalty
        self.rendering_initialized = False
        self.viewer = None

    def reset_env(
        self, key: chex.PRNGKey, params: GridworldEnvParams
    ) -> Tuple[chex.Array, GridworldEnvState]:
        """Reset environment state"""
        agent_loc = self.spawn_agent(key)
        goal_loc = self.spawn_goal(key)
        state = GridworldEnvState(
            agent_loc=agent_loc,
            goal_loc=goal_loc,
            time=0,
        )
        return self.get_obs(state, params), state

    def step_env(
        self,
        key: chex.PRNGKey,
        state: GridworldEnvState,
        action: Union[int, float, chex.Array],
        params: GridworldEnvParams,
    ) -> Tuple[
        chex.ArrayNumpy,
        GridworldEnvState,
        chex.ArrayNumpy,
        chex.ArrayNumpy,
        Dict[Any, Any],
    ]:
        """Perform single timestep state transition."""
        # check action is valid
        valid = self.is_valid_action(action, state)
        action = jnp.where(valid, action, Action.NONE.value)

        # process movement
        new_loc = state.agent_loc + ACTION_TO_DIRECTION[action]

        # Update state dict and evaluate termination conditions
        state = GridworldEnvState(
            agent_loc=new_loc,
            goal_loc=state.goal_loc,
            time=state.time + 1,
        )
        goal_reached = jnp.all(state.agent_loc == state.goal_loc)
        done = jnp.logical_or(state.time >= params.max_steps_in_episode, goal_reached)
        return (  # type: ignore
            jax.lax.stop_gradient(self.get_obs(state)),
            jax.lax.stop_gradient(state),
            jnp.where(goal_reached, 1.0, -self.penalty),
            done,
            {},
        )

    def is_valid_action(
        self,
        action: chex.Array,
        state: GridworldEnvState,
    ) -> chex.Array:
        """Check if action is valid for agent at agent_idx."""
        new_pos = state.agent_loc + ACTION_TO_DIRECTION[action]
        out_of_bounds = jnp.logical_or(
            jnp.any(new_pos < 0), jnp.any(new_pos >= self.grid_size)
        )
        return jnp.logical_not(out_of_bounds)

    def spawn_agent(self, key: chex.PRNGKey) -> chex.Array:
        return jax.random.randint(key, (2,), 1, self.grid_size - 1)

    def spawn_goal(self, key: chex.PRNGKey) -> chex.Array:
        corners = jnp.array(
            [
                [0, 0],
                [0, self.grid_size - 1],
                [self.grid_size - 1, 0],
                [self.grid_size - 1, self.grid_size - 1],
            ]
        )
        return corners[jax.random.randint(key, (), 0, 4)]

    def get_obs(self, state: GridworldEnvState, params=None, key=None) -> chex.Array:  # type: ignore
        return jnp.concatenate([state.agent_loc, state.goal_loc])

    def action_space(
        self, params: Optional[GridworldEnvParams] = None
    ) -> spaces.Discrete:
        return spaces.Discrete(len(Action))

    def observation_space(self, params: Optional[GridworldEnvParams]) -> spaces.Box:
        min_obs = jnp.zeros((4,), dtype=jnp.float32)
        max_obs = (self.grid_size - 1) * jnp.ones((4,), dtype=jnp.float32)
        return spaces.Box(min_obs, max_obs, shape=(4,), dtype=jnp.float32)

    @property
    def name(self) -> str:
        """Environment name."""
        return "SimpleGridworld"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return len(Action)

    @property
    def default_params(self) -> GridworldEnvParams:
        return GridworldEnvParams()

    def _init_render(self):
        from .rendering import Viewer

        self.viewer = Viewer((self.grid_size, self.grid_size))
        self.rendering_initialized = True

    def render(self, state: GridworldEnvState, mode="human"):
        if not self.rendering_initialized:
            self._init_render()
        if self.viewer is None:
            return
        return_rgb = mode == "rgb_array"
        return self.viewer.render(state, return_rgb_array=return_rgb)

    def close(self):
        if self.viewer:
            self.viewer.close()

    def heuristic_policy(
        self,
        key: chex.PRNGKey,
        obs: chex.Array,
    ) -> chex.Array:
        agent_pos, goal_pos = obs[:2], obs[2:]
        direction = jnp.sign(goal_pos - agent_pos)
        vertical_action = (Action.NORTH.value * (direction[0] == -1)) + (
            Action.SOUTH.value * (direction[0] == 1)
        )
        horizontal_action = (Action.WEST.value * (direction[1] == -1)) + (
            Action.EAST.value * (direction[1] == 1)
        )

        # randomly choose between vertical and horizontal movement if both are valid
        choice = jax.random.randint(key, (), 0, 2)
        action = jnp.where(choice == 0, vertical_action, horizontal_action)

        # if move action is 0, set move action to the maximum of the two
        return jnp.where(
            action == 0, jnp.maximum(vertical_action, horizontal_action), action
        )


def make(
    grid_size=10,
    penalty=0.01,
) -> Tuple[SimpleGridworldEnv, GridworldEnvParams]:
    env = SimpleGridworldEnv(
        grid_size=grid_size,
        penalty=penalty,
    )
    return env, env.default_params

from collections import namedtuple, defaultdict
from enum import Enum
from itertools import product
from typing import Optional, Tuple, Union, Dict, Any

import chex
from flax import struct
import jax
import jax.numpy as jnp
import gymnax.environments.environment as environment
from gymnax.environments import spaces

from src.utils import to_one_hot


class Action(Enum):
    NONE = 0
    NORTH = 1
    SOUTH = 2
    WEST = 3
    EAST = 4
    LOAD = 5


ACTION_TO_DIRECTION = jnp.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1], [0, 0]])


def get_prefs_support(n_fruit_types: int) -> chex.Array:
    return jnp.array(list(product([0.0, 1.0], repeat=n_fruit_types)))


@struct.dataclass
class LBFEnvState(environment.EnvState):
    agent_locs: chex.Array
    agent_levels: chex.Array
    agent_types: chex.Array
    fruit_locs: chex.Array
    fruit_levels: chex.Array
    fruit_types: chex.Array
    fruit_consumed: chex.Array
    goals_attempted: chex.Array
    max_available_reward: float
    time: int


@struct.dataclass
class NPCPolicyParams:
    from_centre: bool = True
    aware_of_level: bool = False
    aware_of_prefs: bool = True


@struct.dataclass
class LBFEnvParams(environment.EnvParams):
    learner_agent_type: int = 6
    npc_policy_params: NPCPolicyParams = NPCPolicyParams()
    npc_type_dist: chex.Array = jnp.array(-1.0)
    normalise_reward: bool = True
    max_steps_in_episode: int = 50


class LBFEnv(environment.Environment):
    def __init__(
        self,
        grid_size: int,
        n_players: int,
        n_fruits: int,
        n_fruit_types: int,
        max_player_level: int,
        penalty=0.0,
    ):
        super().__init__()

        self.grid_size = grid_size
        self.n_players = n_players
        self.n_fruits = n_fruits
        self.n_fruit_types = n_fruit_types
        self.max_player_level = max_player_level
        self.penalty = penalty

        self.prefs_support = get_prefs_support(self.n_fruit_types)
        self.n_agent_types = len(self.prefs_support)
        self.default_type_dist = jnp.concatenate(
            [jnp.array([0.0]), jnp.ones((self.n_agent_types - 1,))]
        )
        self.default_type_dist /= self.default_type_dist.sum()

        self.rendering_initialized = False
        self.viewer = None

    def reset_env(
        self, key: chex.PRNGKey, params: LBFEnvParams
    ) -> Tuple[chex.Array, LBFEnvState]:
        """Reset environment state"""
        agent_locs, agent_levels, agent_types = self.spawn_agents(key, params)
        fruit_locs, fruit_levels, fruit_types, fruits_consumed = self.spawn_fruits(
            key, agent_locs, agent_levels
        )
        state = LBFEnvState(
            agent_locs=agent_locs,
            agent_levels=agent_levels,
            agent_types=agent_types,
            fruit_locs=fruit_locs,
            fruit_levels=fruit_levels,
            fruit_types=fruit_types,
            fruit_consumed=fruits_consumed,
            goals_attempted=jnp.zeros_like(fruit_types, dtype=jnp.int32),
            max_available_reward=1.0,
            time=0,
        )
        max_reward = self.compute_max_available_reward(state)
        max_reward = jnp.where(params.normalise_reward, max_reward, 1.0)
        state = state.replace(max_available_reward=max_reward)
        return self.get_obs(state, params), state

    def step_env(
        self,
        key: chex.PRNGKey,
        state: LBFEnvState,
        action: Union[int, float, chex.Array],
        params: LBFEnvParams,
    ) -> Tuple[chex.Array, LBFEnvState, chex.Array, chex.Array, Dict[Any, Any]]:
        """Perform single timestep state transition."""
        rewards = jnp.zeros((self.n_players,))

        npc_actions = jax.vmap(
            self.heuristic_policy,
            in_axes=(0, None, 0, None),
        )(
            jax.random.split(key, self.n_players - 1),
            state,
            jnp.arange(1, self.n_players),
            params.npc_policy_params,
        )
        actions = jnp.concatenate([jnp.array([action]), npc_actions], dtype=jnp.int32)

        # check valid actions
        valid_actions = jax.vmap(
            self.is_valid_action, in_axes=(0, 0, None), out_axes=0
        )(actions, jnp.arange(self.n_players), state)
        actions = jnp.where(valid_actions, actions, Action.NONE.value)

        # PROCESS MOVEMENTS
        new_locs = jnp.copy(state.agent_locs)
        for i in range(self.n_players):
            maybe_new_loc = state.agent_locs[i] + ACTION_TO_DIRECTION[actions[i]]
            occupied_by_agent = jnp.any(jnp.all(new_locs == maybe_new_loc, axis=-1))
            occupied_by_fruit = jnp.any(
                jnp.logical_and(
                    jnp.all(state.fruit_locs == maybe_new_loc, axis=-1),
                    jnp.logical_not(state.fruit_consumed),
                )
            )
            occupied = jnp.logical_or(occupied_by_agent, occupied_by_fruit)
            new_locs = new_locs.at[i].set(
                jnp.where(occupied, state.agent_locs[i], maybe_new_loc)
            )

        # PROCESS FRUIT COLLECTION
        trying_to_load = (actions == Action.LOAD.value).astype(int)
        consumed = jnp.copy(state.fruit_consumed)
        learner_goal_attempts = jnp.zeros_like(state.fruit_consumed)
        for i in range(self.n_fruits):
            # determine if fruit is successfully collected
            attempting = self.adjacent_agents(i, state) * trying_to_load
            learner_goal_attempts = learner_goal_attempts.at[i].add(attempting[0])
            level_sum = (state.agent_levels * attempting).sum()
            success = jnp.logical_and(
                level_sum >= state.fruit_levels[i], jnp.logical_not(consumed[i])
            )
            # increment rewards
            rs = jax.vmap(self.compute_reward, in_axes=(0, None, None))(
                state.agent_types, state.fruit_types[i], state.fruit_levels[i]
            )
            rewards += rs * success * attempting
            # remove fruit (mark as consumed)
            consumed = consumed.at[i].set(jnp.logical_or(consumed[i], success))

        # Update state dict and evaluate termination conditions
        state = LBFEnvState(
            agent_locs=new_locs,
            agent_levels=state.agent_levels,
            agent_types=state.agent_types,
            fruit_locs=state.fruit_locs,
            fruit_levels=state.fruit_levels,
            fruit_types=state.fruit_types,
            fruit_consumed=consumed,
            goals_attempted=state.goals_attempted | learner_goal_attempts,
            max_available_reward=state.max_available_reward,
            time=state.time + 1,
        )
        done = self.is_terminal(state, params)
        reward_norm_factor = jnp.where(
            state.max_available_reward == 0, 1.0, state.max_available_reward
        )
        r = rewards[0] / reward_norm_factor
        return (  # type: ignore
            jax.lax.stop_gradient(self.get_obs(state)),
            jax.lax.stop_gradient(state),
            r,
            done,
            {},
        )

    def compute_reward(
        self, agent_type: int, fruit_type: int, fruit_level: int
    ) -> chex.Array:
        return fruit_level * self.prefs_support[agent_type][fruit_type]

    def compute_max_available_reward(self, state: LBFEnvState) -> chex.Array:
        max_reward = 0.0
        for i in range(self.n_fruits):
            u_self = self.compute_reward(
                state.agent_types[0], state.fruit_types[i], state.fruit_levels[i]
            )
            u_npc = self.compute_reward(
                state.agent_types[1], state.fruit_types[i], state.fruit_levels[i]
            )
            obtainable = jnp.logical_and(
                u_self > 0,
                jnp.logical_or(
                    state.fruit_levels[i] <= state.agent_levels[0], u_npc > 0
                ),
            )
            max_reward += jnp.where(obtainable, u_self, 0.0)
        return max_reward

    def is_valid_action(
        self,
        action: chex.Array,
        agent_idx: chex.Array,
        state: LBFEnvState,
    ) -> chex.Array:
        """Check if action is valid for agent at agent_idx."""
        new_pos = state.agent_locs[agent_idx] + ACTION_TO_DIRECTION[action]
        out_of_bounds = jnp.logical_or(
            jnp.any(new_pos < 0), jnp.any(new_pos >= self.grid_size)
        )
        return jnp.logical_not(out_of_bounds)

    def adjacent_fruit(self, agent_idx: chex.Array, state: LBFEnvState) -> chex.Array:
        """return index of first adjacent fruit or -1 if no adjacent fruit"""
        agent_pos = state.agent_locs[agent_idx]
        is_adjacent = jnp.logical_and(
            jnp.abs(state.fruit_locs - agent_pos).sum(axis=-1) == 1,
            jnp.logical_not(state.fruit_consumed),
        )
        return jnp.where(is_adjacent, size=1, fill_value=-1)[0][0]

    def adjacent_agents(self, fruit_idx: int, state: LBFEnvState) -> chex.Array:
        agent_dists = jnp.abs(state.agent_locs - state.fruit_locs[fruit_idx]).sum(
            axis=-1
        )
        return (agent_dists == 1.0).astype(int)

    def spawn_agents(
        self, key: chex.PRNGKey, params: LBFEnvParams
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        # sample types
        type_dist: chex.Array = jnp.where(  # type: ignore
            params.npc_type_dist == -1.0, self.default_type_dist, params.npc_type_dist
        )
        type_dist /= type_dist.sum()
        npc_types = jax.random.choice(
            key, jnp.arange(self.n_agent_types), (self.n_players - 1,), p=type_dist
        )
        types = jnp.concatenate([jnp.array([params.learner_agent_type]), npc_types])

        # sample levels
        levels = jax.random.randint(
            key, (self.n_players,), 1, self.max_player_level + 1
        )

        # sample positions
        positions = jax.random.permutation(key, jnp.arange(self.grid_size**2))[
            : self.n_players
        ]
        positions = jnp.unravel_index(positions, (self.grid_size, self.grid_size))
        positions = jnp.stack(positions, axis=-1)

        return positions, levels, types

    def spawn_fruits(
        self, key: chex.PRNGKey, agent_locs: chex.Array, agent_levels: chex.Array
    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
        # sample types
        types = jax.random.randint(key, (self.n_fruits,), 0, self.n_fruit_types)

        # sample levels
        max_level = agent_levels.sum()
        levels = jax.random.randint(key, (self.n_fruits,), 1, max_level + 1)

        # sample positions without replacement
        # need to ensure we don't sample locations containing agents
        probs = jnp.ones((self.grid_size, self.grid_size), dtype=jnp.float32)
        probs = probs.at[agent_locs].set(0.0)
        probs = probs.reshape(-1) / probs.sum()
        positions = jax.random.choice(
            key, jnp.arange(self.grid_size**2), (self.n_fruits,), replace=False, p=probs
        )
        positions = jnp.unravel_index(positions, (self.grid_size, self.grid_size))
        positions = jnp.stack(positions, axis=-1)

        return positions, levels, types, jnp.zeros((self.n_fruits,), dtype=jnp.int32)

    def get_obs(self, state: LBFEnvState, params=None, key=None) -> chex.Array:  # type: ignore
        """Applies observation function to state."""
        fruit_chunk_len = 3 + self.n_fruit_types
        agent_chunk_len = 3 + self.n_agent_types

        # initialize observation vector
        obs = jnp.zeros(
            (fruit_chunk_len * self.n_fruits + agent_chunk_len * self.n_players,),
            dtype=jnp.float32,
        )

        # fruits
        for i in range(self.n_fruits):
            start, end = fruit_chunk_len * i, fruit_chunk_len * (i + 1)
            chunk = jnp.array(
                [
                    state.fruit_locs[i][0],
                    state.fruit_locs[i][1],
                    state.fruit_levels[i],
                    *to_one_hot(state.fruit_types[i], self.n_fruit_types),
                ]
            )
            # if fruit is consumed, set all values to -1
            chunk = jnp.where(state.fruit_consumed[i], -1, chunk)
            obs = obs.at[start:end].set(chunk)

        # agents
        for i in range(self.n_players):
            start, end = (
                fruit_chunk_len * self.n_fruits + agent_chunk_len * i,
                fruit_chunk_len * self.n_fruits + agent_chunk_len * (i + 1),
            )
            chunk = jnp.array(
                [
                    state.agent_locs[i][0],
                    state.agent_locs[i][1],
                    state.agent_levels[i],
                    *to_one_hot(state.agent_types[i], self.n_agent_types),
                ]
            )
            obs = obs.at[start:end].set(chunk)

        return obs

    def is_terminal(self, state: LBFEnvState, params: LBFEnvParams) -> chex.Array:  # type: ignore
        """Check whether state is terminal."""
        max_steps_reached = state.time >= params.max_steps_in_episode
        all_fruits_consumed = state.fruit_consumed.all()
        return jnp.logical_or(max_steps_reached, all_fruits_consumed)

    def action_space(self, params: Optional[LBFEnvParams] = None) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(len(Action))

    def observation_space(self, params: Optional[LBFEnvParams]) -> spaces.Box:
        """Observation space of the environment."""
        fruit_chunk_len = 3 + self.n_fruit_types
        agent_chunk_len = 3 + self.n_agent_types
        obs_len = fruit_chunk_len * self.n_fruits + agent_chunk_len * self.n_players

        min_obs = -1.0 * jnp.ones((obs_len,), dtype=jnp.float32)

        max_loc = self.grid_size - 1
        max_fruit_level = self.n_fruits * self.max_player_level
        max_fruit_chunk = jnp.array(
            [max_loc, max_loc, max_fruit_level, *jnp.ones(self.n_fruit_types)]
        )
        max_agent_chunk = jnp.array(
            [max_loc, max_loc, self.max_player_level, *jnp.ones(self.n_agent_types)]
        )
        max_obs = jnp.concatenate(
            [max_fruit_chunk] * self.n_fruits + [max_agent_chunk] * self.n_players
        )

        return spaces.Box(min_obs, max_obs, shape=(obs_len,), dtype=jnp.float32)

    @property
    def name(self) -> str:
        """Environment name."""
        return "lbf"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return len(Action)

    @property
    def default_params(self) -> LBFEnvParams:
        return LBFEnvParams()

    def _init_render(self):
        from .rendering import Viewer

        self.viewer = Viewer((self.grid_size, self.grid_size))
        self.rendering_initialized = True

    def render(self, state: LBFEnvState, mode="human"):
        if not self.rendering_initialized:
            self._init_render()
        if self.viewer is None:
            return
        return_rgb = mode == "rgb_array"
        return self.viewer.render(state, return_rgb_array=return_rgb)

    def close(self):
        if self.viewer:
            self.viewer.close()

    def centre_of_players(self, state: LBFEnvState) -> chex.Array:
        return jnp.round(state.agent_locs.mean(axis=0)).astype(int)

    def fruit_distances(self, state: LBFEnvState, pos: chex.Array) -> chex.Array:
        return jnp.abs(state.fruit_locs - pos).sum(axis=-1)

    def fruit_values(self, state: LBFEnvState, agent_idx) -> chex.Array:
        prefs = self.prefs_support[state.agent_types[agent_idx]]
        vals = prefs[state.fruit_types]
        return jnp.where(state.fruit_consumed, 0, state.fruit_levels * vals)

    def load_or_move_towards(
        self,
        key: chex.PRNGKey,
        agent_pos: chex.Array,
        fruit_pos: chex.Array,
        state: LBFEnvState,
    ) -> chex.Array:
        distance = jnp.abs(fruit_pos - agent_pos).sum()

        # first get the locations of tiles adjacent to fruit_pos
        adjacents = jnp.array(
            [
                [fruit_pos[0] - 1, fruit_pos[1]],
                [fruit_pos[0] + 1, fruit_pos[1]],
                [fruit_pos[0], fruit_pos[1] - 1],
                [fruit_pos[0], fruit_pos[1] + 1],
            ]
        )

        # then check which adjacent locations are valid targets
        # i.e. not out of bounds and not occupied by another agent or fruit
        valid = jnp.zeros((4,), dtype=jnp.int32)
        for i in range(4):
            out_of_bounds = jnp.logical_or(
                jnp.any(adjacents[i] < 0), jnp.any(adjacents[i] >= self.grid_size)
            )
            occupied_by_agent = jnp.any(
                jnp.all(state.agent_locs == adjacents[i], axis=-1)
            )
            occupied_by_fruit = jnp.any(
                jnp.all(state.fruit_locs == adjacents[i], axis=-1)
            )
            occupied = jnp.logical_or(occupied_by_agent, occupied_by_fruit)
            valid = valid.at[i].set(
                jnp.logical_not(jnp.logical_or(out_of_bounds, occupied))
            )

        # find the closest valid adjacent location
        distances = jnp.abs(adjacents - agent_pos).sum(axis=-1)
        distances = jnp.where(valid, distances, jnp.inf)
        closest = adjacents[distances.argmin()]

        # get the direction to move in
        direction = jnp.sign(closest - agent_pos)
        vertical_action = (Action.NORTH.value * (direction[0] == -1)) + (
            Action.SOUTH.value * (direction[0] == 1)
        )
        horizontal_action = (Action.WEST.value * (direction[1] == -1)) + (
            Action.EAST.value * (direction[1] == 1)
        )

        # randomly choose between vertical and horizontal movement if both are valid
        choice = jax.random.randint(key, (), 0, 2)
        move_action = jnp.where(choice == 0, vertical_action, horizontal_action)

        # if move action is 0, set move action to the maximum of the two
        move_action = jnp.where(
            move_action == 0,
            jnp.maximum(vertical_action, horizontal_action),
            move_action,
        )

        # if we're adjacent to the fruit, load it, otherwise return move action
        return jnp.where(distance == 1, Action.LOAD.value, move_action)

    def heuristic_policy(
        self,
        key: chex.PRNGKey,
        state: LBFEnvState,
        agent_idx: int,
        params: NPCPolicyParams,
    ) -> chex.Array:
        agent_pos = state.agent_locs[agent_idx]
        centre = self.centre_of_players(state)
        start = jnp.where(params.from_centre, centre, agent_pos)

        values = self.fruit_values(state, agent_idx)
        min_value = jnp.where(params.aware_of_prefs, values.max(), values.min())
        max_level = jnp.where(
            params.aware_of_level, state.agent_levels[agent_idx], jnp.inf
        )

        dists = self.fruit_distances(state, start)
        dists = jnp.where(values < min_value, jnp.inf, dists)
        dists = jnp.where(state.fruit_levels > max_level, jnp.inf, dists)
        dists = jnp.where(state.fruit_consumed, jnp.inf, dists)

        best = jnp.argmin(dists)
        action = self.load_or_move_towards(
            key, state.agent_locs[agent_idx], state.fruit_locs[best], state
        )
        return jnp.where(dists[best] == jnp.inf, Action.NONE.value, action)

    def reference_policy(
        self, key: chex.PRNGKey, state: LBFEnvState, agent_idx: int
    ) -> chex.Array:
        self_idx, npc_idx = agent_idx, 1 - agent_idx
        own_fruit_vals = self.fruit_values(state, self_idx)
        teammate_fruit_vals = self.fruit_values(state, npc_idx)

        attainable = jnp.logical_or(
            state.fruit_levels <= state.agent_levels[self_idx], teammate_fruit_vals > 0
        )
        own_fruit_vals *= attainable.astype(int)

        distances = self.fruit_distances(state, state.agent_locs[self_idx])
        distances = jnp.where(
            own_fruit_vals == own_fruit_vals.max(), distances, jnp.inf
        )
        best = jnp.argmin(distances)

        action = self.load_or_move_towards(
            key, state.agent_locs[self_idx], state.fruit_locs[best], state
        )

        return jnp.where(own_fruit_vals.max() == 0, Action.NONE.value, action)


def make(
    grid_size=10,
    n_players=2,
    n_fruits=5,
    n_fruit_types=3,
    max_player_level=2,
) -> Tuple[LBFEnv, LBFEnvParams]:
    env = LBFEnv(
        grid_size=grid_size,
        n_players=n_players,
        n_fruits=n_fruits,
        n_fruit_types=n_fruit_types,
        max_player_level=max_player_level,
    )
    return env, env.default_params

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

ACTION_TO_DIRECTION = jnp.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]])

@struct.dataclass
class PursuitEnvState(environment.EnvState):
    predator_locs: chex.Array
    predator_types: chex.Array
    prey_locs: chex.Array
    prey_types: chex.Array
    prey_consumed: chex.Array
    max_available_reward: float
    time: int


@struct.dataclass
class NPCPolicyParams:
    from_centre: bool = False
    aware_of_prefs: bool = False

@struct.dataclass
class PursuitEnvParams(environment.EnvParams):
    learner_agent_type: int = 0
    npc_type_dist: chex.Array = jnp.array(-1.0)
    npc_policy_params: NPCPolicyParams = NPCPolicyParams()
    normalise_reward: bool = False
    max_steps_in_episode: int = 100


class PursuitEnv(environment.Environment):
    def __init__(
        self,
        grid_size: int,
        n_predators: int,
        n_prey: int,
        n_prey_types: int,
        penalty=0.0,
    ):
        super().__init__()

        self.grid_size = grid_size
        self.n_predators = n_predators
        self.n_prey = n_prey
        self.n_prey_types = n_prey_types
        self.penalty = penalty

        self.prefs_support = jnp.array(
            list(product([0.0, 1.0], repeat=self.n_prey_types))
        )
        self.n_predator_types = len(self.prefs_support)
        self.default_type_dist = jnp.concatenate(
            [jnp.array([0.0]), jnp.ones((self.n_predator_types - 1,))]
        )
        self.default_type_dist /= self.default_type_dist.sum()

        self.rendering_initialized = False
        self.viewer = None

    def reset_env(
        self, key: chex.PRNGKey, params: PursuitEnvParams
    ) -> Tuple[chex.Array, PursuitEnvState]:
        """Reset environment state"""
        predator_locs, predator_types = self.spawn_predators(key, params)
        prey_locs, prey_types, prey_consumed = self.spawn_prey(key, predator_locs)
        state = PursuitEnvState(
            predator_locs=predator_locs,
            predator_types=predator_types,
            prey_locs=prey_locs,
            prey_types=prey_types,
            prey_consumed=prey_consumed,
            max_available_reward=1.0,
            time=0,
        )
        # max_reward = self.compute_max_available_reward(state)
        # max_reward = (
        #     jnp.where(params.normalise_reward, max_reward, 1.0)
        # )
        # state = state.replace(max_available_reward=max_reward)
        return self.get_obs(state, params), state

    def step_env(
        self,
        key: chex.PRNGKey,
        state: PursuitEnvState,
        action: Union[int, float, chex.Array],
        params: PursuitEnvParams,
    ) -> Tuple[chex.Array, PursuitEnvState, chex.Array, chex.Array, Dict[Any, Any]]:
        """Perform single timestep state transition."""
        rewards = jnp.zeros((self.n_predators,))

        new_predator_locs = jnp.copy(state.predator_locs)
        new_prey_locs = jnp.copy(state.prey_locs)
        consumed = jnp.copy(state.prey_consumed)

        prey_actions = jax.random.randint(key, (self.n_prey,), 0, 5)
        npc_actions = jax.vmap(
            self.heuristic_policy,
            in_axes=(None, 0, None),
        )(
            state,
            jnp.arange(1, self.n_predators),
            params.npc_policy_params,
        )
        actions = jnp.concatenate([jnp.array([action]), npc_actions], dtype=jnp.int32)

        # PROCESS PREY CONSUMPTION / ACTIONS
        for i in range(self.n_prey):
            # determine if prey is surrounded
            surrounding = self.surrounding(i, state)
            success = jnp.logical_and(
                surrounding.sum() >= 3,
                jnp.logical_not(consumed[i])
            )
            # update rewards
            rs = jax.vmap(
                self.compute_reward, in_axes=(0, None)
            )(state.predator_types, state.prey_types[i])
            rewards += (success * rs * surrounding)
            # remove prey (mark as consumed)
            consumed = consumed.at[i].set(
                jnp.logical_or(consumed[i], success)
            )
            # if not consumed, apply random move action
            maybe_new_loc = new_prey_locs[i] + ACTION_TO_DIRECTION[prey_actions[i]]
            occupied = self.is_occupied(
                maybe_new_loc,
                new_predator_locs,
                new_prey_locs,
                consumed
            )
            new_prey_locs = new_prey_locs.at[i].set(
                jnp.where(occupied, new_prey_locs[i], maybe_new_loc)
            )

        # PROCESS PREDATOR MOVEMENTS
        for i in range(self.n_predators):
            maybe_new_loc = new_predator_locs[i] + ACTION_TO_DIRECTION[actions[i]]
            occupied = self.is_occupied(
                maybe_new_loc,
                new_predator_locs,
                new_prey_locs,
                consumed
            )
            new_predator_locs = new_predator_locs.at[i].set(
                jnp.where(occupied, new_predator_locs[i], maybe_new_loc)
            )

        # Update state dict and evaluate termination conditions
        state = PursuitEnvState(
            predator_locs=new_predator_locs,
            predator_types=state.predator_types,
            prey_locs=new_prey_locs,
            prey_types=state.prey_types,
            prey_consumed=consumed,
            max_available_reward=state.max_available_reward,
            time=state.time + 1,
        )
        done = self.is_terminal(state, params)
        reward_norm_factor = jnp.where(state.max_available_reward == 0, 1.0, state.max_available_reward)
        r = rewards[0] / reward_norm_factor
        return (  # type: ignore
            jax.lax.stop_gradient(self.get_obs(state)),
            jax.lax.stop_gradient(state),
            r,
            done,
            {},
        )

    def is_occupied(
        self,
        loc: chex.Array,
        predator_locs: chex.Array,
        prey_locs: chex.Array,
        prey_consumed: chex.Array
    ) -> chex.Array:
        occupied_by_predator = jnp.any(
            jnp.all(predator_locs == loc, axis=-1)
        )
        occupied_by_prey = jnp.any(
            jnp.logical_and(
                jnp.all(prey_locs == loc, axis=-1),
                jnp.logical_not(prey_consumed),
            )
        )
        return jnp.logical_or(occupied_by_predator, occupied_by_prey)

    def compute_reward(self, predator_type: int, prey_type: int) -> chex.Array:
        return self.prefs_support[predator_type][prey_type]

    # def compute_max_available_reward(self, state: PursuitEnvState) -> chex.Array:
    #     max_reward = 0.0
    #     for i in range(self.n_fruits):
    #         u_self = self.compute_reward(state.agent_types[0], state.fruit_types[i], state.fruit_levels[i])
    #         u_npc = self.compute_reward(state.agent_types[1], state.fruit_types[i], state.fruit_levels[i])
    #         can_solo = (state.fruit_levels[i] <= state.agent_levels[0]).astype(int)
    #         obtainable = jnp.logical_and(
    #             u_self > 0,
    #             jnp.logical_or(
    #                 can_solo,
    #                 u_npc > 0
    #             )
    #         )
    #         max_reward += jnp.where(obtainable, u_self, 0.0)
    #     return max_reward

    def surrounding(self, prey_idx: int, state: PursuitEnvState) -> chex.Array:
        dists = jnp.abs(state.predator_locs - state.prey_locs[prey_idx]).sum(
            axis=-1
        )
        return (dists <= 1.0).astype(int)

    def spawn_predators(self, key: chex.PRNGKey, params: PursuitEnvParams) -> Tuple[chex.Array, chex.Array]:
        # sample types
        type_dist: chex.Array = jnp.where( # type: ignore
            params.npc_type_dist == -1.0,
            self.default_type_dist,
            params.npc_type_dist
        )
        type_dist /= type_dist.sum()
        npc_types = jax.random.choice(
            key, jnp.arange(self.n_predator_types), (self.n_predators - 1,), p=type_dist
        )
        types = jnp.concatenate([jnp.array([params.learner_agent_type]), npc_types])

        # sample positions
        positions = jax.random.permutation(key, jnp.arange(self.grid_size**2))[
            : self.n_predators
        ]
        positions = jnp.unravel_index(positions, (self.grid_size, self.grid_size))
        positions = jnp.stack(positions, axis=-1)

        return positions, types

    def spawn_prey(self, key: chex.PRNGKey, predator_locs: chex.Array) -> Tuple[chex.Array, chex.Array, chex.Array]:
        # sample types
        types = jax.random.randint(key, (self.n_prey,), 0, self.n_prey_types)

        # sample positions without replacement
        # need to ensure we don't sample locations containing agents
        probs = jnp.ones((self.grid_size, self.grid_size), dtype=jnp.float32)
        probs = probs.at[predator_locs].set(0.0)
        probs = probs.reshape(-1) / probs.sum()
        positions = jax.random.choice(
            key, jnp.arange(self.grid_size**2), (self.n_prey,), replace=False, p=probs
        )
        positions = jnp.unravel_index(positions, (self.grid_size, self.grid_size))
        positions = jnp.stack(positions, axis=-1)

        return positions, types, jnp.zeros((self.n_prey,), dtype=jnp.int32)

    def get_obs(self, state: PursuitEnvState, params=None, key=None) -> chex.Array:  # type: ignore
        """Applies observation function to state."""
        prey_chunk_len = 2 + self.n_prey_types
        predator_chunk_len = 2 + self.n_predator_types

        # initialize observation vector
        obs = jnp.zeros(
            (prey_chunk_len * self.n_prey + predator_chunk_len * self.n_predators,),
            dtype=jnp.float32,
        )

        # prey
        for i in range(self.n_prey):
            start, end = prey_chunk_len * i, prey_chunk_len * (i + 1)
            chunk = jnp.array(
                [
                    state.prey_locs[i][0],
                    state.prey_locs[i][1],
                    *to_one_hot(state.prey_types[i], self.n_prey_types),
                ]
            )
            # if prey is consumed, set all values to -1
            chunk = jnp.where(state.prey_consumed[i], -1, chunk)
            obs = obs.at[start:end].set(chunk)

        # predators
        for i in range(self.n_predators):
            start, end = (
                prey_chunk_len * self.n_prey + predator_chunk_len * i,
                prey_chunk_len * self.n_prey + predator_chunk_len * (i + 1),
            )
            chunk = jnp.array(
                [
                    state.predator_locs[i][0],
                    state.predator_locs[i][1],
                    *to_one_hot(state.predator_types[i], self.n_predator_types),
                ]
            )
            obs = obs.at[start:end].set(chunk)

        return obs

    def is_terminal(self, state: PursuitEnvState, params: PursuitEnvParams) -> chex.Array:  # type: ignore
        """Check whether state is terminal."""
        max_steps_reached = state.time >= params.max_steps_in_episode
        all_prey_consumed = state.prey_consumed.all()
        return jnp.logical_or(max_steps_reached, all_prey_consumed)

    def action_space(self, params: Optional[PursuitEnvParams] = None) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(len(Action))

    def observation_space(self, params: Optional[PursuitEnvParams]) -> spaces.Box:
        """Observation space of the environment."""
        prey_chunk_len = 3 + self.n_prey_types
        predator_chunk_len = 3 + self.n_predator_types
        obs_len = prey_chunk_len * self.n_prey + predator_chunk_len * self.n_predators

        min_obs = -1.0 * jnp.ones((obs_len,), dtype=jnp.float32)

        max_loc = self.grid_size - 1
        max_prey_chunk = jnp.array(
            [max_loc, max_loc, *jnp.ones(self.n_prey_types)]
        )
        max_predator_chunk = jnp.array(
            [max_loc, max_loc, *jnp.ones(self.n_predator_types)]
        )
        max_obs = jnp.concatenate(
            [max_prey_chunk] * self.n_prey + [max_predator_chunk] * self.n_predators
        )

        return spaces.Box(min_obs, max_obs, shape=(obs_len,), dtype=jnp.float32)

    @property
    def name(self) -> str:
        """Environment name."""
        return "Pursuit"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return len(Action)

    @property
    def default_params(self) -> PursuitEnvParams:
        return PursuitEnvParams()

    def _init_render(self):
        from .rendering import Viewer
        self.viewer = Viewer((self.grid_size, self.grid_size))
        self.rendering_initialized = True

    def render(self, state: PursuitEnvState, mode="human"):
        if not self.rendering_initialized:
            self._init_render()
        if self.viewer is None:
            return
        return_rgb = mode == "rgb_array"
        return self.viewer.render(state, return_rgb_array=return_rgb)

    def close(self):
        if self.viewer:
            self.viewer.close()

    def centre_of_players(self, state: PursuitEnvState) -> chex.Array:
        return jnp.round(state.predator_locs.mean(axis=0)).astype(int)

    def prey_distances(self, state: PursuitEnvState, pos: chex.Array) -> chex.Array:
        return jnp.abs(state.prey_locs - pos).sum(axis=-1)

    def prey_values(self, state: PursuitEnvState, agent_idx) -> chex.Array:
        return self.prefs_support[state.predator_types[agent_idx]][state.prey_types]

    def surround_or_move_towards(self, key: chex.PRNGKey, predator_pos: chex.Array, prey_pos: chex.Array, state: PursuitEnvState) -> chex.Array:
        distance = jnp.abs(prey_pos - predator_pos).sum()

        # first get the locations of tiles adjacent to prey_pos
        adjacents = jnp.array([
            [prey_pos[0] - 1, prey_pos[1]],
            [prey_pos[0] + 1, prey_pos[1]],
            [prey_pos[0], prey_pos[1] - 1],
            [prey_pos[0], prey_pos[1] + 1],
        ])

        # then check which adjacent locations are valid targets
        # i.e. not out of bounds and not occupied by another predator or prey
        valid = jnp.zeros((4,), dtype=jnp.int32)
        for i in range(4):
            out_of_bounds = jnp.logical_or(
                jnp.any(adjacents[i] < 0), jnp.any(adjacents[i] >= self.grid_size)
            )
            occupied = self.is_occupied(
                adjacents[i], state.predator_locs, state.prey_locs, state.prey_consumed
            )
            valid = valid.at[i].set(jnp.logical_not(jnp.logical_or(out_of_bounds, occupied)))

        # find the closest valid adjacent location
        distances = jnp.abs(adjacents - predator_pos).sum(axis=-1)
        distances = jnp.where(valid, distances, jnp.inf)
        closest = adjacents[distances.argmin()]

        # get the direction to move in
        direction = jnp.sign(closest - predator_pos)
        vertical_action = (Action.NORTH.value * (direction[0] == -1)) + (Action.SOUTH.value * (direction[0] == 1))
        horizontal_action = (Action.WEST.value * (direction[1] == -1)) + (Action.EAST.value * (direction[1] == 1))

        # randomly choose between vertical and horizontal movement if both are valid
        choice = jax.random.randint(key, (), 0, 2)
        move_action = jnp.where(choice == 0, vertical_action, horizontal_action)

        # if move action is 0, set move action to the maximum of the two
        move_action = jnp.where(
            move_action == 0,
            jnp.maximum(vertical_action, horizontal_action),
            move_action
        )

        # if we're already adjacent to the prey do nothing, otherwise return move action
        return jnp.where(distance == 1, Action.NONE.value, move_action)

    def heuristic_policy(
        self,
        key: chex.PRNGKey,
        state: PursuitEnvState,
        agent_idx: int,
        params: NPCPolicyParams,
    ) -> chex.Array:
        self_pos = state.predator_locs[agent_idx]
        centre = self.centre_of_players(state)
        start = jnp.where(params.from_centre, centre, self_pos)

        values = self.prey_values(state, agent_idx)
        min_value = jnp.where(params.aware_of_prefs, values.max(), values.min())

        dists = self.prey_distances(state, start)
        dists = jnp.where(values < min_value, jnp.inf, dists)
        dists = jnp.where(state.prey_consumed, jnp.inf, dists)

        best = jnp.argmin(dists)
        action = self.surround_or_move_towards(
            key,
            state.predator_locs[agent_idx],
            state.prey_locs[best],
            state
        )
        return jnp.where(
            dists[best] == jnp.inf,
            Action.NONE.value,
            action
        )

def make(
    grid_size=10,
    n_predators=5,
    n_prey=5,
    n_prey_types=3,
) -> Tuple[PursuitEnv, PursuitEnvParams]:
    env = PursuitEnv(
        grid_size=grid_size,
        n_predators=n_predators,
        n_prey=n_prey,
        n_prey_types=n_prey_types,
    )
    return env, env.default_params

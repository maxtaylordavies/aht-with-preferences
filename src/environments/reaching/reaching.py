from itertools import product
from typing import Optional, Tuple, Union, Dict, Any

import chex
from flax import struct
import jax
import jax.numpy as jnp
from gymnax.environments import spaces

from ..simple_gridworld import Action, ACTION_TO_DIRECTION
from ..environment import Environment, EnvState, EnvParams


def get_prefs_support() -> chex.ArrayNumpy:
    return jnp.array(list(product([0.0, 1.0], repeat=4)))


def get_goal_locs(grid_size: int) -> chex.ArrayNumpy:
    return jnp.array(  # type: ignore
        [
            [0, 0],
            [grid_size - 1, 0],
            [0, grid_size - 1],
            [grid_size - 1, grid_size - 1],
            [grid_size // 2, grid_size // 2],
        ]
    )


@struct.dataclass
class ReachingEnvState(EnvState):
    agent_locs: chex.ArrayNumpy
    agent_types: chex.ArrayNumpy
    agent_features: chex.ArrayNumpy
    agent_rewarded: chex.ArrayNumpy
    goals_attempted: chex.ArrayNumpy
    max_available_reward: float
    time: int


@struct.dataclass
class ReachingEnvParams(EnvParams):
    learner_agent_type: int = 12  # -> [1, 1, 0, 0]
    npc_type_dist: chex.ArrayNumpy = jnp.array(-1.0)  # type: ignore
    feature_noise: float = 0.05
    max_steps_in_episode: int = 20


class ReachingEnv(Environment):
    def __init__(
        self,
        grid_size: int,
        n_players: int,
        penalty: float,
    ):
        super().__init__()

        self.grid_size = grid_size
        self.n_players = n_players
        self.penalty = penalty

        self.goal_locs = get_goal_locs(self.grid_size)
        self.goal_occupancy_reqs = jnp.array([2, 2, 2, 2, 1])
        self.goal_rewards = jnp.array([1.0, 1.0, 1.0, 1.0, 0.2])

        self.prefs_support = get_prefs_support()
        self.n_agent_types = len(self.prefs_support)
        self.default_type_dist = jnp.concatenate(
            [jnp.array([0.0]), jnp.ones((self.n_agent_types - 1,))]
        )
        self.default_type_dist /= self.default_type_dist.sum()

        self.rendering_initialized = False
        self.viewer = None

    def reset_env(
        self, key: chex.PRNGKey, params: ReachingEnvParams
    ) -> Tuple[chex.Array, ReachingEnvState]:
        """Reset environment state"""
        agent_locs, agent_types = self.spawn_agents(key, params)

        agent_features = []
        for i in range(self.n_players):
            agent_features.append(
                jax.random.multivariate_normal(
                    key,
                    self.prefs_support[agent_types[i]],
                    params.feature_noise * jnp.eye(4),
                )
            )

        state = ReachingEnvState(
            agent_locs=agent_locs,
            agent_types=agent_types,
            agent_features=jnp.stack(agent_features, axis=0),
            agent_rewarded=False,
            goals_attempted=jnp.zeros(5, dtype=jnp.int32),
            max_available_reward=1.0,
            time=0,
        )
        max_reward = self.compute_max_available_reward(state)
        state = state.replace(max_available_reward=max_reward)
        return self.get_obs(state, 0, params), state

    def step_env(
        self,
        key: chex.PRNGKey,
        state: ReachingEnvState,
        action: Union[int, float, chex.Array],
        params: ReachingEnvParams,
    ) -> Tuple[
        chex.ArrayNumpy,
        ReachingEnvState,
        chex.ArrayNumpy,
        chex.ArrayNumpy,
        Dict[Any, Any],
    ]:
        """Perform single timestep state transition."""
        npc_actions, npc_goals = jax.vmap(
            self.basic_heuristic_policy,
            in_axes=(0, None, 0),
        )(
            jax.random.split(key, self.n_players - 1),
            state,
            jnp.arange(1, self.n_players),
        )
        # npc_actions = jnp.zeros_like(npc_actions)
        actions = jnp.concatenate([jnp.array([action]), npc_actions], dtype=jnp.int32)

        # check valid actions
        valid_actions = jax.vmap(
            self.is_valid_action, in_axes=(0, 0, None), out_axes=0
        )(actions, jnp.arange(self.n_players), state)

        # handle invalid actions
        actions = jnp.where(valid_actions, actions, Action.NONE.value)

        # process movements
        new_locs = state.agent_locs + ACTION_TO_DIRECTION[actions]

        # compute reward
        r = jnp.where(actions[0] == Action.NONE.value, 0.0, -self.penalty)
        is_in_goals = jnp.all(new_locs[0] == self.goal_locs, axis=-1).astype(int)
        occupancy_req = jnp.sum(self.goal_occupancy_reqs * is_in_goals)
        num_agents_in_tile = jnp.sum(
            jnp.where(jnp.all(new_locs == new_locs[0], axis=-1), 1, 0)
        )
        rewarded = jnp.logical_and(
            is_in_goals.any(), num_agents_in_tile >= occupancy_req
        )
        prefs = jnp.concatenate(
            [self.prefs_support[state.agent_types[0]], jnp.array([1.0])]
        )
        r += jnp.where(
            jnp.logical_and(rewarded, jnp.logical_not(state.agent_rewarded)),
            jnp.sum(is_in_goals * self.goal_rewards * prefs),
            0.0,
        )

        # update state dict and evaluate termination conditions
        state = ReachingEnvState(
            agent_locs=new_locs,
            agent_types=state.agent_types,
            agent_features=state.agent_features,
            agent_rewarded=rewarded,
            goals_attempted=state.goals_attempted | is_in_goals,
            max_available_reward=state.max_available_reward,
            time=state.time + 1,
        )
        obs = self.get_obs(state, 0)
        done = state.time >= params.max_steps_in_episode
        return (  # type: ignore
            jax.lax.stop_gradient(obs),
            jax.lax.stop_gradient(state),
            r,
            done,
            {
                "teammate_obs": self.get_obs(state, 1),
                "teammate_action": npc_actions[0],
                "teammate_current_goal": npc_goals[0],
                "attempted": state.goals_attempted,
            },
        )

    def compute_max_available_reward(self, state: ReachingEnvState) -> chex.Array:
        achievable = (
            self.prefs_support[state.agent_types[0]]
            * self.prefs_support[state.agent_types[1]]
        )
        achievable = jnp.concatenate([achievable, jnp.array([1.0])])
        return jnp.max(achievable * self.goal_rewards)

    def is_valid_action(
        self,
        action: chex.Array,
        agent_idx: chex.Array,
        state: ReachingEnvState,
    ) -> chex.Array:
        """Check if action is valid for agent at agent_idx."""
        new_pos = state.agent_locs[agent_idx] + ACTION_TO_DIRECTION[action]
        out_of_bounds = jnp.logical_or(
            jnp.any(new_pos < 0), jnp.any(new_pos >= self.grid_size)
        )
        in_goal = jnp.any(
            jnp.all(state.agent_locs[agent_idx] == self.goal_locs, axis=-1)
        )
        return jnp.logical_not(jnp.logical_or(out_of_bounds, in_goal))

    def spawn_agents(
        self, key: chex.PRNGKey, params: ReachingEnvParams
    ) -> Tuple[chex.Array, chex.Array]:
        # sample types
        type_dist: chex.Array = jnp.where(  # type: ignore
            params.npc_type_dist == -1.0, self.default_type_dist, params.npc_type_dist
        )
        type_dist /= type_dist.sum()
        npc_types = jax.random.choice(
            key, jnp.arange(self.n_agent_types), (self.n_players - 1,), p=type_dist
        )
        types = jnp.concatenate([jnp.array([params.learner_agent_type]), npc_types])

        # sample locations
        probs = jnp.ones((self.grid_size, self.grid_size), dtype=jnp.float32)
        probs = probs.at[self.goal_locs].set(0.0)
        probs = probs.reshape(-1) / probs.sum()
        locs = jax.random.choice(
            key,
            jnp.arange(self.grid_size**2),
            (self.n_players,),
            replace=False,
            p=probs,
        )
        locs = jnp.unravel_index(locs, (self.grid_size, self.grid_size))

        return jnp.stack(locs, axis=-1), types

    def get_obs(self, state: ReachingEnvState, agent_idx: int, params=None, key=None) -> chex.Array:  # type: ignore
        teammate_idx = 1 - agent_idx

        own_chunk = state.agent_locs[agent_idx]
        teammate_chunk = jnp.concatenate(
            [
                state.agent_locs[teammate_idx],
                state.agent_features[teammate_idx],
            ]
        )
        return jnp.concatenate([own_chunk, teammate_chunk])

    def get_type(self, obs: chex.Array) -> chex.Array:
        type_vec = obs[-self.n_agent_types :]
        return jnp.argmax(type_vec)

    def get_setting(self, obs: chex.Array) -> chex.Array:
        type = self.get_type(obs)
        return jnp.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])[type]

    def get_nearest_goal(self, obs: chex.Array, agent_idx: int) -> chex.Array:
        loc = obs[2 * agent_idx : (2 * agent_idx) + 2]
        dists = jnp.abs(self.goal_locs - loc).sum(axis=-1)
        return jnp.argmin(dists)

    def action_space(
        self, params: Optional[ReachingEnvParams] = None
    ) -> spaces.Discrete:
        return spaces.Discrete(len(Action))

    def observation_space(self, params: Optional[ReachingEnvParams]) -> spaces.Box:
        obs_len = 2 + ((self.n_players - 1) * 6)
        min_obs = -jnp.ones(obs_len)

        max_chunk = jnp.array([self.grid_size - 1, self.grid_size - 1, 2, 2, 2, 2])
        max_obs = [jnp.array([self.grid_size - 1, self.grid_size - 1])] + (
            [max_chunk] * (self.n_players - 1)
        )
        max_obs = jnp.concatenate(max_obs)

        return spaces.Box(min_obs, max_obs, shape=(obs_len,), dtype=jnp.float32)

    @property
    def name(self) -> str:
        """Environment name."""
        return "reaching"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return len(Action)

    @property
    def default_params(self) -> ReachingEnvParams:
        return ReachingEnvParams()

    def _init_render(self):
        from .rendering import Viewer

        self.viewer = Viewer((self.grid_size, self.grid_size))
        self.rendering_initialized = True

    def render(self, state: ReachingEnvState, mode="human"):
        if not self.rendering_initialized:
            self._init_render()
        if self.viewer is None:
            return
        return_rgb = mode == "rgb_array"
        return self.viewer.render(state, return_rgb_array=return_rgb)

    def close(self):
        if self.viewer:
            self.viewer.close()

    def centre_of_players(self, state: ReachingEnvState) -> chex.Array:
        return jnp.round(state.agent_locs.mean(axis=0)).astype(int)

    def go_to_goal(
        self, key: chex.PRNGKey, own_loc: chex.Array, goal_idx: chex.Array
    ) -> chex.Array:
        adjacent_tiles = own_loc + jnp.array([[-1, 0], [0, -1], [1, 0], [0, 1]])
        distances = jnp.abs(adjacent_tiles - self.goal_locs[goal_idx]).sum(axis=-1)

        # filter out tiles outside of the grid
        distances: chex.Array = jnp.where(
            jnp.all(adjacent_tiles >= 0, axis=1)
            & jnp.all(adjacent_tiles < self.grid_size, axis=1),
            distances,
            jnp.inf,
        )

        # now filter out tiles that are in goal_locs (except goal_locs[goal_idx])
        mask = (jnp.arange(5) == goal_idx).reshape(-1, 1)
        tmp = jnp.ones_like(self.goal_locs) * jnp.array([-1, -1])
        goal_locs = jnp.where(mask, tmp, self.goal_locs)
        distances: chex.Array = jnp.where(
            jnp.any(
                jnp.all(adjacent_tiles[:, None, :] == goal_locs[None, :, :], axis=-1),
                axis=1,
            ),
            jnp.inf,
            distances,
        )

        # now find which adjacent tile is closest to the goal and move to it
        closest_tile = adjacent_tiles[jnp.argmin(distances)]
        direction = jnp.sign(closest_tile - own_loc)
        vertical_action = (Action.NORTH.value * (direction[0] == -1)) + (
            Action.SOUTH.value * (direction[0] == 1)
        )
        horizontal_action = (Action.WEST.value * (direction[1] == -1)) + (
            Action.EAST.value * (direction[1] == 1)
        )

        # randomly choose between vertical and horizontal movement if both are valid
        choice = jax.random.randint(key, (), 0, 2)
        # choice = 0
        action = jnp.where(choice == 0, vertical_action, horizontal_action)

        # if move action is 0, set move action to the maximum of the two
        return jnp.where(
            action == 0, jnp.maximum(vertical_action, horizontal_action), action
        )

    def basic_heuristic_policy(
        self,
        key: chex.PRNGKey,
        state: ReachingEnvState,
        agent_idx: int,
    ) -> Tuple[chex.Array, chex.Array]:
        agent_loc, agent_type = (
            state.agent_locs[agent_idx],
            state.agent_types[agent_idx],
        )
        centre = self.centre_of_players(state)

        goal_distances = jnp.abs(self.goal_locs - centre).sum(axis=-1)
        goal_mask = jnp.concatenate([self.prefs_support[agent_type], jnp.array([0.0])])
        goal_distances = jnp.where(goal_mask, goal_distances, jnp.inf)

        goal_idx = jnp.argmin(goal_distances)
        move_action = self.go_to_goal(key, agent_loc, goal_idx)

        return (
            jnp.where(
                goal_distances[goal_idx] != jnp.inf, move_action, Action.NONE.value
            ),
            jnp.where(goal_distances[goal_idx] == jnp.inf, 5, goal_idx),
        )

    def heuristic_policy(
        self,
        key: chex.PRNGKey,
        state: ReachingEnvState,
        agent_idx: int,
    ) -> Tuple[chex.Array, chex.Array]:
        own_goals = self.prefs_support[state.agent_types[agent_idx]]
        teammate_goals = self.prefs_support[state.agent_types[1 - agent_idx]]

        feasible = jnp.concatenate([own_goals * teammate_goals, jnp.array([1.0])])
        values = feasible * self.goal_rewards

        centre = self.centre_of_players(state)
        goal_distances = jnp.abs(self.goal_locs - centre).sum(axis=-1)
        goal_distances = jnp.where(values == values.max(), goal_distances, jnp.inf)

        goal_idx = jnp.argmin(goal_distances)
        action = self.go_to_goal(key, state.agent_locs[agent_idx], goal_idx)

        is_in_goal = (
            jnp.all(state.agent_locs[agent_idx] == self.goal_locs, axis=-1)
            .astype(int)
            .sum()
        )

        action = jnp.where(is_in_goal, Action.NONE.value, action)
        goal_idx = jnp.where(is_in_goal, 5, goal_idx)

        return action, goal_idx

    def oracle_policy(
        self, key: chex.PRNGKey, state: ReachingEnvState, agent_index: int
    ):
        own_goals = self.prefs_support[state.agent_types[agent_index]]
        teammate_goals = jnp.sign(
            self.prefs_support[state.agent_types].sum(axis=0) - own_goals
        )
        feasible = jnp.concatenate([own_goals * teammate_goals, jnp.array([1.0])])
        values = feasible * self.goal_rewards

        centre = self.centre_of_players(state)
        goal_distances = jnp.abs(self.goal_locs - centre).sum(axis=-1)
        goal_distances = jnp.where(values == values.max(), goal_distances, jnp.inf)

        goal_idx = jnp.argmin(goal_distances)

        return self.go_to_goal(key, state.agent_locs[agent_index], goal_idx)


def make(
    grid_size=9,
    n_players=2,
    penalty=0.0,
) -> Tuple[ReachingEnv, ReachingEnvParams]:
    env = ReachingEnv(
        grid_size=grid_size,
        n_players=n_players,
        penalty=penalty,
    )
    return env, env.default_params

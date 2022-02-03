""" Implements a wrapper that lets the agent act multiple times in one time-step on links at one substation """
from copy import deepcopy
from typing import Dict, Any, Tuple, Union, Optional

import gym
import numpy as np
from gym import spaces
from maze.core.annotations import override
from maze.core.env.action_conversion import ActionType
from maze.core.env.maze_env import MazeEnv
from maze.core.env.observation_conversion import ObservationType
from maze.core.env.structured_env import StructuredEnv, ActorID, StepKeyType
from maze.core.env.structured_env_spaces_mixin import StructuredEnvSpacesMixin
from maze.core.wrappers.wrapper import ObservationWrapper
from maze.utils.bcolors import BColors

from maze_l2rpn.env.maze_env import Grid2OpEnvironment


class AlreadyActiveLinkException(Exception):
    """Exception to raise if an already active link has been selected"""
    pass


class MultiAgentSubChangesWrapper(ObservationWrapper[Grid2OpEnvironment], StructuredEnv, StructuredEnvSpacesMixin):
    """A wrapper that lets the agent act multiple times in one time-step on links at one substation.

    :param env: Environment to wrap.
    :param max_link_changes: The maximum of link changes to perform in one flat step.
    """

    def __init__(self,
                 env: MazeEnv,
                 max_link_changes: int):
        super().__init__(env)

        self._max_link_actions = max_link_changes
        assert self._max_link_actions > 1

        # Initialize class objects needed
        self._current_actions_to_perform = list()
        self._current_actions_to_perform_w_noop = list()
        self._last_obs = None
        self._sub_step_idx = 0
        self._current_sub_station_idx = -1
        self._current_link_to_set_mask = None
        self._noop_idx = self.action_conversion.n_links

        # Init reward to keep track of multistep reward
        self._last_reward = None

        # Update observation space
        self._action_spaces_dict = self.env.action_spaces_dict
        self._observation_spaces_dict = self.env.observation_spaces_dict
        step_key = [key for key, spaces_dict in self._action_spaces_dict.items()
                    if 'link_to_set' in spaces_dict.spaces][0]
        self._observation_spaces_dict[step_key] = spaces.Dict({
            **self.env.observation_spaces_dict[step_key].spaces,
            'already_selected_actions': spaces.Box(dtype=np.float32,
                                                   shape=(
                                                       self.env.observation_conversion.link_graph.n_links(),),
                                                   low=np.float32(0), high=np.float32(1)),
            'already_selected_noop': spaces.Box(dtype=np.float32, shape=(1,), low=np.float32(0),
                                                high=np.float32(1))
        })

    @override(MazeEnv)
    def seed(self, seed: int = None) -> None:
        """Sets the seed for this environment's random number generator(s).

        :param: seed: the seed integer initializing the random number generator.
        """
        self.env.seed(seed)

    @override(MazeEnv)
    def close(self) -> None:
        """Performs any necessary cleanup.
        """
        self.env.close()

    def _get_sub_id_where_action_is_happening(self, link_to_set_idx: int) -> int:
        """Retrieve the substation id corresponding to where the change is happening w.r.t. the given action.

        :param link_to_set_idx: The action, or the link to set.
        :return: The idx of the substation where the change is happening.
        """
        link_type, sub_id_or, bus_id_or, sub_id_ex, bus_id_ex, link_id = \
            self.action_conversion._link_action_idx_to_items[link_to_set_idx]
        maze_tmp = self.get_maze_state()
        if link_type == 'line':
            line_id = self.action_conversion._link_graph._sub_ids_to_line_map[(sub_id_or, sub_id_ex)][link_id]
            before_or_bus = maze_tmp.line_or_bus[line_id]
            before_ex_bus = maze_tmp.line_ex_bus[line_id]

            if before_or_bus != bus_id_or:
                sub_station_id = sub_id_or
            elif before_ex_bus != bus_id_ex:
                sub_station_id = sub_id_ex
            elif before_ex_bus == bus_id_ex and before_or_bus == bus_id_or:
                raise AlreadyActiveLinkException
            else:
                raise ValueError('Link type to change changes two substations at the same time or none at all!!!!\n'
                                 , link_type, sub_id_or, bus_id_or, sub_id_ex, bus_id_ex, link_id, line_id,
                                 before_ex_bus, before_or_bus)
        else:
            if link_type == 'gen' and bus_id_or == maze_tmp.topo_vect[maze_tmp.gen_pos_topo_vect[sub_id_or]]:
                raise AlreadyActiveLinkException
            elif link_type == 'load' and bus_id_or == maze_tmp.topo_vect[maze_tmp.load_pos_topo_vect[sub_id_or]]:
                raise AlreadyActiveLinkException
            sub_station_id = sub_id_ex

        return sub_station_id

    def _build_initial_link_to_set_mask(self):
        """Build the initial link to set mask after the current substation has been retrieved.
        As such the mask only allows actions where the change happens at the same substation."""
        assert self._current_link_to_set_mask is None
        assert self._current_sub_station_idx >= 0

        # Get all links connect to substation: selected_sub_station
        self._current_link_to_set_mask = \
            self.observation_conversion.link_graph.sub_station_to_adjacent_links_mask[self._current_sub_station_idx]

        # Add mask for explicit noop
        self._current_link_to_set_mask = np.concatenate((self._current_link_to_set_mask, np.array([1.0],
                                                                                                  dtype=np.float32)))

        # Mask out actions already taken
        self._current_link_to_set_mask[np.array(self._current_actions_to_perform)] = 0.0

        # Mask out currently active lines:
        self._current_link_to_set_mask = \
            self._last_obs['link_to_set_mask'] * self._current_link_to_set_mask

        for idx in np.where(self._current_link_to_set_mask)[0]:
            if idx == self._noop_idx:
                continue
            # If a given neighbouring line (from the substation where the change is happening) would change at a
            #   different one, that already selected mask it out.
            try:
                sub_where_changes_are_happening = self._get_sub_id_where_action_is_happening(idx)
                if sub_where_changes_are_happening != self._current_sub_station_idx:
                    self._current_link_to_set_mask[idx] = 0.0
            except AlreadyActiveLinkException:
                self._current_link_to_set_mask[idx] = 0.0

    @override(ObservationWrapper)
    def observation(self, observation: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Update the link masking if necessary.

        :param observation: The observation to be checked.
        :return: The critical state augment observation.
        """
        # first sub step
        if self._sub_step_idx == 0:
            observation['already_selected_actions'] = np.zeros(
                self.observation_space['already_selected_actions'].shape)
            observation['already_selected_noop'] = np.array([0]).astype(np.float32)
        # subsequent sub steps
        elif self._sub_step_idx > 0:
            if len(self._current_actions_to_perform) > 0:
                # update action mask
                self._current_link_to_set_mask[np.array(self._current_actions_to_perform)] = 0.0
                # update observation
                observation['already_selected_actions'][np.array(self._current_actions_to_perform)] = 1.0

            # update noop selected observation
            observation['already_selected_noop'] = \
                np.array([float(len(self._current_actions_to_perform) <
                                len(self._current_actions_to_perform_w_noop))]).astype(np.float32)
            # update action maks
            observation['link_to_set_mask'] = observation['link_to_set_mask'] * self._current_link_to_set_mask
        self._last_obs = deepcopy(observation)

        if 'link_to_set_mask' in observation:
            observation['link_to_set_mask'] = observation['link_to_set_mask'].astype(np.float32)
        if 'already_selected_actions' in observation:
            observation['already_selected_actions'] = observation['already_selected_actions'].astype(np.float32)

        return observation

    @staticmethod
    def _get_action_idx(action: ActionType) -> int:
        """Get the action idx for a given action.

        :param action: The action returned from the agent.
        :return: The action idx.
        """
        assert 'link_to_set' in action
        link_id = action['link_to_set']
        if isinstance(link_id, np.ndarray):
            assert link_id.shape == () or link_id.shape == (1,)
            if link_id.shape == ():
                link_id = int(link_id)
            else:
                link_id = link_id[0]
        return link_id

    def rollout_step(self, action: ActionType) -> Tuple[Union[ActionType, ObservationType], bool]:
        """Perform a rollout step of the wrapper, that is step through the wrapper logic for the current substep/agent.
        If that was the last sub step return the collected actions, otherwise the reprocessed observation.
        This is necessary for the rollout script.

        :param action: The action returned by the policy for the current substep/agent.
        :return: Ether an action or an observation, depending on whether the substep is done or not, respectively,
            which is second return value.
        """

        # assert that no masked out action has been selected
        link_to_set_idx = self._get_action_idx(action)
        if self._current_link_to_set_mask is not None:
            is_masked_out_action_selected = not self._current_link_to_set_mask[link_to_set_idx]
        else:
            is_masked_out_action_selected = not self._last_obs['link_to_set_mask'][link_to_set_idx]

        if is_masked_out_action_selected:
            # BColors.print_colored(f'A Masked out action has been selected: {link_to_set_idx}\n', BColors.WARNING)
            raise Exception(f'A Masked out action has been selected: {link_to_set_idx}\n', BColors.WARNING)

        self._current_actions_to_perform_w_noop.append(link_to_set_idx)

        # env with fixed number of sub steps
        if self._sub_step_idx == 0:

            # link change action
            if link_to_set_idx != self._noop_idx:
                # Start internal step logic of this wrapper
                self._current_actions_to_perform.append(link_to_set_idx)
                self._current_sub_station_idx = self._get_sub_id_where_action_is_happening(link_to_set_idx)
                self._build_initial_link_to_set_mask()
            # noop
            else:
                self._current_link_to_set_mask = np.ones(self._last_obs['link_to_set_mask'].shape)

            obs = self._last_obs
            self._sub_step_idx += 1
            return self.observation(obs), False

        # actual link change action selected (not noop)
        if link_to_set_idx != self._noop_idx:
            # assert that action has not been already selected
            assert link_to_set_idx not in self._current_actions_to_perform, \
                f'{link_to_set_idx}, {self._current_actions_to_perform}'
            self._current_actions_to_perform.append(link_to_set_idx)

            # first substep
            if len(self._current_actions_to_perform) == 1:
                self._current_sub_station_idx = self._get_sub_id_where_action_is_happening(link_to_set_idx)
                self._current_link_to_set_mask = None
                self._build_initial_link_to_set_mask()

            # update mask
            self._current_link_to_set_mask[np.array(self._current_actions_to_perform)] = 0.0

        # Only perform env step once the max number of link action has been reached... ensures that recorded
        # observation and action always have the same count. (Practically they are filled with noops)
        if self._sub_step_idx >= self._max_link_actions - 1:

            # Override action with all collected action (excluding the noop at the end)
            flat_action = action.copy()
            flat_action['link_to_set'] = np.array(self._current_actions_to_perform_w_noop)

            # Reset values
            self._reset_values()
            return flat_action, True

        else:
            obs = self._last_obs
            self._sub_step_idx += 1

            return self.observation(obs), False

    @override(MazeEnv)
    def step(self, action: ActionType) -> Tuple[ObservationType, float, bool, Dict[Any, Any]]:
        """Perform a single step of the wrapper, in case the substep is not done, return dummy variables, otherwise
            perform a flat step of the env with the collected actions.

        :param action: The action return by agent for the given substep
        :return: The observation, reward, done and info for the current substep.
        """
        observation_or_action, step_done = self.rollout_step(action, is_rollout=False)

        # flat env step
        if step_done:
            obs, reward, done, info = self.env.step(observation_or_action)
            self._last_reward = reward
            obs = self.observation(obs)
        # sub step
        else:
            obs, reward, done, info = observation_or_action, 0.0, False, {}

        return obs, reward, done, info

    def _reset_values(self) -> None:
        """Reset all internal values"""

        self._current_sub_station_idx = -1
        self._current_link_to_set_mask = None
        self._current_actions_to_perform = list()
        self._current_actions_to_perform_w_noop = list()
        self._last_obs = None

        self._sub_step_idx = 0

    def reset(self) -> ObservationType:
        """Randomly resets the environment either to the beginning at step 0 or close to the step where a noop action
        sequence would cause a blackout.

        :return: The initial observation.
        """
        obs = self.env.reset()
        self._reset_values()

        return self.observation(obs)

    @property
    @override(StructuredEnv)
    def agent_counts_dict(self) -> Dict[StepKeyType, int]:
        """Single agent for selection and single agent for cutting sub-step."""
        return {0: self._max_link_actions}

    @override(StructuredEnv)
    def actor_id(self) -> ActorID:
        """Implementation of :class:`~maze.core.env.structured_env_spaces_mixin.StructuredEnvSpacesMixin` interface.
        """
        return ActorID(0, self._sub_step_idx)

    @override(StructuredEnv)
    def is_actor_done(self) -> bool:
        """Implementation of :class:`~maze.core.env.structured_env_spaces_mixin.StructuredEnvSpacesMixin` interface.
        """
        return False

    @override(StructuredEnv)
    def get_actor_rewards(self) -> Optional[np.ndarray]:
        """Return half of the last reward for each of the substeps."""
        return np.array([self._last_reward / float(self._max_link_actions)] * self._max_link_actions)

    @override(ObservationWrapper)
    def clone_from(self, env: 'MultiAgentSubChangesWrapper') -> None:
        """Reset this gym environment to the given state by creating a deep copy of the `env.state` instance variable"""
        self._max_link_actions = env._max_link_actions
        self._sub_step_idx = env._sub_step_idx
        self._current_sub_station_idx = env._current_sub_station_idx

        self._noop_idx = env._noop_idx
        self._observation_spaces_dict = env._observation_spaces_dict

        self.env.clone_from(env)

        self._current_link_to_set_mask = None if env._current_link_to_set_mask is None else \
            env._current_link_to_set_mask.copy()
        self._current_actions_to_perform_w_noop = None if env._current_actions_to_perform_w_noop is None else \
            env._current_actions_to_perform_w_noop.copy()
        self._last_obs = None if env._last_obs is None else deepcopy(env._last_obs)
        self._current_actions_to_perform = None if env._current_actions_to_perform is None else \
            env._current_actions_to_perform.copy()

    @property
    @override(StructuredEnvSpacesMixin)
    def observation_space(self) -> gym.spaces.Dict:
        """Implementation of :class:`~maze.core.env.structured_env_spaces_mixin.StructuredEnvSpacesMixin` interface.
        """
        return self._observation_spaces_dict[self.actor_id()[0]]

    @property
    @override(StructuredEnvSpacesMixin)
    def observation_spaces_dict(self) -> Dict[Union[int, str], gym.spaces.Dict]:
        """Implementation of :class:`~maze.core.env.structured_env_spaces_mixin.StructuredEnvSpacesMixin` interface.
        """
        return self._observation_spaces_dict

    @property
    @override(StructuredEnvSpacesMixin)
    def action_space(self) -> gym.spaces.Dict:
        """Implementation of :class:`~maze.core.env.structured_env_spaces_mixin.StructuredEnvSpacesMixin` interface.
        """
        return self.env.action_space

    @property
    @override(StructuredEnvSpacesMixin)
    def action_spaces_dict(self) -> Dict[Union[int, str], gym.spaces.Dict]:
        """Implementation of :class:`~maze.core.env.structured_env_spaces_mixin.StructuredEnvSpacesMixin` interface.
        """
        return self._action_spaces_dict

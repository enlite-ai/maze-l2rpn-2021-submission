""" Implements a critical state observer. """
import numpy as np
from grid2op.Observation import CompleteObservation
from gym import spaces
from typing import Dict, Optional, Tuple, Any

from maze.core.env.action_conversion import ActionType
from maze.core.env.observation_conversion import ObservationType
from maze.utils.bcolors import BColors

from maze_l2rpn.env.maze_env import Grid2OpEnvironment
from maze.core.annotations import override
from maze.core.env.maze_env import MazeEnv
from maze.core.wrappers.wrapper import ObservationWrapper


class CriticalStateObserverSimulateWrapper(ObservationWrapper[Grid2OpEnvironment]):
    """An wrapper updating the environment observation with a critical state flag ('critical_state' in {0, 1}).

    :param env: Environment/wrapper to wrap.
    :param max_rho: The maximum line capacity be for considered a critical state.
    :param max_rho_simulate: The maximum line capacity in the next simulated observation (with noop) if value < -np.inf.
    """

    def __init__(self,
                 env: MazeEnv,
                 max_rho: Optional[np.float32],
                 max_rho_simulate: float):
        super().__init__(env)
        self._max_rho = max_rho
        self._max_rho_simulate = max_rho_simulate

        # extend observation space
        for sub_space_key in self.observation_spaces_dict.keys():
            self.observation_spaces_dict[sub_space_key].spaces["critical_state"] = \
                spaces.Box(low=np.float32(0), high=np.float32(1), shape=(1,), dtype=np.float32)

        # Cache the calculation
        self._current_state_is_critical = None

    def _update_link_mask(self, obs: ObservationType) -> None:
        """ Updates the link prediction mask.

        :param obs: The observation to be updated.
        """
        link_mask = self.observation_conversion.link_graph.link_mask(
                state=self.get_maze_state(), with_link_masking=True)
        obs["link_to_set_mask"][:len(link_mask)] = obs["link_to_set_mask"][:len(link_mask)] * link_mask

    @override(ObservationWrapper)
    def observation(self, observation: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Check for critical states.

        :param observation: The observation to be checked.
        :return: The critical state augment observation.
        """

        # update observation with critical state flag
        observation["critical_state"] = np.asarray(self.is_in_critical_state(), dtype=np.float32)

        return observation

    def is_in_critical_state(self) -> bool:
        """Assess if the current state of the underlying env is critical."""

        if self._current_state_is_critical is not None:
            print('using cashed critical state... ')
            return self._current_state_is_critical

        # get grid2op state object
        state: CompleteObservation = self.get_maze_state()

        # critical state tests
        critical_state = False

        # line capacity check
        if self._max_rho:
            if np.any(np.isnan(state.rho)) or np.any(state.rho > self._max_rho):
                critical_state = True

        # simulation check (only needs to be performed if no critical_state has been observed yet)
        if self._max_rho_simulate < np.inf and not critical_state and (np.any(np.isnan(state.rho))
                                                                        or np.any(state.rho > 0.8)):
            noop_action = self.action_conversion.noop_action()
            playable_action = self.action_conversion.space_to_maze(noop_action, state)
            sim_obs, sim_reward, sim_done, sim_info = state.simulate(playable_action)
            if np.any(np.isnan(sim_obs.rho)) or np.any(sim_obs.rho > self._max_rho_simulate):
                critical_state = True

        self._current_state_is_critical = critical_state

        return critical_state

    def reset(self) -> Any:
        """Invalidate critical state cache"""
        self._current_state_is_critical = None
        observation = self.env.reset()
        return self.observation(observation)

    def step(self, action: ActionType) -> Tuple[Any, Any, bool, Dict[Any, Any]]:
        """Perform a step with the given action.

        :param action: The action from the agent.
        :return: The observation, reward, done and info for the current step.
        """
        self._current_state_is_critical = None
        observation, reward, done, info = self.env.step(action)
        return self.observation(observation), reward, done, info

    def set_state(self, state: CompleteObservation):
        """Invalidate critical state cache"""
        self._current_state_is_critical = None
        BColors.print_colored('State is set in critical observation wrapper', BColors.WARNING)
        self.env.set_state(state)

    def set_max_rho(self, max_rho: float) -> None:
        """Set the maximum rho.

        :param max_rho: The value to set the maximum allowed rho to.
        """
        self._max_rho = max_rho

    def set_max_rho_simulate(self, max_rho_simulate: float) -> None:
        """Set the maximum simulated rho.

        :param max_rho_simulate: The value to set the maximum allowed rho of the simulation to.
        """
        self._max_rho_simulate = max_rho_simulate

    def clone_from(self, env: 'CriticalStateObserverWrapper') -> None:
        """implementation of :class:`~maze.core.env.simulated_env_mixin.SimulatedEnvMixin`."""
        self._current_state_is_critical = env._current_state_is_critical
        self._max_rho = env._max_rho
        self._max_rho_simulate = env._max_rho_simulate
        self.env.clone_from(env)

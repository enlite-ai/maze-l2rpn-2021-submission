"""Contains a state to observation interface converting the state into a vectorized feature space."""
from abc import ABC
from typing import Dict

import grid2op
import numpy as np
from grid2op.Observation import CompleteObservation
from gym import spaces
from maze.core.annotations import override
from maze.core.env.observation_conversion import ObservationConversionInterface

from maze_l2rpn.env.grid_networkx_graph import GridNetworkxGraph


class ObservationConversion(ObservationConversionInterface, ABC):
    """Object representing an observation.
    For more information consider: https://grid2op.readthedocs.io/en/latest/space.html

    :param grid2op_env: The grid2op environment.
    """

    def __init__(self,
                 grid2op_env: grid2op.Environment.Environment):
        super().__init__()

        self.observation_space = grid2op_env.observation_space
        self._thermal_limit = grid2op_env.get_thermal_limit()

        self.n_gen = self.observation_space.n_gen
        self.n_load = self.observation_space.n_load
        self.n_sub = self.observation_space.n_sub
        self.n_line = self.observation_space.n_line
        self.max_buses = 2
        self.max_links = np.amax(self.observation_space.sub_info)

        self.load_to_subid = self.observation_space.load_to_subid
        self.gen_to_subid = self.observation_space.gen_to_subid
        self.line_or_to_subid = self.observation_space.line_or_to_subid
        self.line_ex_to_subid = self.observation_space.line_ex_to_subid

        # initialize topology graph
        self.link_graph = GridNetworkxGraph(space=self.observation_space)

        self._link_mask_shape = self.link_graph.n_links()
        self._link_mask_shape += 1

        self._n_features = 4 * self.n_line + self.n_sub

    @override(ObservationConversionInterface)
    def maze_to_space(self, state: CompleteObservation) -> Dict[str, np.ndarray]:
        """Converts core environment state to space observation.
        For more information consider: https://grid2op.readthedocs.io/en/latest/observation.html#objectives

        :param state: The state returned by the powergrid env step.
        :return: The resulting dictionary observation.
        """

        # compile link mask and current adjacency
        link_mask = self.link_graph.link_mask(state)

        link_mask = np.concatenate((link_mask, np.ones(1, dtype=np.float32)))

        # current flow in powerline (n_line)
        current_flow = state.rho * self._thermal_limit

        # cumulative consumed power within substation
        sub_loads = np.zeros((state.n_sub,), dtype=np.float32)
        for load_id in range(self.observation_space.n_load):
            sub_id = self.observation_space.load_to_subid[load_id]
            sub_loads[sub_id] += state.load_p[load_id]

        # cumulative generated power within substation
        sub_gens = np.zeros((state.n_sub,), dtype=np.float32)
        for gen_id in range(self.observation_space.n_gen):
            sub_id = self.observation_space.gen_to_subid[gen_id]
            sub_gens[sub_id] += state.gen_p[gen_id]

        # power delta within substation (n_sub)
        sub_power_deltas = sub_gens - sub_loads

        # power delta between substations (n_line)
        line_deltas = np.zeros((self.n_line,), dtype=np.float32)
        for line_id in range(self.n_line):
            sub_or_id = state.line_or_to_subid[line_id]
            sub_ex_id = state.line_ex_to_subid[line_id]
            line_deltas[line_id] = sub_power_deltas[sub_or_id] - sub_power_deltas[sub_ex_id]

        features = np.concatenate([state.line_status, state.rho, current_flow, line_deltas,
                                   sub_power_deltas])

        return {
            "features": features,
            "topology": state.topo_vect,
            "link_to_set_mask": link_mask
        }

    @override(ObservationConversionInterface)
    def space_to_maze(self, observation: dict) -> CompleteObservation:
        """Converts space observation to core environment state.
        (This is most like not possible for most observation space_to_maze)
        """
        raise NotImplementedError

    @override(ObservationConversionInterface)
    def space(self) -> spaces.Dict:
        """Return the observation space shape based on the given params.

        :return: Gym space object.
        """
        float_max = np.finfo(np.float32).max
        float_min = np.finfo(np.float32).min

        return spaces.Dict({
            "features": spaces.Box(dtype=np.float32, shape=(self._n_features,),
                                   low=float_min, high=float_max),
            "topology": spaces.Box(dtype=np.float32, shape=(self.observation_space.dim_topo,),
                                   low=-1, high=2),
            "link_to_set_mask": spaces.Box(dtype=np.float32, shape=(self._link_mask_shape,),
                                           low=np.float32(0), high=np.float32(1)),
        })

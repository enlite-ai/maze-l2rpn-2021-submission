"""Maze Link Prediction ActionConversion interface."""
from abc import ABC
from typing import Dict, Union, List, Optional, Any

import grid2op
import numpy as np
from grid2op.Action import TopologyAndDispatchAction, PlayableAction
from grid2op.Observation.CompleteObservation import CompleteObservation
from gym import spaces
from maze.core.annotations import override
from maze.core.env.action_conversion import ActionConversionInterface

from maze_l2rpn.env.grid_networkx_graph import LinkInfoType, GridNetworkxGraph

ActionType = Dict[str, Union[int, List[float]]]


class ActionConversion(ActionConversionInterface, ABC):
    """Interface specifying the conversion of space to actual environment actions.

    :param grid2op_env: The grid2op environment.
    """

    def __init__(self, grid2op_env: grid2op.Environment):
        super().__init__()
        self.action_space = grid2op_env.action_space

        self.n_gen = self.action_space.n_gen
        self.n_load = self.action_space.n_load
        self.n_sub = self.action_space.n_sub
        self.n_line = self.action_space.n_line
        self.max_buses = 2
        self.max_links = np.amax(self.action_space.sub_info)

        # initialize topology graph
        self._link_graph = GridNetworkxGraph(space=self.action_space)

        # derive link change actions
        self.n_links = self._link_graph.n_links()
        full_link_list = self._link_graph.full_link_list()
        self._link_action_idx_to_items: Dict[int, LinkInfoType] = dict(zip(range(self.n_links), full_link_list))

    @override(ActionConversionInterface)
    def space_to_maze(self, action: ActionType, state: Optional[CompleteObservation]) -> TopologyAndDispatchAction:
        """Converts space to environment action.
        :param action: the dictionary action.
        :param state: the environment state.
        :return: action object.
        """

        # Bypass checks if action is already a PlayableAction
        if isinstance(action, PlayableAction):
            return action

        # Bypass more complex checks if action is already a L2RPN action dict.
        if all([key in {"change_bus", "change_line_status"} for key in action]) and len(action) == 2:
            return self._to_playable(action)

        # extract topology info
        topo_vect = state.topo_vect
        or_topo = state.line_or_pos_topo_vect
        ex_topo = state.line_ex_pos_topo_vect

        # init action
        l2rpn_action_dict = dict()

        # extract action
        if "link_to_set" not in action:
            # translate to noop action
            link_idx = self.n_links
        else:
            link_idx = action["link_to_set"]

        #  link_idx == self.n_links is the noop
        link_idx_list = np.array([link_idx]) if isinstance(link_idx, int) or link_idx.shape == () else link_idx
        for link_idx in link_idx_list:
            if self.n_links > link_idx >= 0:

                if isinstance(link_idx, np.ndarray) and link_idx.shape == ():
                    link_idx = int(link_idx)
                link_type, sub_id_or, bus_id_or, sub_id_ex, bus_id_ex, link_id = \
                    self._link_action_idx_to_items[link_idx]

                # change link to generator
                if link_type == "gen":
                    gen_idx = sub_id_or
                    bus_idx = bus_id_or

                    prev_bus = topo_vect[state.gen_pos_topo_vect[gen_idx]]
                    if bus_idx != prev_bus:
                        if 'set_bus' not in l2rpn_action_dict:
                            l2rpn_action_dict["set_bus"] = dict()
                        if 'generators_id' not in l2rpn_action_dict["set_bus"]:
                            l2rpn_action_dict["set_bus"]["generators_id"] = list()
                        l2rpn_action_dict["set_bus"]["generators_id"].append((gen_idx, bus_idx))

                # change link to load
                elif link_type == "load":
                    load_idx = sub_id_or
                    bus_idx = bus_id_or

                    prev_bus = topo_vect[state.load_pos_topo_vect[load_idx]]
                    if bus_idx != prev_bus:
                        if 'set_bus' not in l2rpn_action_dict:
                            l2rpn_action_dict["set_bus"] = dict()
                        if 'loads_id' not in l2rpn_action_dict["set_bus"]:
                            l2rpn_action_dict["set_bus"]["loads_id"] = list()
                        l2rpn_action_dict["set_bus"]["loads_id"].append((load_idx, bus_idx))

                # change link of powerline
                elif link_type == "line":

                    # request line id
                    line_idx = self._link_graph.sub_ids_to_line(sub_id_or, sub_id_ex, link_id)

                    # get current buses for respective line
                    lor_bus = topo_vect[or_topo[line_idx]]
                    lex_bus = topo_vect[ex_topo[line_idx]]

                    # change origin bus
                    if lor_bus != bus_id_or:
                        assert lex_bus == bus_id_ex, \
                            f"{lex_bus}, {lor_bus}, {bus_id_ex}, {bus_id_or}, {line_idx}, {link_idx}"
                        if 'set_bus' not in l2rpn_action_dict:
                            l2rpn_action_dict["set_bus"] = dict()
                        if 'lines_or_id' not in l2rpn_action_dict["set_bus"]:
                            l2rpn_action_dict["set_bus"]["lines_or_id"] = list()
                        l2rpn_action_dict["set_bus"]["lines_or_id"].append((line_idx, bus_id_or))
                    # change extremity bus
                    elif lex_bus != bus_id_ex:
                        assert lor_bus == bus_id_or, f"{lex_bus}, {lor_bus}, {bus_id_ex}, {bus_id_or}"
                        if 'set_bus' not in l2rpn_action_dict:
                            l2rpn_action_dict["set_bus"] = dict()
                        if 'lines_ex_id' not in l2rpn_action_dict["set_bus"]:
                            l2rpn_action_dict["set_bus"]["lines_ex_id"] = list()
                        l2rpn_action_dict["set_bus"]["lines_ex_id"].append((line_idx, bus_id_ex))

                else:
                    raise ValueError("Not Reachable!")

        # automatically turn power lines on again if in cool down
        for line_idx in np.nonzero(state.line_status == 0)[0]:
            if state.time_before_cooldown_line[line_idx] == 0:
                l2rpn_action_dict["set_line_status"] = [(line_idx, 1)]
                break

        if 'raise_alarm' in action:
            l2rpn_action_dict['raise_alarm'] = action['raise_alarm']

        # Convert selection to playable action required by grid2op
        return self._to_playable(l2rpn_action_dict)

    @override(ActionConversionInterface)
    def maze_to_space(self, action: TopologyAndDispatchAction) -> Dict[str, int]:
        """Converts environment to agent action.

        :param: action: the environment action to convert.
        :return: the dictionary action.
        """
        raise NotImplementedError

    @override(ActionConversionInterface)
    def space(self) -> spaces.Dict:
        """Returns respective gym action space.
        :return: Gym action space.
        """
        action_spaces_dict = dict()

        # add link prediction action space
        n_actions = self.n_links + 1
        action_spaces_dict["link_to_set"] = spaces.Discrete(n_actions)

        return spaces.Dict(action_spaces_dict)

    def _to_playable(self, l2rpn_action_dict: Dict[str, Any]) -> TopologyAndDispatchAction:
        """Compiles a playable action from an authorized_keys action dictionary.

        :param l2rpn_action_dict: The l2rpn action dictionary.
        :return: A l2rpn TopologyAndDispatchAction.
        """
        action = self.action_space(l2rpn_action_dict)
        assert isinstance(action, PlayableAction)
        return action

    @override(ActionConversionInterface)
    def noop_action(self) -> Dict[str, int]:
        """Return the noop action"""
        action = dict()

        action["link_to_set"] = self.n_links

        return action

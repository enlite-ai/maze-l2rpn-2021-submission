""" Contains a networkx graph representing the underlying powergrid. """
from collections import defaultdict
from typing import Dict, Tuple, Optional, List, Union, Iterable, Set

import networkx as nx
import numpy as np
from grid2op.Action import ActionSpace
from grid2op.Observation import ObservationSpace, CompleteObservation

LinkInfoType = Tuple[str, int, int, Optional[int], Optional[int], int]


class GridNetworkxGraph:
    """Represents the grid as a networkx graph.

    :param space: Either a grid2op action or observation space object.
    """

    def __init__(self, space: Union[ObservationSpace, ActionSpace]):

        # extract relevant info from observation space
        self.n_sub = space.n_sub
        self.n_line = space.n_line
        self.n_gen = space.n_gen
        self.n_load = space.n_load

        self.load_to_subid = space.load_to_subid
        self.gen_to_subid = space.gen_to_subid
        self.or_sub = space.line_or_to_subid
        self.ex_sub = space.line_ex_to_subid
        self.or_topo = space.line_or_pos_topo_vect
        self.ex_topo = space.line_ex_pos_topo_vect

        self.gen_pos_topo_vect = space.gen_pos_topo_vect
        self.load_pos_topo_vect = space.load_pos_topo_vect

        # mapping of substation ids to powerlines
        self._sub_ids_to_line_map = defaultdict(list)
        for i_line in range(self.n_line):
            sub_or = space.line_or_to_subid[i_line]
            sub_ex = space.line_ex_to_subid[i_line]
            self._sub_ids_to_line_map[(sub_or, sub_ex)].append(i_line)

        # compile graphs
        self.node_graph = nx.MultiGraph()
        self.line_graph = None
        self._node_idx_to_label: Dict[int, str] = dict()
        self._node_labels_to_idx: Dict[str, int] = dict()
        self._edge_to_link_idx: Dict[Tuple[int, int, int], int] = dict()
        self._link_list = None
        self._link_to_index = defaultdict(list)
        self.substation_id_to_adjacent_links_idx = None
        self.build_graph()

    def build_graph(self) -> None:
        """Compiles the networkx grid2op topology graph."""

        self._add_load_nodes()
        self._add_generator_nodes()
        self._add_substation_nodes()

        # invert node dictionary
        self._node_labels_to_idx = dict([(v, k) for k, v in self._node_idx_to_label.items()])

        self._add_load_links()
        self._add_generator_links()
        self._add_powerline_links()

        # compute edge to link indices
        self._edge_to_link_idx = dict()
        for i, (v, w, j) in enumerate(self.node_graph.edges):
            self._edge_to_link_idx[(v, w, j)] = i

        # convert node to line graph
        self._compile_line_graph()

        self._init_link_list()
        self._init_substation_id_to_adjacent_links_idx()

    def _init_substation_id_to_adjacent_links_idx(self) -> None:
        """Initialize the substation id to adjacent links array."""
        self.sub_station_to_adjacent_links_mask = np.zeros((self.n_sub, len(self._link_list)))
        for sub_station_id in range(self.n_sub):
            node_id_0 = self._sub_node_idx(sub_station_id, 1)
            node_id_1 = self._sub_node_idx(sub_station_id, 2)

            link_connected_to_sub = list()
            for idx, (u, v, k) in enumerate(self.node_graph.edges):
                if node_id_0 in (u, v) or node_id_1 in (u, v):
                    assert idx == self._edge_to_link_idx[(u, v, k)]
                    link_connected_to_sub.append(idx)
            self.sub_station_to_adjacent_links_mask[sub_station_id][np.array(link_connected_to_sub)] = 1

    def _compile_line_graph(self) -> None:
        """Compile the line graph from the node graph"""
        self.line_graph = nx.line_graph(self.node_graph)

    def _add_substation_nodes(self) -> None:
        """Add substation nodes to the graph."""
        # add substations - bus 1
        for sub_idx in range(self.n_sub):
            node_idx = self._sub_node_idx(sub_idx, bus=1)
            label = "S{}-{}".format(sub_idx, 1)
            self._node_idx_to_label[node_idx] = label
            self.node_graph.add_node(node_idx, label=label, node_type="sub", bus=1)

        # add substations - bus 2
        for sub_idx in range(self.n_sub):
            node_idx = self._sub_node_idx(sub_idx, bus=2)
            label = "S{}-{}".format(sub_idx, 2)
            self._node_idx_to_label[node_idx] = label
            self.node_graph.add_node(node_idx, label=label, node_type="sub", bus=2)

    def _sub_node_idx(self, sub_idx: int, bus: int) -> int:
        """Computes a substation node index.
        :param sub_idx: The substation index.
        :param bus: The bus id.
        :return: The index of the node in the link graph.
        """
        return self.n_load + self.n_gen + sub_idx + (bus - 1) * self.n_sub

    def _add_load_nodes(self) -> None:
        """Adds load nodes to the graph."""
        for load_idx in range(self.n_load):
            node_idx = self._load_node_idx(load_idx)
            label = "L{}".format(load_idx)
            self._node_idx_to_label[node_idx] = label
            self.node_graph.add_node(node_idx, label=label, node_type="load")

    @classmethod
    def _load_node_idx(cls, load_idx: int) -> int:
        """Computes a load node index.
        :param load_idx: The load index.
        :return: The index of the node in the link graph.
        """
        return load_idx

    def _add_generator_nodes(self) -> None:
        """Adds generator nodes to the graph."""
        for gen_idx in range(self.n_gen):
            node_idx = self._generator_node_idx(gen_idx)
            label = "G{}".format(gen_idx)
            self._node_idx_to_label[node_idx] = label
            self.node_graph.add_node(node_idx, label=label, node_type="gen")

    def _generator_node_idx(self, gen_idx: int) -> int:
        """Computes a generator node index.
        :param gen_idx: The generator index.
        :return: The index of the node in the link graph.
        """
        return gen_idx + self.n_load

    def _add_load_links(self) -> None:
        """Adds substation load links to the graph."""
        for load_idx in range(self.n_load):
            node_idx = self._load_node_idx(load_idx)
            sub_id = self.load_to_subid[load_idx]
            for bus in [1, 2]:
                target = self._sub_node_idx(sub_id, bus)
                self.node_graph.add_edge(node_idx, target, key=0, link_type="load",
                                         link_info=("load", load_idx, bus, sub_id, None, 0))

    def _add_generator_links(self) -> None:
        """Adds generator load links to the graph."""
        for gen_idx in range(self.n_gen):
            node_idx = self._generator_node_idx(gen_idx)
            sub_id = self.gen_to_subid[gen_idx]
            for bus in [1, 2]:
                target = self._sub_node_idx(sub_id, bus)
                self.node_graph.add_edge(node_idx, target, key=0, link_type="gen",
                                         link_info=("gen", gen_idx, bus, sub_id, None, 0))

    def _add_powerline_links(self) -> None:
        """Adds powerline links to the graph."""

        # Set lines edges
        for line_idx in range(self.n_line):

            # Get substation index for current line
            lor_sub = self.or_sub[line_idx]
            lex_sub = self.ex_sub[line_idx]

            # add bus combinations
            for or_bus, ex_bus in [(1, 1), (2, 2), (1, 2), (2, 1)]:
                # Compute edge vertices indices for current graph
                left_v = self._node_labels_to_idx["S{}-{}".format(lor_sub, or_bus)]
                right_v = self._node_labels_to_idx["S{}-{}".format(lex_sub, ex_bus)]
                edge = (left_v, right_v)

                # Register edge in graph
                link_id = self._sub_ids_to_line_map[(lor_sub, lex_sub)].index(line_idx)
                self.node_graph.add_edge(edge[0], edge[1], key=link_id, link_type="line",
                                         link_info=("line", lor_sub, or_bus, lex_sub, ex_bus, link_id))

    def _init_link_list(self) -> None:
        """Initialized the list of all links in the graph."""
        self._link_list = []
        link_infos = nx.get_edge_attributes(self.node_graph, "link_info")
        for link_idx, edge in enumerate(self.node_graph.edges):
            info = link_infos[edge]
            self._link_list.append(info)

            # write an auxiliary index
            if info[0] == "line":
                self._link_to_index[("line", info[1], info[3], info[5])].append(link_idx)
            elif info[0] == "gen" or info[0] == "load":
                self._link_to_index[(info[0], info[1])].append(link_idx)

    def sub_ids_to_line(self, or_sub_id: int, ex_sub_id: int, link_id: int) -> int:
        """Computes a mapping of two substation ids to a powerline.
        :param or_sub_id: The originating substation.
        :param ex_sub_id: The extremity substation.
        :param link_id: The link id for modeling multiple power lines between two substations
        :return: The powerline id.
        """
        return self._sub_ids_to_line_map[(or_sub_id, ex_sub_id)][link_id]

    def full_link_list(self) -> List[LinkInfoType]:
        """Computes a list of all links in the graph.

        structure: link_type, sub_id_or, bus_id_or, sub_id_ex, bus_id_ex
            - lines: "line", sub_id_or, bus_id_or, sub_id_ex, bus_id_ex
            - loads: "load", load_id, bus_id, None, None
            - gen: "gen", gen_id, bus_id, None, None

        :return: The full link list.
        """
        return self._link_list

    def link_mask(self, state: CompleteObservation) -> np.ndarray:
        """Computes a link mask, the active node adjacency matrix, a matrix of link features
        and the graph of currently active edges.
        :param state: The current state of the environment.
        :return: Tuple of (link_mask, active_adjacency, link_features, active_graph)
        """
        link_mask = np.ones(self.n_links(), dtype=np.float32)

        # mask out invalid line link changes
        for i_link, link in enumerate(self._link_list):

            # extract link and edge data
            link_type, sub_id_or, bus_id_or, sub_id_ex, bus_id_ex, link_id = link

            if link_type == "load":
                load_id = sub_id_or
                bus = state.topo_vect[self.load_pos_topo_vect[load_id]]

                if bus_id_or == bus:
                    link_mask[i_link] = 0

            elif link_type == "gen":
                gen_id = sub_id_or
                bus = state.topo_vect[self.gen_pos_topo_vect[gen_id]]

                if bus_id_or == bus:
                    link_mask[i_link] = 0

            elif link_type == "line":
                line_idx = self.sub_ids_to_line(sub_id_or, sub_id_ex, link_id)

                # -> mask out double link changes

                # get current buses for respective line
                lor_bus = state.topo_vect[self.or_topo[line_idx]]
                lex_bus = state.topo_vect[self.ex_topo[line_idx]]

                if (lor_bus != bus_id_or) and (lex_bus != bus_id_ex):
                    # Mask out links that would require changes on two substations
                    link_mask[i_link] = 0

                elif (lor_bus == bus_id_or) and (lex_bus == bus_id_ex):
                    link_mask[i_link] = 0

                # -> mask out inactive power line links
                if bool(state.line_status[line_idx]) is False:
                    link_mask[i_link] = 0

        for i_link in np.nonzero(link_mask)[0]:
            # extract link info
            link_type, sub_id_or, bus_id_or, sub_id_ex, bus_id_ex, link_id = self._link_list[i_link]

            if link_type in ["load", "gen"]:
                # substations need a cooldown before next bus switch
                bus, sub_id = bus_id_or, sub_id_ex
                if sub_id is not None and state.time_before_cooldown_sub[sub_id]:
                    link_mask[i_link] = False
                    continue

            if link_type == "line":

                # substations need a cooldown before next bus switch
                if (sub_id_or is not None and state.time_before_cooldown_sub[sub_id_or] or
                        sub_id_ex is not None and state.time_before_cooldown_sub[sub_id_ex]):
                    link_mask[i_link] = False
                    continue

                # mask lines in cooldown
                line_id = self.sub_ids_to_line(sub_id_or, sub_id_ex, link_id)
                if state.time_before_cooldown_line[line_id]:
                    link_mask[i_link] = False
                    continue

        return link_mask

    def n_links(self) -> int:
        """Computes the number of links in the graph.
        :return: The link count.
        """
        return self.n_line * 4 + self.n_gen * 2 + self.n_load * 2

    def n_nodes(self) -> int:
        """Computes the number of nodes in the graph.
        :return: The node count.
        """
        return self.n_load + self.n_gen + 2 * self.n_sub


"""Implements an RL-based power grid agent that uses a trained torch policy with predicts discrete topology change
    actions. Given this policy a beam search is performed whenever a critical state would lead to a diverging power flow
    (blackout). Additionally the presence of the CriticalStateSimulationObservation Wrapper is assumed, such that the
    agent only queries the policy in case a critical state is encountered. Otherwise the noop action is performed."""

import copy
import dataclasses
import math
import os
from collections import Counter
from datetime import timedelta
from itertools import chain
from typing import Union, Any, Dict, Tuple, Optional, List, Sequence

import grid2op
import numpy as np
from grid2op.Action import PlayableAction
from grid2op.Agent import BaseAgent
from grid2op.Exceptions import DivergingPowerFlow
from grid2op.Observation import CompleteObservation
from grid2op.Reward import LinesCapacityReward, AlarmReward
from gym import spaces
from maze.core.agent.serialized_torch_policy import SerializedTorchPolicy
from maze.core.agent.torch_policy import TorchPolicy
from maze.core.env.action_conversion import ActionType
from maze.core.env.maze_env import MazeEnv
from maze.core.env.observation_conversion import ObservationType
from maze.core.env.structured_env import StructuredEnv
from maze.core.env.structured_env_spaces_mixin import StructuredEnvSpacesMixin
from maze.core.utils.config_utils import EnvFactory, read_config
from maze.core.utils.seeding import set_seeds_globally, MazeSeeding
from maze.core.wrappers.wrapper import ActionWrapper
from maze.core.wrappers.wrapper import ObservationWrapper, Wrapper
from omegaconf import OmegaConf, DictConfig

from maze_l2rpn.env.core_env import Grid2OpCoreEnvironment
from maze_l2rpn.utils import prepare_simulated_env
from maze_l2rpn.wrappers.critical_state_observer_simulate_wrapper import CriticalStateObserverSimulateWrapper

# path to experiment (training artifacts)
EXPERIMENT_PATH = 'experiment_data'
WRAPPER_PATH_OBS_NORM = 'maze.core.wrappers.observation_normalization.observation_normalization_wrapper.' \
                        'ObservationNormalizationWrapper'
# The max rho used in the simulated environment.
SIMULATION_MAX_RHO = 0.97
# Optimal alarm delta for raising the alarm.
OPTIMAL_ALARM_DELTA = 7

ActionHash = Tuple[Tuple[int, ...]]


@dataclasses.dataclass
class SimulationResult:
    """Dataclass holding the simulation results as it is returned from the grid2op simulator."""

    simulated_maze_state: CompleteObservation
    """The simulated complete grid2op maze_state."""

    reward: float
    """The simulated reward."""

    done: bool
    """The simulated done."""

    info: Dict
    """The simulated info dict."""


@dataclasses.dataclass
class CompleteSimulationResult:
    """A dataclass for holding the complete simulation results as links to previous results of the same path."""

    simulation_result: SimulationResult
    """The simulation result."""

    simulation_score: float
    """The numeric simulation score."""

    maze_action: ActionType
    """The maze action at the current step."""

    playable_action: PlayableAction
    """The playable action at the current step."""

    simulated_observation: ObservationType
    """The simulated maze observation."""

    previous_complete_simulation_result: Optional['CompleteSimulationResult']
    """Optional value holding the complete simulation result of the previous simulated step."""

    action_probs_sequence: List[float]
    """The current action probability sequence."""

    is_leaf = False
    """Whether the current step is a leaf."""

    def line_status_till_simulation_horizon(self) -> List[np.ndarray]:
        """Retrieve a list of line-status arrays w.r.t. to the current simulation sequence.
        :return: The list of line status arrays.
        """
        if self.previous_complete_simulation_result is not None:
            return self.previous_complete_simulation_result.line_status_till_simulation_horizon() + [
                self.simulation_result.simulated_maze_state.line_status]
        else:
            return [self.simulation_result.simulated_maze_state.line_status]

    @property
    def resolves_critical_state(self) -> bool:
        """Property indicating whether the current simulation path resolved the critical state.

        :return: A boolean indicating whether the simulation path resolves the critical state.
        """
        return self.simulated_observation['critical_state'] == 0 and not self.simulation_result.done

    def get_complete_action_probs_sequence(self) -> List[float]:
        """Retrieve the complete action probability sequence of the current simulation path.

        :return: The complete action probability sequence of all simulated steps.
        """
        if self.previous_complete_simulation_result is not None:
            return self.previous_complete_simulation_result.get_complete_action_probs_sequence() + \
                   self.action_probs_sequence
        else:
            return self.action_probs_sequence

    def get_action_path_prob(self) -> float:
        """Get the combined probability estimate of the current simulation path under the independence assumption.

        :return: The combined probability estimate of the current simulation path.
        """
        return float(np.prod([pp for pp in self.get_complete_action_probs_sequence()]))

    def get_simulation_horizon(self) -> int:
        """Return the length of the current simulation path.

        :return: The length of the current simulation path.
        """
        if self.previous_complete_simulation_result is not None:
            return self.previous_complete_simulation_result.get_simulation_horizon() + 1
        else:
            return 1


def pass_action_through_wrapper_stack(env: Union[StructuredEnv, StructuredEnvSpacesMixin, MazeEnv],
                                      maze_action: ActionType) -> ActionType:
    """Get the action by recursively passing it thorough the wrapper stack. (Happens implicitly at training time.)

    :param env: The current env in the wrapper stack.
    :param maze_action: The current action in the wrapper stack.

    :return: The maze action passed through the complete wrapper stack.
    """
    if type(env).__bases__[0] == ActionWrapper:
        maze_action = env.action(maze_action)
        return pass_action_through_wrapper_stack(env.env, maze_action)
    elif type(env).__bases__[0] in (ObservationWrapper, Wrapper):
        return pass_action_through_wrapper_stack(env.env, maze_action)
    else:
        return maze_action


def pass_observation_through_wrapper_stack(env: Union[StructuredEnv, StructuredEnvSpacesMixin, MazeEnv],
                                           observation: ObservationType,
                                           force_critical_state: bool) -> ObservationType:
    """Get the observation by recursively passing it thorough the wrapper stack. (Happens implicitly at training time.)

    :param env: The current env in the wrapper stack.
    :param observation: The current observation in the wrapper stack.
    :param force_critical_state: Specify whether to force a critical state.

    :return: The maze observation passed through the complete wrapper stack.
    """

    observation, _ = pass_observation_through_wrapper_stack_rec(env, observation, force_critical_state)
    return observation


def pass_observation_through_wrapper_stack_rec(env: Union[StructuredEnv, StructuredEnvSpacesMixin, MazeEnv],
                                               observation: ObservationType,
                                               force_critical_state: bool) -> Tuple[ObservationType, bool]:
    """Get the observation by recursively passing it thorough the wrapper stack.

    :param env: The current env in the wrapper stack.
    :param observation: The current observation in the wrapper stack.
    :param force_critical_state: Specify whether to force a critical state.

    :return A tuple holding the processed maze-observation and a boolean indicating whether to stop the conversion.
    """
    if type(env).__bases__[0] in (ObservationWrapper, CriticalStateObserverSimulateWrapper):
        obs, stop_conversion = pass_observation_through_wrapper_stack_rec(env.env, observation,
                                                                          force_critical_state)
        if type(env) == CriticalStateObserverSimulateWrapper:
            env._current_state_is_critical = None
        if not stop_conversion:
            obs = env.observation(obs)
            if type(env) == CriticalStateObserverSimulateWrapper and obs[
                'critical_state'] == 0 and not force_critical_state:
                # If critical state is set, break wrapper stack, since it performs unnecessary computation
                stop_conversion = True
            elif type(env) == CriticalStateObserverSimulateWrapper and obs[
                'critical_state'] == 0 and force_critical_state:
                obs['critical_state'] = np.array(1.0)
        return obs, stop_conversion
    elif type(env).__bases__[0] in (Wrapper, ActionWrapper):
        obs, stop_conversion = pass_observation_through_wrapper_stack_rec(env.env, observation,
                                                                          force_critical_state)
        return obs, stop_conversion
    else:
        return observation, False


class ActionTree:
    """The action tree class is the basis for beam search, holds a tree of actions and implements methods for
    dynamically adding nodes and querying this tree to get not evaluated action sequences.

    :param action: Optional Maze action of the current node.
    :param action_prob: Optional action probability corresponding to the action.
    """

    def __init__(self, action: Optional[ActionType], action_prob: Optional[float]):
        self.action = action
        self.action_prob = action_prob
        self.children = None
        self.is_leaf = False

    def get_next_action_sequence_and_node_and_probs(self) -> Tuple[Union[bool, List[ActionType]], Optional[List[float]],
                                                                   Optional['ActionTree']]:
        """Retrieve the next (not yet evaluated) action sequence in the tree.

        :return: A tuple holding the action sequence, the corresponding action probability sequence and the last node of
            the sequence if a not yet evaluated sequence exits, otherwise a False, None, None
        """
        if self.children is None and not self.is_leaf:
            return [self.action] if self.action is not None else [], [
                self.action_prob] if self.action is not None else [], self
        elif self.children is None and self.is_leaf:
            return False, None, None
        elif self.children is not None and self.is_leaf:
            self.children = None
            return False, None, None
        else:
            for cc in self.children:
                child_sequence, action_prob_sequence, node = cc.get_next_action_sequence_and_node_and_probs()
                if isinstance(child_sequence, bool) and not child_sequence:
                    continue
                if isinstance(child_sequence, list):
                    return ([self.action] if self.action is not None else []) + child_sequence, \
                           ([self.action_prob] if self.action is not None else []) + action_prob_sequence, node
            # All child sequences already lead to a leave node
            return False, None, None

    def add_children(self, actions: Sequence[ActionType], probabilities: Sequence[float]) -> None:
        """Add new children to the current node.

        :param actions: The actions to add to the tree.
        :param probabilities: The probabilities corresponding to the actions.
        """

        for cc, pp in zip(actions, probabilities):
            child = self.get_child_by_action(cc)
            if child is not None:
                child.action_prob += pp
            else:
                if self.children is None:
                    self.children = list()
                self.children.append(ActionTree(cc, pp))

    def get_child_by_action(self, action: ActionType) -> Optional['ActionTree']:
        """Get a child of the current node by the action it represents.

        :parma action: The action that is looked for.

        :return: If the given action corresponds to a child then return this ActionTree otherwise None.
        """
        if self.children is None:
            return None
        for cc in self.children:
            if cc.action == action:
                return cc
        return None

    def __eq__(self, other: 'ActionTree') -> bool:
        """Check for equality between two ActionTree instances.

        :param other: The ActionTree instance to compare this element against.

        :return: A bool indicating whether self and other are the same object.
        """
        return self.action == other.action


class MultiStepSimulationTree:
    """A class for performing beam search on the grid2op environment based on a policy with only discrete actions.
        The search is carried out over multiple steps with a parameter defined branching.

    :param simulated_maze_env: The simulated environment object.
    :param maze_policy: The Maze structured torch policy.
    :param num_simulations_per_substep: A list of list indicating the branching to perform in the beam search.

    """

    def __init__(self, simulated_maze_env: Union[StructuredEnv, StructuredEnvSpacesMixin, MazeEnv,
                                                 Grid2OpCoreEnvironment],
                 maze_policy: TorchPolicy,
                 num_simulations_per_substep: List[List[int]]):

        # Environments
        self.maze_env = None
        self.simulated_maze_env = simulated_maze_env

        # Simulation variables
        self.num_simulations_per_substep = num_simulations_per_substep
        self.num_substeps = len(self.num_simulations_per_substep[0])
        self.simulated_action_sequences: Dict[ActionHash, CompleteSimulationResult] = dict()

        self.org_maze_state = None
        self.next_step_forecasting_inj = None

        # Torch policy
        self.maze_policy = maze_policy

    def reset_simulated_env(self) -> None:
        """Reset the simulated env to the state of the maze env."""
        self.simulated_maze_env.clone_from(self.maze_env)
        if isinstance(self.simulated_maze_env, CriticalStateObserverSimulateWrapper):
            self.simulated_maze_env.set_max_rho_simulate(np.inf)
            self.simulated_maze_env.set_max_rho(SIMULATION_MAX_RHO)

    def action_sequence_hash(self, actions: List[ActionType], filter_noop: bool) -> Tuple[Tuple[int, ...]]:
        """Hash the action sequence for more convenient comparison.

        :param actions: The actions to hash.
        :param filter_noop: Specify whether to filter out the noop action.

        :return: The action hash as a Tuple of tuples of int.
        """
        noop_idx = self.simulated_maze_env.action_conversion.n_links
        link_sub_action_idxs = list()
        link_idxs: List[int] = list()
        for idx, action in enumerate(actions):
            assert 'link_to_set' in action
            assert len(action) == 1
            if isinstance(action['link_to_set'], int):
                link_idxs.append(action['link_to_set'])
            elif isinstance(action['link_to_set'], np.ndarray) and action['link_to_set'].shape == ():
                link_idxs.append(int(action['link_to_set']))
            elif isinstance(action['link_to_set'], np.ndarray) and action['link_to_set'].shape == (1,):
                link_idxs.append(int(action['link_to_set'][0]))
            else:
                raise ValueError
            if len(link_idxs) == len(self.num_simulations_per_substep[0]) or idx == len(actions) - 1:
                if filter_noop:
                    link_idxs = list(filter(lambda x: x != noop_idx, link_idxs))
                link_sub_action_idxs.append(tuple(sorted(link_idxs)))
                link_idxs = list()
        return tuple(link_sub_action_idxs)

    def action_not_yet_stepped(self, already_stepped_actions: List[ActionType],
                               action_sequence: List[ActionType]) -> List[ActionType]:
        """Return the actions that have not yet been stepped by the env from a list of actions already stepped and a
        target list of actions to step.

        :param already_stepped_actions: The actions already stepped in the env.
        :param action_sequence: The action sequence that should be stepped.
        :return: The action sequence the agent should perform.
        """
        action_not_yet_stepped_by_substep = list()
        for idx in range(math.ceil(len(action_sequence))):
            already_stepped_actions_step = already_stepped_actions[idx: idx + 1]
            action_sequence_step = action_sequence[idx: idx + 1]
            already_stepped_actions_counter = Counter(
                self.action_sequence_hash(already_stepped_actions_step, filter_noop=False))
            action_sequence_counter = Counter(self.action_sequence_hash(action_sequence_step, filter_noop=False))
            action_not_taken = list((action_sequence_counter - already_stepped_actions_counter).elements())
            action_not_yet_stepped_by_substep.append([{'link_to_set': np.array(idx)} for idx in action_not_taken])
        return list(chain.from_iterable(action_not_yet_stepped_by_substep))

    def get_simulation_maze_state(self, current_depth: int, action_sequence_hash: ActionHash) -> CompleteObservation:
        """Return the maze_state needed for simulating an action.

        :param current_depth: The current simulation depth in the search tree.
        :param action_sequence_hash: The current action_sequence_hash.
        :return: A complete maze_state ready for simulating on.
        """
        if current_depth == 0:
            maze_state = self.org_maze_state.copy()
        else:
            # Get previous action hash
            one_step_down_action_hash = action_sequence_hash[:current_depth]
            if not isinstance(one_step_down_action_hash[0], tuple):
                one_step_down_action_hash = (one_step_down_action_hash,)
            sim_maze_sate = self.simulated_action_sequences[one_step_down_action_hash].simulation_result.simulated_maze_state
            sim_maze_sate.action_helper = copy.deepcopy(self.org_maze_state.action_helper)
            sim_maze_sate._obs_env = self.org_maze_state._obs_env.copy()

            sim_maze_sate._obs_env.times_before_line_status_actionable_init = sim_maze_sate.time_before_cooldown_line
            sim_maze_sate._obs_env.times_before_topology_actionable_init = sim_maze_sate.time_before_cooldown_sub
            sim_maze_sate._obs_env.duration_next_maintenance_init = sim_maze_sate.time_next_maintenance

            next_step = (self.next_step_forecasting_inj[0] + timedelta(minutes=5 * current_depth),
                         self.next_step_forecasting_inj[1])
            sim_maze_sate._forecasted_inj = [self.org_maze_state._forecasted_inj[-1], next_step]
            maze_state = sim_maze_sate
        return maze_state

    def simulate_action(self, simulation_maze_state: CompleteObservation, action_sequence_hash: ActionHash,
                        step_action: ActionType, current_depth: int, action_probs_sequence: List[float]) \
            -> Tuple[ObservationType, CompleteSimulationResult]:
        """For a given maze_state and action, simulate the next expected maze_state.

        If the action has already been performed retrieve the simulation result, otherwise perform the simulation.

        :param simulation_maze_state: The maze_state to be used for the simulation.
        :param action_sequence_hash: The current action sequence hash.
        :param step_action: The action to simulate.
        :param current_depth: The current simulation depth in the search tree.
        :param action_probs_sequence: The action probability sequence.

        :return: A tuple holding the resulting maze observation and the complete simulation result.
        """

        if action_sequence_hash not in self.simulated_action_sequences:
            playable_action = self.simulated_maze_env.action_conversion.space_to_maze(step_action,
                                                                                      simulation_maze_state)
            simulation_result = SimulationResult(*simulation_maze_state.simulate(playable_action, time_step=1))
            if simulation_result.info['is_illegal'] or simulation_result.info['is_ambiguous'] or len(
                    simulation_result.info['exception']) > 0 and not isinstance(simulation_result.info['exception'][0],
                                                                                DivergingPowerFlow):
                simulation_result.done = True

            simulation_result.simulated_maze_state.time_before_cooldown_sub[
                simulation_maze_state.time_before_cooldown_sub > 0] = \
                simulation_maze_state.time_before_cooldown_sub[simulation_maze_state.time_before_cooldown_sub > 0] - 1
            simulation_result.simulated_maze_state.time_before_cooldown_line[
                simulation_maze_state.time_before_cooldown_line > 0] = \
                simulation_maze_state.time_before_cooldown_line[
                    simulation_maze_state.time_before_cooldown_line > 0] - 1
            simulation_result.simulated_maze_state.time_next_maintenance[
                simulation_maze_state.time_next_maintenance >= 0] = \
                simulation_maze_state.time_next_maintenance[simulation_maze_state.time_next_maintenance >= 0] - 1

            # Set simulated env state to result of simulation
            self.simulated_maze_env.set_maze_state(simulation_result.simulated_maze_state)
            # Convert simulated new maze_state to observation
            observation = self.simulated_maze_env.observation_conversion.maze_to_space(
                simulation_result.simulated_maze_state)
            observation = pass_observation_through_wrapper_stack(self.simulated_maze_env, observation, False)

            # Retrieve previous Simulation result if current depth > 0
            if current_depth > 0:
                step_one_key = action_sequence_hash[:current_depth]
                if not isinstance(step_one_key[0], tuple):
                    step_one_key = (step_one_key,)
                previous_complete_simulation_result = self.simulated_action_sequences[step_one_key]
            else:
                previous_complete_simulation_result = None

            # Retrieve relevant action probability sequence
            relevant_action_probs_sequence = action_probs_sequence[current_depth * self.num_substeps:
                                                                   (current_depth + 1) * self.num_substeps]

            # Compute simulation score with scoring function
            simulation_score = max(simulation_result.simulated_maze_state.rho) if not simulation_result.done else \
                np.inf

            # Create complete simulation result
            complete_simulation_result = CompleteSimulationResult(
                simulation_result=simulation_result, simulation_score=simulation_score, maze_action=step_action,
                playable_action=playable_action, simulated_observation=observation,
                previous_complete_simulation_result=previous_complete_simulation_result,
                action_probs_sequence=relevant_action_probs_sequence)

            # Store complete simulation result
            self.simulated_action_sequences[action_sequence_hash] = complete_simulation_result

        else:
            # If action sequence has already been simulated retrieve it.
            complete_simulation_result = self.simulated_action_sequences[action_sequence_hash]

            # Pass the maze_state through the wrapper stack
            self.simulated_maze_env.set_maze_state(complete_simulation_result.simulation_result.simulated_maze_state)
            observation = self.simulated_maze_env.observation_conversion.maze_to_space(
                complete_simulation_result.simulation_result.simulated_maze_state)
            observation = pass_observation_through_wrapper_stack(self.simulated_maze_env, observation, False)

        return observation, complete_simulation_result

    def retrieve_top_actions(self, observation: ObservationType, last_path_to_best_score: List[ActionHash],
                             maze_state: CompleteObservation,
                             maze_env: Union[MazeEnv]) -> Dict[ActionHash, CompleteSimulationResult]:
        """Retrieve the top actions for the current time step based on the search parameters of the class

        :param observation: The current maze observation.
        :param last_path_to_best_score: The most probable path of the previous step.
        :param maze_state: The complete grid2op maze_state.
        :param maze_env: The maze env at the current time step.

        :return: All simulated action sequences as a dict.
        """

        # Initialize objects for current time step
        self.simulated_action_sequences: Dict[ActionHash, CompleteSimulationResult] = dict()
        self.org_maze_state = maze_state
        self.next_step_forecasting_inj = copy.deepcopy(self.org_maze_state._forecasted_inj[-1])
        self.maze_env = maze_env
        # Currently stepped action sequence
        action_sequence_already_stepped = []
        # Variable for recording whether the last step was a leaf node in order to reduce cloning operations by 1
        last_step_was_leaf = True
        # Initialize the search tree root node
        root = ActionTree(None, None)

        # Initialize whether to add the predicted best path to the tree
        plan_added = not (last_path_to_best_score is not None and len(last_path_to_best_score) > 0)

        while True:
            # Retrieve next action sequence from action tree
            action_sequence_to_step, action_probs_sequence, node = root.get_next_action_sequence_and_node_and_probs()

            # Add previous planed action path to action tree
            if isinstance(action_sequence_to_step, bool) and not action_sequence_to_step and not plan_added:
                node = root
                for action_idx in last_path_to_best_score[0][0]:
                    action = {'link_to_set': action_idx}
                    node.add_children([action], [0.1])
                    node = node.get_child_by_action(action)
                plan_added = True
                action_sequence_to_step, action_probs_sequence, node = \
                    root.get_next_action_sequence_and_node_and_probs()

            # If action sequence to step is False, all actions have been computed and simulated
            if isinstance(action_sequence_to_step, bool) and not action_sequence_to_step and plan_added:
                break

            # Starting new Action Sequence.. clone simulated env
            if len(action_sequence_already_stepped) == 0 and last_step_was_leaf:
                self.reset_simulated_env()

            # Retrieve action not yet taken in time-order
            action_not_taken = self.action_not_yet_stepped(action_sequence_already_stepped, action_sequence_to_step)

            # Step through the simulated env
            for action in action_not_taken:
                try:
                    observation_or_action, multistep_done = self.simulated_maze_env.rollout_step(action)
                    action_sequence_already_stepped.append(action)
                except Exception:
                    # In case an exception is encountered, set the current node as a leaf and continue (prune path)
                    node.is_leaf = True
                    break

                current_depth = len(action_sequence_already_stepped) // len(self.num_simulations_per_substep[0]) - 1
                # If multistep done, simulate the next action
                if multistep_done:
                    # Retrieve the action sequence hash used for storing the results
                    action_sequence_hash = self.action_sequence_hash(action_sequence_already_stepped, filter_noop=True)
                    # Pass the computed action through the wrapper stack
                    step_action = pass_action_through_wrapper_stack(self.simulated_maze_env, observation_or_action)
                    # Retrieve the simulation maze_state
                    simulation_maze_state = self.get_simulation_maze_state(current_depth, action_sequence_hash)
                    # Simulate the action and store result
                    observation, complete_simulation_result = self.simulate_action(
                        simulation_maze_state=simulation_maze_state,
                        action_sequence_hash=action_sequence_hash, step_action=step_action,
                        current_depth=current_depth, action_probs_sequence=action_probs_sequence)

                    # Check whether the current sequence is finished
                    if observation['critical_state'] == 0 or complete_simulation_result.simulation_result.done or \
                            len(action_sequence_already_stepped) == len(self.num_simulations_per_substep) * \
                            self.num_substeps:
                        node.is_leaf = True
                        complete_simulation_result.is_leaf = True
                        break
                # In case the multistep is not yet done, the returned object is the observation
                else:
                    observation = observation_or_action

            # Record whether the last step encountered a leaf node
            last_step_was_leaf = node.is_leaf

            # If the current node is not a leaf node, retrieve new actions from the policy w.r.t. the the specified
            #   branching.
            if not node.is_leaf:
                # Retrieve the actor id
                actor_id = self.simulated_maze_env.actor_id()
                # Infer the number of actions to retrieve
                max_allowed_actions = len(np.where(observation['link_to_set_mask'])[0])
                num_candidates = min(max_allowed_actions, self.num_simulations_per_substep[
                    len(action_sequence_to_step) // self.num_substeps][
                    len(action_sequence_to_step) % self.num_substeps])

                # Retrieve the actions and action-probabilities
                top_actions, probs = self.maze_policy.compute_top_action_candidates(observation, actor_id=actor_id,
                                                                                    num_candidates=num_candidates,
                                                                                    maze_state=None, env=None)
                # Add the action as leaf nodes to the current node
                node.add_children(top_actions, probs)
            else:
                # Leaf encountered, reset values
                action_sequence_already_stepped = []

        return self.simulated_action_sequences


class MyStructuredLoadedAgentCritical(BaseAgent):
    """
    Maze environment based power grid agent that uses a trained torch policy (with only discrete actions) as a
    basis. Building on this a beam search is performed if a critical state leads to a blackout. Additionally the
    presence of the CriticalStateSimulationObservation Wrapper is assumed, such that the  agent only queries the policy
    in case a critical state is encountered. Otherwise the noop action is performed.

    :param action_space: The action space of the env.
    :param maze_env: The initialized env.
    :param simulated_env: The simulated initialized environment.
    :param structured_policy: The initialized structured policy with the loaded models.
    """

    def __init__(self, action_space: spaces.Space,
                 maze_env: Union[StructuredEnv, StructuredEnvSpacesMixin, MazeEnv, Grid2OpCoreEnvironment],
                 simulated_env: Union[StructuredEnv, StructuredEnvSpacesMixin, MazeEnv, Grid2OpCoreEnvironment],
                 structured_policy: TorchPolicy):
        BaseAgent.__init__(self, action_space=action_space)

        # Environments
        self.maze_env = maze_env
        self.simulated_env = simulated_env

        # Policy
        self.structured_policy = structured_policy

        # Random number generators
        self.policy_rng = np.random.RandomState()
        self.env_rng = np.random.RandomState()

        # Stats
        self._last_alarm_raised_at_step = -np.inf
        self.path_to_best_score = None
        self.step_count = -np.inf

        # Initialize multi-step-simulation-tree for encountered critical states
        self.msst_most_probable = MultiStepSimulationTree(simulated_maze_env=simulated_env,
                                                          maze_policy=self.structured_policy,
                                                          num_simulations_per_substep=[[1, 1, 1], [1, 1, 1], [1, 1, 1],
                                                                                       [1, 1, 1],
                                                                                       [1, 1, 1], [1, 1, 1]])

        # Initialize multi-step-simulation-tree for for encountered potential blackouts
        self.msst_search = MultiStepSimulationTree(simulated_maze_env=simulated_env,
                                                   maze_policy=self.structured_policy,
                                                   num_simulations_per_substep=[[4, 2, 1], [2, 1, 1], [2, 1, 1],
                                                                                [1, 1, 1], [1, 1, 1], [1, 1, 1]])

    def act(self, observation: CompleteObservation, reward: float, done: bool = False) -> PlayableAction:
        """The action that your agent will choose depending on
        the maze_state, the reward, and whether the state is terminal.

        :param maze_state: The maze_state passed from the (unseen) evaluation env.
        :param reward: The reward passed from the (unseen) evaluation env.
        :param done: The done passed from the (unseen) evaluation env. (This is always False).

        :return: The action to be takes next.
        """
        maze_state = observation

        # Check if new scenario started
        if maze_state.current_step < self.step_count:
            self.reset_components()
        self.step_count = maze_state.current_step

        # Process the given maze_state --------------------------------------------------------------------------------

        # Set the maze state
        self.maze_env.set_maze_state(maze_state)
        # Convert the maze action to a space
        observation = self.maze_env.observation_conversion.maze_to_space(maze_state)
        # Recursively pass the observation through the wrapper stack
        observation = pass_observation_through_wrapper_stack(self.maze_env, observation, False)
        # Check if current state is critical one
        is_safe_state = 'critical_state' in observation and observation['critical_state'] == 0

        # Compute the topology action to perform -----------------------------------------------------------------------

        # critical state encountered, ask agent to resolve
        if not is_safe_state:
            original_path_to_best_score = copy.deepcopy(self.path_to_best_score)
            # try to resolve critical state with most probable path
            action, most_probable_path = self.best_action_by_tree_simulation(maze_state, observation,
                                                                             self.msst_most_probable)
            # scale up simulation to find better solution (action)
            if most_probable_path.simulation_result.done:
                self.path_to_best_score = original_path_to_best_score
                action, search_most_probable_path = self.best_action_by_tree_simulation(
                    maze_state, observation, self.msst_search)
                # could not resolve, pick most probable action sequence
                if search_most_probable_path.simulation_result.done:
                    most_probable_path = search_most_probable_path
        # all good, nothing to do
        else:
            action = self.maze_env.action_conversion.noop_action()
            most_probable_path = None
            if self.path_to_best_score is not None and len(self.path_to_best_score) > 0:
                observation = pass_observation_through_wrapper_stack(self.maze_env, observation, True)
                action, most_probable_path = self.best_action_by_tree_simulation(
                    maze_state, observation, self.msst_most_probable)
        done_in_sim = most_probable_path is not None and most_probable_path.simulation_result.done

        # Compute alarm action to perform ------------------------------------------------------------------------------

        if done_in_sim and self._last_alarm_raised_at_step + OPTIMAL_ALARM_DELTA < self.step_count and \
                maze_state.attention_budget >= 1:
            action.update(self.compute_attention_action(maze_state, most_probable_path))

        # Process the computed action ----------------------------------------------------------------------------------

        # Pass the action through the wrapper stack
        action = pass_action_through_wrapper_stack(self.maze_env, action)
        # In case the env is expected to fail in the next step only perform noop action (controlled failure).
        if done_in_sim and most_probable_path.get_simulation_horizon() == 1:
            action = {}
        # Convert the action back to a playable action
        playable_action = self.maze_env.action_conversion.space_to_maze(action, maze_state)
        return playable_action

    def reset_components(self) -> None:
        """Reset the individual components of the submission agent as well as the statistics."""
        self.maze_env.reset()

        # Reset stats
        self._last_alarm_raised_at_step = -np.inf
        self.step_count = -np.inf
        self.path_to_best_score = None

    def seed(self, seed: int) -> None:
        """The seeding method of the base agent

        :param seed: The seed to set.
        """
        # Generate an agent seed and set the seed globally for the model initialization
        self.policy_rng = np.random.RandomState(seed)
        set_seeds_globally(MazeSeeding.generate_seed_from_random_state(self.policy_rng), True,
                           info_txt=f'training runner (Pid:{os.getpid()})')
        self.maze_env.seed(MazeSeeding.generate_seed_from_random_state(self.policy_rng))

    @staticmethod
    def filter_simulation_results(simulation_results: Dict[ActionHash, CompleteSimulationResult],
                                  maze_state: CompleteObservation) -> Tuple[Dict[ActionHash, CompleteSimulationResult],
                                                                             bool]:
        """Filter the dict of simulation results and keep only paths not resulting in a blackout.

        :param simulation_results: The simulation results dict.
        :param maze_state: The current maze_state:

        :return: A tuple fo the filtered simulation results dict as well as a bool indicating whether all actions are
            bad.
        """

        # Remove actions path resulting in blackouts
        current_action_candidates = dict(
            filter(lambda item: item[1].simulation_score < np.inf, simulation_results.items()))

        # In case all action lead to blackout, chose from the list of all actions
        all_bad_actions = len(current_action_candidates) == 0
        if all_bad_actions:
            current_action_candidates = simulation_results

        def is_path_to_risky(candidate: CompleteSimulationResult, current_maze_state: CompleteObservation) -> bool:
            """Decide whether action sequences are too risky. Action sequences are deemed too risky if they increase the
             rho too much in order to arrive at a decreased state.

             :param candidate: A complete simulation results holding one possible action sequence path.
             :param current_maze_state: The current maze_state.

             :return: A boolean value indicating whether path is too risky.
             """
            base_rho = current_maze_state.rho
            last_rho = candidate.simulation_result.simulated_maze_state.rho

            previous_rhos = list()
            current_simulation_result = candidate.previous_complete_simulation_result
            while True:
                if current_simulation_result is None:
                    break

                previous_rhos.append(current_simulation_result.simulation_result.simulated_maze_state.rho)
                current_simulation_result = current_simulation_result.previous_complete_simulation_result

            if len(previous_rhos) == 0:
                return False

            previous_rhos = previous_rhos[::-1]
            if any([max(np.max((base_rho, last_rho), axis=0)) + 0.1 < max(
                    previous_rho)
                    for idx, previous_rho in enumerate(previous_rhos)]):
                return True
            else:
                return False

        current_action_candidates = dict(filter(lambda x: not is_path_to_risky(x[1], maze_state),
                                                current_action_candidates.items()))

        # Choose all action that resolve the critical state
        actions_resolving_critical_state = dict(
            filter(lambda x: x[1].resolves_critical_state, current_action_candidates.items()))

        # If no action sequence resolved the critical state filter out the short ones
        if len(actions_resolving_critical_state) == 0:
            max_simulated_length = max(map(len, current_action_candidates.keys()))
            current_action_candidates = dict(filter(lambda x: len(x[0]) == max_simulated_length,
                                                    current_action_candidates.items()))

            min_rho = min(
                map(lambda x: max(x[1].simulation_result.simulated_maze_state.rho), current_action_candidates.items()))

            current_action_candidates = dict(filter(lambda x: np.isclose(
                max(x[1].simulation_result.simulated_maze_state.rho),
                min_rho, atol=0.1),
                                                    current_action_candidates.items()))

        elif len(actions_resolving_critical_state) > 0:
            # If some actions resolve the critical state, pick them. (Only version where we should compare action of
            #   different length... )
            current_action_candidates = actions_resolving_critical_state

        return current_action_candidates, all_bad_actions

    def best_action_by_tree_simulation(self, maze_state: CompleteObservation, observation: ObservationType,
                                       msst: MultiStepSimulationTree) -> Tuple[ActionType, CompleteSimulationResult]:
        """Method for sampling possible actions with the given MultiStepSimulationTree and selecting the best candidate.

        :param maze_state: Complete maze_state.
        :param observation: Processed Maze observation.
        :param msst: The MultistepSimulation tree to be used.

        :return: Tuple of the maze action to take and the expected path the action returned action will result in.
        """

        # Retrieve the simulation results with the given MultiStepSimulationTree
        simulation_results = msst.retrieve_top_actions(observation=observation,
                                                       maze_state=maze_state,
                                                       maze_env=self.maze_env,
                                                       last_path_to_best_score=self.path_to_best_score)

        # Filter simulation results
        current_action_candidates, all_bad_actions = self.filter_simulation_results(simulation_results, maze_state)

        # Final sorting of the action candidates
        # sort the candidates by simulation score first and action probability second
        current_action_candidates = dict(
            sorted(current_action_candidates.items(),
                   key=lambda x: (-x[1].get_action_path_prob(), x[1].simulation_score)))

        top_simulation_result = list(current_action_candidates.items())[0]
        top_simulation_hash = top_simulation_result[0]
        top_level_0_action_hash = (top_simulation_hash[0],)
        top_level_0_action = simulation_results[top_level_0_action_hash].maze_action

        # If not all actions are bad actions (leading to blackout) store the best path in order to add it to next steps
        #   tree search.
        if all_bad_actions:
            self.path_to_best_score = None
        else:
            self.path_to_best_score = [(vv,) for vv in top_simulation_hash[1:]]

        # Compute the most probable path the selected action will result in
        most_probable_path = list(
            sorted(filter(lambda x: top_level_0_action_hash[0] == x[0][0], simulation_results.items()),
                   key=lambda x: (-len(x[0]), -x[1].get_action_path_prob(), x[1].simulation_score)))

        return top_level_0_action, most_probable_path[0][1]

    def compute_attention_action(self, maze_state: CompleteObservation,
                                 most_probable_path: CompleteSimulationResult) -> Dict[str, np.ndarray]:
        """Compute the attention action based on the current maze_state and the most probable path the action will take
         in the next few steps.

         This methods assumes that the most_probable_path given will end in a blackout. As such the line_status of the
         path are traversed in reverse order in order to find the origin of the cascading line failure.

        :param maze_state: The current maze_state.
        :param most_probable_path: The most probable path the agent will take over the next few steps based on
            simulation and beam search.
        :return: The alarm action as a dictionary, which is empty in case no alarm should be raised. Otherwise it holds
            a list of indices of the areas the alarm should be raised in.
        """
        simulation_step_cascading_fail_detected = 0
        if most_probable_path is None:
            disc_lines = np.argsort(maze_state.rho)[::-1][:1]
        else:
            assert most_probable_path.simulation_result.done
            disconnected_lines = [(np.ones(maze_state.line_status.shape, dtype=np.bool))]
            for line_status_current in most_probable_path.line_status_till_simulation_horizon()[::-1]:
                simulation_step_cascading_fail_detected += 1
                new_elem = disconnected_lines[-1] & ~line_status_current
                if all(~disconnected_lines[-1]):
                    break
                else:
                    disconnected_lines.append(new_elem)

            disc_lines = np.where(disconnected_lines[-1])[0]

        if len(disc_lines) == 0 or len(disc_lines) > 10:
            disc_lines = np.argsort(maze_state.rho)[::-1][:1]

        zone_for_each_lines = self.maze_env.wrapped_env.alarms_lines_area
        alarms_area_names = self.maze_env.wrapped_env.alarms_area_names

        zones_these_lines = set()
        for line_id in disc_lines:
            line_name = maze_state.name_line[line_id]
            for zone_name in zone_for_each_lines[line_name]:
                zones_these_lines.add(zone_name)

        zones_these_lines = list(zones_these_lines)
        zones_ids_these_lines = np.array([alarms_area_names.index(zone) for zone in zones_these_lines])
        if simulation_step_cascading_fail_detected > 2:
            self._last_alarm_raised_at_step = self.step_count
            action = {'raise_alarm': zones_ids_these_lines}
        else:
            action = {}
        return action

    @classmethod
    def build(cls, grid2op_env: grid2op.Environment.Environment,
              this_directory_path: str) -> 'MyStructuredLoadedAgentCritical':
        """Factory method for creating the MyStructuredLoadedAgentCritical class object.

        :param grid2op_env: The initialized grid2op environment.
        :param this_directory_path: The current directory path.
        :return: The Initialized agent to be used for evaluation.
        """

        # Retrieve experiment files
        hydra_config_path = os.path.join(this_directory_path, EXPERIMENT_PATH, '.hydra/config.yaml')
        spaces_dict_path = os.path.join(this_directory_path, EXPERIMENT_PATH, 'spaces_config.pkl')
        state_dict_path = os.path.join(this_directory_path, EXPERIMENT_PATH, 'state_dict.pt')

        # Parse Hydra config file
        hydra_config_unresolved = DictConfig(read_config(hydra_config_path))
        hydra_config: Dict[str, Any] = OmegaConf.to_container(hydra_config_unresolved, resolve=True)

        # Update the observation normalization wrapper path for statistics
        assert WRAPPER_PATH_OBS_NORM in hydra_config['wrappers']
        hydra_config['wrappers'][WRAPPER_PATH_OBS_NORM]['statistics_dump'] = \
            os.path.join(this_directory_path, EXPERIMENT_PATH,
                         hydra_config['wrappers'][WRAPPER_PATH_OBS_NORM]['statistics_dump'])

        # Create the config for the simulated environment
        simulated_env_config = {'_target_': 'maze.core.utils.config_utils.make_env',
                                'env': hydra_config['env'],
                                'wrappers': DictConfig(hydra_config['wrappers'])}

        # substitute given environment
        hydra_config['env']['core_env']['power_grid'] = grid2op_env
        simulated_env_config['env']['core_env']['power_grid'] = grid2op_env

        # Build Maze-Environment
        maze_env = EnvFactory(hydra_config['env'], hydra_config['wrappers'])()

        # Build Simulation-Maze-Environment
        exclude_wrappers = ['maze.core.wrappers.monitoring_wrapper.MazeEnvMonitoringWrapper',
                            'maze.core.wrappers.log_stats_wrapper.LogStatsWrapper']
        simulated_env = prepare_simulated_env(exclude_wrappers=exclude_wrappers, main_env=maze_env,
                                              policy_rng=np.random.RandomState(),
                                              simulated_env=simulated_env_config)
        simulated_env.set_max_rho_simulate(np.inf)
        simulated_env.set_max_rho(SIMULATION_MAX_RHO)

        # Build Torch Policy
        torch_policy = SerializedTorchPolicy(hydra_config['model'], state_dict_path, spaces_dict_path, device='cpu')

        # Initialize Submission Agent
        return cls(action_space=maze_env.action_space, maze_env=maze_env, structured_policy=torch_policy,
                   simulated_env=simulated_env)


# Add additional rewards for more information on the agent
other_rewards = {"LinesCapacityReward": LinesCapacityReward,
                 "AlarmReward": AlarmReward}


def make_agent(grid2op_env: grid2op.Environment.Environment, this_directory_path: str) -> BaseAgent:
    """Build the desired agent for the evaluation

    :param grid2op_env: The initialized grid2op environment.
    :param this_directory_path: The current directory path.
    :return: The Initialized agent to be used for evaluation.
    """
    return MyStructuredLoadedAgentCritical.build(grid2op_env, this_directory_path)

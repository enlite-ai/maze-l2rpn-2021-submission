"""Contains a random policy for the link prediction environment."""
import numpy as np
from gym import spaces
from typing import Union, Dict, Optional, Tuple, Sequence

from maze.core.agent.policy import Policy
from maze.core.annotations import override
from maze.core.env.action_conversion import ActionType
from maze.core.env.base_env import BaseEnv
from maze.core.env.maze_state import MazeStateType
from maze.core.env.observation_conversion import ObservationType
from maze.core.env.structured_env import ActorID
from maze.core.utils.seeding import MazeSeeding

from maze_l2rpn.env.core_env import L2RPNSeedInfo


class RandomLinkPredictionEnvPolicy(Policy):
    """ A policy that samples random actions for the link prediction environment considering the action mask.

    :param action_spaces_dict: The action spaces.
    """

    def __init__(self, action_spaces_dict: Dict[Union[int, str], spaces.Dict]):
        super().__init__()

        self.action_spaces = action_spaces_dict
        self.rng = np.random.RandomState(None)

    @override(Policy)
    def seed(self, seed: Union[L2RPNSeedInfo, int]) -> None:
        """Seed the policy by setting the action space seeds."""

        # Convert to int, if passed as L2RPNSeedInfo.
        if isinstance(seed, L2RPNSeedInfo):
            seed = seed.random_seed

        self.rng = np.random.RandomState(seed)
        for key, action_space in self.action_spaces.items():
            action_space.seed(MazeSeeding.generate_seed_from_random_state(self.rng))

    @override(Policy)
    def needs_state(self) -> bool:
        """This policy does not require the state object to compute the action."""
        return False

    @override(Policy)
    def needs_env(self) -> bool:
        """This policy does not require the env object to compute the action."""
        return False

    @override(Policy)
    def compute_action(self, observation: ObservationType, maze_state: Optional[MazeStateType], env: Optional[BaseEnv],
                       actor_id: Optional[ActorID] = None, deterministic: bool = False) -> ActionType:
        """Samples a random link change action.
        Implementation of :py:attr:~maze.core.agent.policy.Policy.compute_action.
        """

        # init action dict
        action = dict()

        if actor_id is None:
            assert len(self.action_spaces) == 1
            policy_id = list(self.action_spaces.keys())[0]
        else:
            policy_id = actor_id.step_key

        # sample discrete redispatch action
        for action_key in self.action_spaces[policy_id].spaces.keys():
            if "redispatch_" in action_key:
                action[action_key] = 0

        # sample continuous redispatch action
        if "redispatch" in self.action_spaces[policy_id].spaces:
            action["redispatch"] = np.zeros_like(self.action_spaces[policy_id]["redispatch"].sample())

        # sample link prediction action
        if "link_to_set" in self.action_spaces[policy_id].spaces:
            action["link_to_set"] = self.rng.choice(np.nonzero(observation["link_to_set_mask"])[0])

        return action

    @override(Policy)
    def compute_top_action_candidates(self, observation: ObservationType, num_candidates: int,
                                      maze_state: Optional[MazeStateType], env: Optional[BaseEnv],
                                      actor_id: Union[str, int] = None) -> Tuple[Sequence[ActionType], Sequence[float]]:
        """
        Implementation of :py:attr:`~maze.core.agent.policy.Policy.compute_top_action_candidates`.
        """

        raise NotImplementedError

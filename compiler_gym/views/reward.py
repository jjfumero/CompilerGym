# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings
from typing import Callable, Dict, List, Optional

from compiler_gym.spaces import RewardSpace


class RewardView(object):
    """A view into a set of reward spaces.

    Example usage:

    >>> env = gym.make("llvm-v0")
    >>> env.reset()
    >>> env.reward.spaces["codesize"].range
    (-np.inf, 0)
    >>> env.reward["codesize"]
    -1243

    :ivar spaces: Specifications of available reward spaces.
    :vartype spaces: Dict[str, RewardSpaceSpec]
    """

    def __init__(
        self,
        spaces: List[RewardSpace],
        get_cost: Optional[Callable[[str], None]] = None,
    ):
        self.spaces: Dict[str, RewardSpace] = {s.id: s for s in spaces}
        self.get_cost = get_cost

    def __getitem__(self, reward_space: str) -> float:
        """Request an observation from the given space.

        :param reward_space: The reward space to query.
        :return: A reward.
        :raises KeyError: If the requested reward space does not exist.
        """
        if not self.spaces:
            raise ValueError("No reward spaces")
        return self.spaces[reward_space].update(self.get_cost)

    def reset(self) -> None:
        for space in self.spaces:
            space.reset()

    def add_space(self, space: RewardSpace) -> None:
        if space.id in self.spaces:
            warnings.warn(f"Replacing existing reward space '{space.id}'")
        self.spaces[space.id] = space

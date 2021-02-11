# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Callable, Optional, Tuple

import numpy as np

from compiler_gym.service import observation_t
from compiler_gym.spaces.scalar import Scalar


class RewardSpace(Scalar):
    def __init__(
        self,
        id: str,
        default_value: float = False,
        min: Optional[float] = None,
        max: Optional[float] = None,
        default_negates_returns: bool = False,
        success_threshold: Optional[float] = None,
        deterministic: bool = False,
        platform_dependent: bool = True,
        cb: Callable[["CompilerEnv"], float] = None,  # noqa: F821
    ):
        super().__init__(
            min=-np.inf if min is None else min,
            max=np.inf if max is None else max,
            dtype=np.float64,
        )
        self.id = id
        self.default_value: float = default_value
        self.default_negates_returns: bool = default_negates_returns
        self.success_threshold = success_threshold
        self.deterministic = deterministic
        self.platform_dependent = platform_dependent
        self.cb = cb

    def reset(self) -> None:
        raise NotImplementedError("abstract class")

    def update(
        self, get_cost: Callable[[str], float], cost: observation_t = None
    ) -> float:
        return self.cb(None)

    def reward_on_error(self, episode_reward: float) -> float:
        """Return the reward value for an error condition.

        This method should be used to produce the reward value that should be
        used if the compiler service cannot be reached, e.g. because it has
        crashed or the connection has dropped.

        :param episode_reward: The current cumulative reward of an episode.
        :return: A reward.
        """
        if self.default_negates_returns:
            return self.default_value - episode_reward
        else:
            return self.default_value

    @property
    def range(self) -> Tuple[float, float]:
        return (self.min, self.max)

    def __repr__(self):
        return self.id


class CostFunctionRewardSpace(RewardSpace):
    def __init__(self, cost_function: str, init_cost_function: str, **kwargs):
        super().__init__(**kwargs)
        self.cost_function: str = cost_function
        self.init_cost_function: str = init_cost_function
        self.previous_cost = None

    def reset(self) -> None:
        self.previous_cost = None

    def update(
        self, get_cost: Callable[[str], float], cost: observation_t = None
    ) -> float:
        if cost is None:
            cost = get_cost(self.cost_function)
        if self.previous_cost is None:
            self.previous_cost = get_cost(self.init_cost_function)
        reward = self.previous_cost - cost
        self.previous_cost = cost
        return float(reward)


class NormalizedRewardSpace(CostFunctionRewardSpace):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cost_norm = None

    def reset(self) -> None:
        super().reset()
        self.cost_norm = None

    def update(
        self, get_cost: Callable[[str], float], cost: observation_t = None
    ) -> float:
        if self.cost_norm is None:
            self.cost_norm = self.get_cost_norm(get_cost)
        return super().update(get_cost, cost) / self.cost_norm

    def get_cost_norm(self, get_cost: Callable[[str], float]) -> float:
        return get_cost(self.init_cost_function)


class BaselineImprovementNormalizedRewardSpace(NormalizedRewardSpace):
    def __init__(self, baseline_cost_function: str, **kwargs):
        super().__init__(**kwargs)
        self.baseline_cost_function = baseline_cost_function

    def get_cost_norm(self, get_cost: Callable[[str], float]) -> float:
        init_cost = get_cost(self.init_cost_function)
        baseline_cost = get_cost(self.baseline_cost_function)
        return min(init_cost - baseline_cost, 1)

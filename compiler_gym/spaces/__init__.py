# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from compiler_gym.spaces.commandline import Commandline, CommandlineFlag
from compiler_gym.spaces.named_discrete import NamedDiscrete
from compiler_gym.spaces.reward import (
    BaselineImprovementNormalizedRewardSpace,
    CostFunctionRewardSpace,
    NormalizedRewardSpace,
    RewardSpace,
)
from compiler_gym.spaces.scalar import Scalar
from compiler_gym.spaces.sequence import Sequence

__all__ = [
    "Scalar",
    "Sequence",
    "NamedDiscrete",
    "Commandline",
    "CommandlineFlag",
    "RewardSpace",
    "CostFunctionRewardSpace",
    "NormalizedRewardSpace",
    "BaselineImprovementNormalizedRewardSpace",
]

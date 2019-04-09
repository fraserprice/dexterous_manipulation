from enum import Enum


class ActionSpace(Enum):
    CONTINUOUS = 0
    DISCRETE = 1


class MotorActionType(Enum):
    RATE = 0
    RATE_FORCE = 1


class LinkMode(Enum):
    FIXED = 0
    RANDOM = 1
    OPTIMAL = 2
    GENERAL_OPTIMAL = 3


class RewardType:
    DENSE = 0
    SPARSE = 1
    SHAPED = 2


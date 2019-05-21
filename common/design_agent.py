from abc import ABC, abstractmethod
from enum import Enum


class DesignAgentType(Enum):
    RL = 0
    BO = 1


class DesignAgent(ABC):
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.design_agent = None


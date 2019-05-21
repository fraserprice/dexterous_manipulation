import math
from abc import ABC, abstractmethod

from gym import Env

from common.utils import ModelStorage
import numpy as np


class NormalizedEnv(Env, ABC):
    def __init__(self):
        self.observation_max = []
        self.observation_min = []
        self.observation_n = 0
        self.observation_means = []
        self.observation_deviations = []
        self.ext_reward_history = []
        self.curiosity_loss_history = []

        self.curiosity_module = None
        self.steps = 0
        self.previous_observation = None

    @abstractmethod
    def step(self, action, render=False):
        pass

    def get_infos(self):
        cur = 0 if len(self.curiosity_loss_history) == 0 \
            else sum(self.curiosity_loss_history) / len(self.curiosity_loss_history)
        rew = 0 if len(self.ext_reward_history) == 0 else sum(self.ext_reward_history) / len(self.ext_reward_history)
        info = {
            "cur": cur,
            "rew": rew,
            "norm": {
                "observation_max": self.observation_max,
                "observation_min": self.observation_min,
                "observation_n": self.observation_n
            }
        }
        self.curiosity_loss_history = []
        self.ext_reward_history = []

        return info

    def normalize_observation(self, obs):
        obs = np.array(obs)
        normalized_obs = []
        for i, o in enumerate(obs):
            if len(self.observation_min) == 0:
                self.observation_max = obs.copy()
                self.observation_min = obs.copy()
                return [0] * len(obs)
            elif o > self.observation_max[i]:
                self.observation_max[i] = o
            elif o < self.observation_min[i]:
                self.observation_min[i] = o

            diff = self.observation_max[i] - self.observation_min[i]
            normalized_o = (o - self.observation_min[i]) / diff if diff != 0 else 0
            normalized_obs.append(normalized_o)
        return np.array(normalized_obs)

    # noinspection PyTypeChecker
    def standardize_observation(self, obs):
        self.observation_n += 1
        if self.observation_n == 1:
            self.observation_means = obs
            self.observation_deviations = [0] * len(obs)
            return self.observation_deviations
        standardized_obs = []
        for i, o in enumerate(obs):
            new_mean = self.observation_means[i] + (o - self.observation_means[i]) / self.observation_n
            self.observation_deviations[i] = math.sqrt(((self.observation_n - 1) * self.observation_deviations[i] ** 2
                                                        + (o - new_mean)
                                                        * (o - self.observation_means[i])) / self.observation_n)
            self.observation_means[i] = new_mean
            if self.observation_deviations[i] == 0:
                standardized_obs.append(0)
            else:
                standardized_obs.append((o - self.observation_means[i]) / self.observation_deviations[i])
        return np.array(standardized_obs)

    def load_normalization_info(self, model_name):
        model = ModelStorage().get_model(model_name)
        self.observation_max = model['normalization_info']['observation_max']
        self.observation_min = model['normalization_info']['observation_min']

    def add_curiosity_module(self, curiosity_module):
        self.curiosity_module = curiosity_module

    def get_curiosity_reward(self, reward, action, obs):
        curiosity_loss = None
        if self.curiosity_module is not None:
            if self.previous_observation is not None:
                curiosity_loss = 0.001 * self.curiosity_module.get_curiosity_loss(action, self.previous_observation,
                                                                                 obs).item()
                reward += curiosity_loss
            self.previous_observation = obs
        self.steps += 1

        ext_reward = reward if curiosity_loss is None else reward - curiosity_loss
        curiosity_loss = 0 if curiosity_loss is None else curiosity_loss
        self.curiosity_loss_history.append(curiosity_loss)
        self.ext_reward_history.append(ext_reward)

        return reward

    @abstractmethod
    def get_observation(self):
        return NotImplemented

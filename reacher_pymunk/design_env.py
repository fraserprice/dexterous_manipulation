import os

import pygame
from gym import Env, spaces
import numpy as np

from common.constants import LinkMode, MotorActionType
from common.normalized_env import NormalizedEnv
from common.ppo2_agent import PPO2Agent
from reacher_pymunk.env import Reacher

SEG_LENGTH_RANGE = (20, 200)
REPO_PATH = os.getcwd()


class ReacherDesign(NormalizedEnv):
    """
    Reacher env should be fixed type, reacher agent should be generalizing type (i.e. trained on random)
    """
    def __init__(self, reacher_agent_name, reacher_env=None, render=False):
        super().__init__()
        self.obs = None
        self.reward = 0
        self.design = []

        if reacher_env is None:
            self.reacher_env = Reacher(granularity=None, link_mode=LinkMode.FIXED, sparse_distance=0.08,
                                       motor_action_type=MotorActionType.RATE_FORCE, render=render)
        else:
            self.reacher_env = reacher_env

        model_path = f"{REPO_PATH}/reacher_pymunk/models/agent/{reacher_agent_name}"
        self.reacher_agent = PPO2Agent(self.reacher_env, subproc=False)
        self.reacher_agent.load_model(model_path, reacher_agent_name)

        self.observation_space = spaces.Box(low=0.,
                                            high=1.,
                                            shape=(2,),
                                            dtype=np.float32)
        self.action_space = spaces.Box(low=-1., high=1., shape=(2,), dtype=np.float32)

    # noinspection PyTypeChecker
    def step(self, action, render=False):
        action = [(a + 1) / 2 for a in action]
        self.design = [l * (SEG_LENGTH_RANGE[1] - SEG_LENGTH_RANGE[0]) + SEG_LENGTH_RANGE[0] for l in action]
        self.reacher_env.set_evaluation_design({
            'link_1': self.design[0],
            'link_2': self.design[1]
        })

        self.obs = self.reacher_env.get_observation()
        rewards = []
        while True:
            if render:
                self.reacher_env.render()
            action, _ = self.reacher_agent.model.predict(self.obs)
            self.obs, reward, done, _ = self.reacher_env.step(action)
            rewards.append(reward)
            if done:
                break
        self.reward = np.mean(np.array(rewards))
        self.ext_reward_history.append(self.reward)
        print(self.reacher_env.target)
        print(f"Design: {self.design} \t Reward: {self.reward}")

        return action, self.reward, True, {}

    def reset(self):
        self.reacher_env.reset()
        return np.array([0, 0])

    def render(self, mode='human'):
        print(f"Design: {self.design} \t Reward: {self.reward}")

    def get_observation(self):
        pass


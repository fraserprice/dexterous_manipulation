import os

from gym import spaces
import numpy as np

from common.constants import LinkMode, MotorActionType
from common.normalized_env import NormalizedEnv
from common.ppo2_agent import PPO2Agent
from grabber_pymunk.env import Grabber

SEG_LENGTH_RANGE = (50, 250)
REPO_PATH = os.getcwd()


class GrabberDesign(NormalizedEnv):
    """
    Reacher env should be fixed type, reacher agent should be generalizing type (i.e. trained on random)
    """
    def __init__(self, grabber_agent_name, grabber_env=None, render=False, square_target=False):
        super().__init__()
        self.obs = None
        self.reward = 0
        self.design = []

        if grabber_env is None:
            self.grabber_env = Grabber(granularity=None, link_mode=LinkMode.FIXED, square_target=square_target,
                                       motor_action_type=MotorActionType.RATE_FORCE, render=render)
        else:
            self.grabber_env = grabber_env

        model_path = f"{REPO_PATH}/grabber_pymunk/models/agent/{grabber_agent_name}"
        self.reacher_agent = PPO2Agent(self.grabber_env, subproc=False)
        self.reacher_agent.load_model(model_path, grabber_agent_name)

        self.observation_space = spaces.Box(low=0.,
                                            high=1.,
                                            shape=(4,),
                                            dtype=np.float32)
        self.action_space = spaces.Box(low=-1., high=1., shape=(4,), dtype=np.float32)

    # noinspection PyTypeChecker
    def step(self, action, render=False):
        action = [(a + 1) / 2 for a in action]
        self.design = [l * (SEG_LENGTH_RANGE[1] - SEG_LENGTH_RANGE[0]) + SEG_LENGTH_RANGE[0] for l in action]
        self.grabber_env.set_evaluation_design({
            'l11': self.design[0],
            'l12': self.design[1],
            'l21': self.design[2],
            'l22': self.design[3]
        })

        self.obs = self.grabber_env.get_observation()
        rewards = []
        while True:
            if render:
                self.grabber_env.render()
            action, _ = self.reacher_agent.model.predict(self.obs)
            self.obs, reward, done, _ = self.grabber_env.step(action)
            rewards.append(reward)
            if done:
                break
        self.reward = np.mean(np.array(rewards))
        self.ext_reward_history.append(self.reward)
        print(f"Design: {self.design} \t Reward: {self.reward}")

        return action, self.reward, True, {}

    def reset(self):
        self.grabber_env.reset()
        return np.array([0, 0, 0, 0])

    def render(self, mode='human'):
        print(f"Design: {self.design} \t Reward: {self.reward}")

    def get_observation(self):
        pass

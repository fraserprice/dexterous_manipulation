import time

import numpy as np

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines import PPO2
from vrep_grabber.env import VrepGrabber

DEFAULT_VREP_PORT = 19997
VREP_PORTS = range(19998, 20014)


def create_grabber_env(scene_path, port, action_granularity=None):
    print(f"Creating env on port {port}")
    return VrepGrabber(scene_path, port, action_granularity=action_granularity)


class GrabberAgent:
    def __init__(self, scene_path, action_granularity=None, subproc=True):
        if subproc:
            # TODO: Find way to make this look better lol... For loop fails
            creation_functions = [
                lambda: create_grabber_env(scene_path, 19998, action_granularity),
                lambda: create_grabber_env(scene_path, 19999, action_granularity),
                lambda: create_grabber_env(scene_path, 20000, action_granularity),
                lambda: create_grabber_env(scene_path, 20001, action_granularity),
                lambda: create_grabber_env(scene_path, 20002, action_granularity),
                lambda: create_grabber_env(scene_path, 20003, action_granularity),
                lambda: create_grabber_env(scene_path, 20004, action_granularity),
                lambda: create_grabber_env(scene_path, 20005, action_granularity),
                lambda: create_grabber_env(scene_path, 20006, action_granularity),
                lambda: create_grabber_env(scene_path, 20007, action_granularity),
                lambda: create_grabber_env(scene_path, 20008, action_granularity),
                lambda: create_grabber_env(scene_path, 20009, action_granularity),
                lambda: create_grabber_env(scene_path, 20010, action_granularity),
                lambda: create_grabber_env(scene_path, 20011, action_granularity),
                lambda: create_grabber_env(scene_path, 20012, action_granularity),
                lambda: create_grabber_env(scene_path, 20013, action_granularity)
            ]
            self.env = SubprocVecEnv(creation_functions)
        else:
            self.base_env = VrepGrabber(scene_path, DEFAULT_VREP_PORT, action_granularity=action_granularity)
            self.env = DummyVecEnv([lambda: self.base_env])
        self.model = None

    def load_model(self, path):
        self.model = PPO2.load(path, self.env)

    def save_model(self, path="ppo2_simple_robot"):
        if self.model is None:
            raise AssertionError("Model does not exist- cannot be saved.")
        self.model.save(path)

    def new_model(self, policy=MlpPolicy, gamma=0.99):
        self.model = PPO2(policy, self.env, verbose=1, gamma=gamma)

    def learn(self, timesteps, learning_handler, checkpoint_interval=1000, path=None):
        if self.model is None:
            self.new_model()
        if checkpoint_interval is not None:
            for checkpoint in range(int(timesteps/checkpoint_interval)):
                print(f"Checkpointing model. Total timesteps: {checkpoint * checkpoint_interval}")
                cb = learning_handler.get_learn_callback(checkpoint * checkpoint_interval)
                self.model.learn(total_timesteps=checkpoint_interval, callback=cb)
                self.save_model(path)
        else:
            cb = learning_handler.get_learn_callback()
            self.model.learn(total_timesteps=timesteps, callback=cb)
            self.save_model(path)

    def demo(self, timestep_sleep=0.2):
        obs = self.base_env.reset()
        while True:
            action, _states = self.model.predict(obs)
            obs, _, done, info = self.base_env.step(action)
            self.base_env.render()
            time.sleep(timestep_sleep)
            if done:
                obs = self.base_env.reset()

    def validate(self, n_episodes):
        obs = self.base_env.reset()
        ep_histories = None
        for i in range(n_episodes):
            ep_history = []
            while True:
                action, _states = self.model.predict(obs)
                obs, reward, done, info = self.base_env.step(action)
                ep_history.append(info['distance'])
                if done:
                    if ep_histories is None:
                        ep_histories = np.array([ep_history])
                    else:
                        ep_histories = np.concatenate((ep_histories, [ep_history]))
                    obs = self.base_env.reset()
                    break
        return ep_histories

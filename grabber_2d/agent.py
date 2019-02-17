import multiprocessing
import time

import numpy as np
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv


class Grabber2DAgent:
    def __init__(self, grabber2d_env, subproc=True):
        self.base_env = grabber2d_env
        if subproc:
            self.env = SubprocVecEnv([lambda: grabber2d_env for _ in range(multiprocessing.cpu_count())])
        else:
            self.env = DummyVecEnv([lambda: self.base_env])
        self.model = None

    def load_model(self, path):
        self.model = PPO2.load(path, self.env)

    def save_model(self, path="ppo2_simple_robot"):
        if self.model is None:
            raise AssertionError("Model does not exist- cannot be saved.")
        self.model.save(path)

    def new_model(self, policy=MlpPolicy, gamma=0.99):
        self.model = PPO2(policy, self.env, verbose=1, gamma=gamma, n_steps=512)

    def learn(self, timesteps, learning_handler, checkpoint_interval=1000, path=None, learning_rate=0.00025,
              curiosity_path=None):
        self.model.learning_rate = learning_rate
        if self.model is None:
            self.new_model()
        if checkpoint_interval is not None:
            for checkpoint in range(int(timesteps / checkpoint_interval)):
                print(f"Checkpointing model. Total timesteps: {checkpoint * checkpoint_interval}")
                if curiosity_path is None:
                    cb = learning_handler.get_learn_callback(checkpoint * checkpoint_interval)
                else:
                    cb = learning_handler.get_curiosity_learn_callback(checkpoint * checkpoint_interval)
                self.model.learn(total_timesteps=checkpoint_interval, callback=cb)
                self.save_model(path)
                if curiosity_path is not None:
                    self.base_env.curiosity_module.save_forward(curiosity_path)
        else:
            if curiosity_path is None:
                cb = learning_handler.get_learn_callback()
            else:
                cb = learning_handler.get_curiosity_learn_callback()
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

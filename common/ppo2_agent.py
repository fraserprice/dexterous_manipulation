import matplotlib
import multiprocessing
import time
import numpy as np

from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv


class PPO2Agent:
    def __init__(self, base_env, subproc=True, envs=64, ):
        self.base_env = base_env
        self.subproc = subproc
        if subproc:
            envs = multiprocessing.cpu_count() if envs is None else envs
            self.env = SubprocVecEnv([lambda: base_env for _ in range(envs)])
        else:
            self.env = DummyVecEnv([lambda: self.base_env])
        self.model = None

    def load_model(self, path, model_name):
        self.model = PPO2.load(path, self.env)
        self.env.env_method("load_normalization_info", model_name=model_name)

    def save_model(self, path="ppo2_simple_robot"):
        if self.model is None:
            raise AssertionError("Model does not exist- cannot be saved.")
        self.model.save(path)

    def new_model(self, policy=MlpPolicy, gamma=0.99, batch_size=128):
        self.model = PPO2(policy, self.env, verbose=1, gamma=gamma, n_steps=batch_size)

    def learn(self, timesteps, learning_handler, checkpoint_interval=1000, path=None, learning_rate=0.00025,
              curiosity_path=None, batch_size=128):
        curiosity = curiosity_path is not None
        self.model.learning_rate = learning_rate
        if self.model is None:
            self.new_model(batch_size=batch_size)
        if checkpoint_interval is not None:
            for checkpoint in range(int(timesteps / checkpoint_interval)):
                cb = learning_handler.get_learn_callback(checkpoint * checkpoint_interval, curiosity=curiosity,
                                                         subproc=self.subproc, batch_size=batch_size)
                self.model.learn(total_timesteps=checkpoint_interval, callback=cb, reset_num_timesteps=False)
                self.save_model(path)
                if curiosity:
                    self.base_env.curiosity_module.save_forward(curiosity_path)

                matplotlib.use('Agg')
                m = learning_handler.model_storage.get_model(learning_handler.model_name)
                learning_handler.save_plot(m['realtime_data']['plot_path'], real_time=True, curiosity=curiosity)
                learning_handler.save_plot(m['timestep_data']['plot_path'], real_time=False, curiosity=curiosity)
        else:
            cb = learning_handler.get_learn_callback(curiosity=curiosity, subproc=self.subproc)
            self.model.learn(total_timesteps=timesteps, callback=cb, reset_num_timesteps=False)
            self.save_model(path)

    def demo(self, timestep_sleep=0):
        obs = self.base_env.reset()
        while True:
            action, _states = self.model.predict(obs)
            obs, _, done, info = self.base_env.step(action, render=True)
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

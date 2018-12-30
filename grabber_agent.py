import matplotlib
import time

import numpy as np
from scipy.stats import binned_statistic

from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, LstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, VecEnv, DummyVecEnv
from stable_baselines import PPO2
from grabber_env import VrepGrabber


class GrabberAgent:
    def __init__(self, grabber_env):
        self.base_env = grabber_env
        self.env = DummyVecEnv([lambda: grabber_env for _ in range(1)])  # multiprocessing.cpu_count())])
        self.model = None

    def load_model(self, path):
        self.model = PPO2.load(path, self.env)

    def save_model(self, path="ppo2_simple_robot"):
        if self.model is None:
            raise AssertionError("Model does not exist- cannot be saved.")
        self.model.save(path)

    def new_model(self, policy=MlpPolicy, gamma=0.99):
        self.model = PPO2(policy, self.env, verbose=1, gamma=gamma)

    def learn(self, timesteps, callback=None, checkpoint_interval=1000, path=None):
        if self.model is None:
            self.new_model()
        for checkpoint in range(int(timesteps/checkpoint_interval)):
            self.model.learn(total_timesteps=checkpoint_interval, callback=callback)
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


class LossPlotter:
    def __init__(self, max_points=200):
        from matplotlib import pyplot as plt
        self.plt = plt
        self.timesteps = []
        self.rewards = []
        self.aggregated_timesteps = []
        self.aggregated_rewards = []
        fig = plt.figure()
        self.ax = fig.add_subplot()
        self.max_points = max_points
        self.bin_size = 1
        self.t_start = None
        self.last_checkpoint = None

    def add_point(self, distance):
        t = time.time()
        if self.t_start is None:
            self.t_start = t
        self.timesteps.append(t - self.t_start)
        self.rewards.append(distance)
        return t

    def plot(self):
        self.plt.xlabel("Time/s")
        self.plt.ylabel("Average reward")
        if len(self.timesteps) <= self.max_points:
            self.plt.plot(self.timesteps, self.rewards)
        else:
            self.aggregated_rewards, self.aggregated_timesteps, _ = binned_statistic(self.timesteps, self.rewards,
                                                                                     bins=self.max_points)
            # self.aggregated_variance, _, _ = binned_statistic(self.timesteps, self.rewards, bins=self.max_points,
            #                                                   statistic=np.var)
            # print(self.aggregated_variance)
            # lower_var, upper_var = [], []
            # for i, var in enumerate(self.aggregated_variance):
            #     lower_var.append(self.aggregated_rewards[i] - var)
            #     upper_var.append(self.aggregated_rewards[i] + var)
            self.plt.cla()
            # self.plt.fill_between(self.aggregated_timesteps[1:], lower_var, upper_var)
            self.plt.plot(self.aggregated_timesteps[1:], self.aggregated_rewards)
        # self.plt.pause(0.000001)

    def get_plot_callback(self, checkpoint_interval=1000, filename=None, verbose=False):
        def f(inp1, _):

            mean_reward = None
            if 'true_reward' in inp1:
                mean_reward = np.array(inp1['true_reward']).mean()
            if 'info' in inp1:
                print(inp1['info'])
            if mean_reward is not None:
                t = self.add_point(mean_reward)
                elapsed = 0 if self.t_start is None else time.time() - self.t_start
                if 1 <= checkpoint_interval <= elapsed - (0 if self.last_checkpoint is None else self.last_checkpoint):
                    self.last_checkpoint = elapsed
                    if not verbose:
                        matplotlib.use('Agg')
                    self.save(filename)
                if verbose:
                    self.plot()
                    self.plt.pause(0.000001)
                return t
            return None

        return f

    def save(self, filename):
        self.plot()
        self.plt.savefig(filename)


def plot_validation(ep_histories, filename=None):
    from matplotlib import pyplot as plt
    ep_length = len(ep_histories[0])

    xs = range(ep_length)
    means = []
    stds = []
    for timestep in range(ep_length):
        rewards = []
        for episode_index in range(len(ep_histories)):
            rewards.append(ep_histories[episode_index][timestep])
        means.append(np.mean(rewards))
        stds.append(np.std(rewards))
    lower = np.array(means) - stds
    upper = np.array(means) + stds
    plt.plot(xs, means)
    plt.fill_between(xs, lower, upper, facecolor='#FF9848')
    if filename is not None:
        plt.savefig(filename)
    plt.show()


class CustomMlpPolicy(MlpPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, **_kwargs):
        super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, **_kwargs)


class CustomMlpLstmPolicy(MlpLstmPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, **_kwargs):
        super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, layers=[256, 256, 256],
                         **_kwargs)


if __name__ == "__main__":
    PATH_3DOF = "/Users/fraser/Documents/University/Fourth Year/Dexterous Manipulation/VREP Repo/3D Model/3dof_grabber.ttt"
    PATH_6DOF = "/Users/fraser/Documents/University/Fourth Year/Dexterous Manipulation/VREP Repo/3D Model/6dof_grabber.ttt"

    name = "64_64_100k_6dof_no_pen"
    fig_name = f"figures/{name}"
    model_name = f"models/{name}"

    base_env = VrepGrabber(PATH_6DOF, headless=True)
    ppo_agent = GrabberAgent(base_env)

    ppo_agent.new_model(policy=CustomMlpPolicy, gamma=0.99)
    # ppo_agent.load_model("models/" + "256_256_256_500k")
    loss_plotter = LossPlotter(max_points=150)
    ppo_agent.learn(100000,
                    callback=loss_plotter.get_plot_callback(verbose=False, filename=fig_name, checkpoint_interval=60),
                    checkpoint_interval=1000,
                    path=model_name)
    loss_plotter.save(fig_name)
    ppo_agent.save_model(model_name)

    # ppo_agent.load_model(model_name)
    # ppo_agent.demo(timestep_sleep=0.1)

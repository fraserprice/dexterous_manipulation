import matplotlib
import time

import numpy as np
from pymongo import MongoClient
from scipy.stats import binned_statistic

DB_NAME = 'dexterous_manipulation'


class Storage:
    def __init__(self):
        self.client = MongoClient()
        self.db = self.client[DB_NAME]


class ModelStorage(Storage):
    def __init__(self):
        super().__init__()
        self.models = self.db.models

    def new_model(self, name, description, model_path, r_plot_path, t_plot_path):
        data = {
            'name': name,
            'description': description,
            'model_path': model_path,
            'realtime_data': {
                'plot_path': r_plot_path,
                'elapsed': [],
                'rewards': []
            },
            'timestep_data': {
                'plot_path': t_plot_path,
                'timesteps': [],
                'rewards': []
            }
        }
        if self.get_model(name) is not None:
            self.remove_model(name)
        return self.models.insert_one(data).inserted_id

    def get_model(self, name):
        return self.models.find_one({'name': name})

    def get_all_models(self):
        return self.models.find()

    def add_realtime_point(self, name, elapsed, reward):
        update = {
            'realtime_data.elapsed': int(elapsed),
            'realtime_data.rewards': float(reward)
        }
        self.models.update_one({"name": name}, {"$push": update}, upsert=False)

    def add_timestep_point(self, name, timestep, reward):
        update = {
            'timestep_data.timesteps': int(timestep),
            'timestep_data.rewards': float(reward)
        }
        self.models.update_one({"name": name}, {"$push": update}, upsert=False)

    def remove_model(self, name):
        self.models.delete_many({"name": name})


class LearningHandler:
    def __init__(self, model_name, model_storage, max_points=200):
        self.model_storage = model_storage
        self.model_name = model_name
        self.max_points = max_points
        self.r_start = None
        self.t_start = None
        self.last_checkpoint = 0

    def add_realtime_point(self, reward):
        t = time.time()
        if self.r_start is None:
            m = self.model_storage.get_model(self.model_name)
            if m is not None and len(m['realtime_data']['elapsed']) > 0:
                last_time = m['realtime_data']['elapsed'][-1]
            else:
                last_time = 0
            self.r_start = t - last_time

        self.model_storage.add_realtime_point(self.model_name, t - self.r_start, reward)

    def add_timestep_point(self, timestep, reward):
        if self.t_start is None:
            m = self.model_storage.get_model(self.model_name)
            if m is not None and len(m['timestep_data']['timesteps']) > 0:
                self.t_start = m['timestep_data']['timesteps'][-1]
            else:
                self.t_start = 0

        self.model_storage.add_timestep_point(self.model_name, timestep + self.t_start, reward)

    def save_plot(self, filename, real_time=True):
        from matplotlib import pyplot as plt

        fig = plt.figure()
        fig.add_subplot()
        plt.xlabel("Time/s" if real_time else "Timesteps")
        plt.ylabel("Average reward")

        m = self.model_storage.get_model(self.model_name)
        data = m['realtime_data' if real_time else 'timestep_data']
        x, y = data['elapsed' if real_time else 'timesteps'], data['rewards']
        if len(x) <= self.max_points:
            plt.plot(x, y)
        else:
            aggregated_rewards, aggregated_timesteps, _ = binned_statistic(x, y, bins=self.max_points)
            # plt.cla()
            plt.plot(aggregated_timesteps[1:], aggregated_rewards)
        plt.savefig(filename)
        plt.close()

    def get_learn_callback(self, elapsed_timesteps=0):
        def f(info, _):
            reward = None
            timestep = info['timestep'] if 'timestep' in info else None
            if 'true_reward' in info:
                reward = np.array(info['true_reward']).mean()
            if reward is not None and timestep is not None:
                self.add_realtime_point(reward)
                self.add_timestep_point(timestep * 128 + elapsed_timesteps, reward)

                matplotlib.use('Agg')
                m = self.model_storage.get_model(self.model_name)
                self.save_plot(m['realtime_data']['plot_path'], real_time=True)
                self.save_plot(m['timestep_data']['plot_path'], real_time=False)

        return f


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


if __name__ == "__main__":
    ms = ModelStorage()
    print(ms.get_model("128_128_2m-4m_3dof_02height_no-pen_15gran"))

import multiprocessing

import matplotlib
import time

import numpy as np
from pymongo import MongoClient
from scipy.stats import binned_statistic
from stable_baselines.common.vec_env import SubprocVecEnv
from sshtunnel import SSHTunnelForwarder

DB_NAME = 'dexterous_manipulation'


class Storage:
    def __init__(self):
        # MONGO_HOST = "REMOTE_IP_ADDRESS"
        # MONGO_DB = "DATABASE_NAME"
        # MONGO_USER = "LOGIN"
        # MONGO_PASS = "PASSWORD"
        #
        # server = SSHTunnelForwarder(
        #     MONGO_HOST,
        #     ssh_username=MONGO_USER,
        #     ssh_password=MONGO_PASS,
        #     remote_bind_address=('127.0.0.1', 27017)
        # )
        #
        # server.start()
        #
        # self.client = MongoClient('127.0.0.1', server.local_bind_port)
        self.client = MongoClient()
        self.db = self.client[DB_NAME]


class ModelStorage(Storage):
    def __init__(self):
        super().__init__()
        self.models = self.db.models

    def new_model(self, name, description, model_path, r_plot_path, t_plot_path, curiosity_fwd_path=""):
        data = {
            'name': name,
            'description': description,
            'model_path': model_path,
            'curiosity_forward_path': curiosity_fwd_path,
            'realtime_data': {
                'plot_path': r_plot_path,
                'elapsed': [],
                'rewards': [],
                'curiosity': []
            },
            'timestep_data': {
                'plot_path': t_plot_path,
                'timesteps': [],
                'rewards': [],
                'curiosity': []
            }
        }
        if self.get_model(name) is not None:
            self.remove_model(name)
        return self.models.insert_one(data).inserted_id

    def get_model(self, name):
        return self.models.find_one({'name': name})

    def get_all_models(self):
        return self.models.find()

    def add_realtime_point(self, name, elapsed, reward, curiosity=None):
        update = {
            'realtime_data.elapsed': int(elapsed),
            'realtime_data.rewards': float(reward),
            'realtime_data.curiosity': 0 if curiosity is None else curiosity
        }
        self.models.update_one({"name": name}, {"$push": update}, upsert=False)

    def add_timestep_point(self, name, timestep, reward, curiosity=None):
        update = {
            'timestep_data.timesteps': int(timestep),
            'timestep_data.rewards': float(reward),
            'timestep_data.curiosity': 0 if curiosity is None else curiosity
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

    def add_realtime_point(self, reward, curiosity=None):
        t = time.time()
        if self.r_start is None:
            m = self.model_storage.get_model(self.model_name)
            if m is not None and len(m['realtime_data']['elapsed']) > 0:
                last_time = m['realtime_data']['elapsed'][-1]
            else:
                last_time = 0
            self.r_start = t - last_time

        self.model_storage.add_realtime_point(self.model_name, t - self.r_start, reward, curiosity)

    def add_timestep_point(self, timestep, reward, curiosity=None):
        if self.t_start is None:
            m = self.model_storage.get_model(self.model_name)
            if m is not None and len(m['timestep_data']['timesteps']) > 0:
                self.t_start = m['timestep_data']['timesteps'][-1]
            else:
                self.t_start = 0

        self.model_storage.add_timestep_point(self.model_name, timestep + self.t_start, reward, curiosity)

    def save_plot(self, filename, real_time=True, curiosity=True):
        from matplotlib import pyplot as plt

        fig, ax1 = plt.subplots()

        c1 = 'tab:red'
        ax1.set_xlabel('Time/s' if real_time else "Timesteps")
        ax1.set_ylabel('Average Extrinsic Reward', color=c1)
        ax1.tick_params(axis='y', labelcolor=c1)

        ax2 = None
        c2 = None
        if curiosity:
            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            c2 = 'tab:blue'
            ax2.set_ylabel('Curiosity Loss', color=c2)  # we already handled the x-label with ax1
            ax2.tick_params(axis='y', labelcolor=c2)

        m = self.model_storage.get_model(self.model_name)
        data = m['realtime_data' if real_time else 'timestep_data']
        x, y_rew, y_cur = data['elapsed' if real_time else 'timesteps'], data['rewards'], data['curiosity']
        if len(x) <= self.max_points:
            ax1.plot(x, y_rew, color=c1)
            if curiosity:
                ax2.plot(x, y_cur, color=c2)
        else:
            aggregated_rewards, aggregated_timesteps, _ = binned_statistic(x, y_rew, bins=self.max_points)
            aggregated_curiosity, _, _ = binned_statistic(x, y_cur, bins=self.max_points)
            # plt.cla()
            ax1.plot(aggregated_timesteps[1:], aggregated_rewards, color=c1)
            if curiosity:
                ax2.plot(aggregated_timesteps[1:], aggregated_curiosity, color=c2)
        fig.tight_layout()
        plt.savefig(filename)
        plt.close()

    def get_learn_callback(self, elapsed_timesteps=0, batch_size=128, subproc=True, curiosity=True):
        def f(info, _):
            timestep = info['update'] * batch_size * (multiprocessing.cpu_count() if subproc else 1)
            info = info['runner'].env.env_method('get_infos')
            curiosity_loss = np.array([c for c, _ in info]).mean()
            ext_reward = np.array([p for _, p in info]).mean()
            if ext_reward is not None and timestep is not None and curiosity_loss is not None:
                if curiosity:
                    self.add_realtime_point(ext_reward, curiosity_loss)
                    self.add_timestep_point(timestep + elapsed_timesteps, ext_reward, curiosity_loss)
                else:
                    self.add_realtime_point(ext_reward)
                    self.add_timestep_point(timestep + elapsed_timesteps, ext_reward)

                matplotlib.use('Agg')
                m = self.model_storage.get_model(self.model_name)
                self.save_plot(m['realtime_data']['plot_path'], real_time=True, curiosity=curiosity)
                self.save_plot(m['timestep_data']['plot_path'], real_time=False, curiosity=curiosity)

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

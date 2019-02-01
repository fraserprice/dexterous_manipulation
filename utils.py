import time

import numpy as np
from bson import ObjectId
from pymongo import MongoClient

DB_NAME = 'dexterous_manipulation'


class Storage:
    def __init__(self):
        self.client = MongoClient()
        self.db = self.client[DB_NAME]


class ModelStorage(Storage):
    def __init__(self):
        super().__init__()
        self.models = self.db.models

    def new_model(self, name, description, model_path, plot_path):
        data = {
            'name': name,
            'description': description,
            'model_path': model_path,
            'plot_path': plot_path,
            'realtime_data': {
                'elapsed': [],
                'rewards': []
            },
            'timestep_data': {
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

    def add_realtime_points(self, name, elapsed, rewards):
        update = {
            'realtime_data.elapsed': {"$each": elapsed},
            'realtime_data.rewards': {"$each": rewards}
        }
        self.models.update_one({"name": name}, {"$push": update}, upsert=False)

    def add_timestep_points(self, name, timesteps, rewards):
        update = {
            'timestep_data.timesteps': {"$each": timesteps},
            'timestep_data.rewards': {"$each": rewards}
        }
        self.models.update_one({"name": name}, {"$push": update}, upsert=False)

    def remove_model(self, name):
        self.models.delete_many({"name": name})


class LearningHandler:
    def __init__(self, model_storage):
        self.model_storage = model_storage
        self.t_start = None

        self.timesteps = []
        self.timestep_rewards = []

        self.aggregated_timesteps = []
        self.aggregated_rewards = []
        fig = plt.figure()
        self.ax = fig.add_subplot()
        self.max_points = max_points
        self.bin_size = 1
        self.t_start = None
        self.last_checkpoint = None

    def get_learn_callback(self, name, checkpoint_interval=1000):
        def f(inp1, _):
            mean_reward = None
            if 'true_reward' in inp1:
                mean_reward = np.array(inp1['true_reward']).mean()

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


if __name__ == "__main__":
    ms = ModelStorage()

from stable_baselines.common.policies import MlpPolicy

from common.ppo2_agent import PPO2Agent
from common.curiosity_module import CuriosityModule
from common.design_optimizer import DesignOptimizer
from common.utils import LearningHandler, ModelStorage
import os

REPO_PATH = os.getcwd()  #""/Users/fraser/Documents/University/Fourth Year/Dexterous Manipulation/Dexterous Manipulation"
CURIOSITY_HIDDEN = [32, 32, 32, 32]
POLICY_HIDDEN = [32, 32, 32, 32]


def custom_policy(hidden_size):
    if hidden_size is None:
        hidden_size = POLICY_HIDDEN

    class CustomMlpPolicy(MlpPolicy):
        def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, **_kwargs):
            super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, layers=hidden_size, **_kwargs)

    return CustomMlpPolicy


class BaseRunner:
    def __init__(self):
        self.agent = None
        self.bo_results = None
        self.env = None
        self.evaluator_class = None
        self.repo_path = REPO_PATH
        self.env_name = ""

    def load_agent(self, name, subproc=True):
        model_path = f"{self.repo_path}/{self.env_name}/models/agent/{name}"
        self.agent = PPO2Agent(self.env, subproc=subproc)
        self.agent.load_model(model_path)

    def train(self,
              model_name,
              description,
              timesteps=500000000,
              model_storage=ModelStorage(),
              policy_hidden=None,
              curiosity_hidden=None,
              new_model=False,
              starting_model_path=None,
              checkpoint_interval=None,
              learning_rate=0.00025,
              gamma=0.99,
              curiosity=False,
              subproc=True,
              batch_size=256):

        r_plot_path = f"{self.repo_path}/{self.env_name}/figures/realtime/{model_name}"
        t_plot_path = f"{self.repo_path}/{self.env_name}/figures/timesteps/{model_name}"
        model_path = f"{self.repo_path}/{self.env_name}/models/agent/{model_name}"
        curiosity_path = f"{self.repo_path}/{self.env_name}/models/curiosity/{model_name}" if curiosity else None

        if curiosity:
            if curiosity_hidden is None:
                curiosity_hidden = CURIOSITY_HIDDEN
            action_space = self.env.action_space.shape[0]
            state_space = self.env.observation_space.shape[0]
            curiosity_module = CuriosityModule(action_space + state_space, state_space, curiosity_hidden)
            if not new_model:
                curiosity_module.load_forward(curiosity_path)
            self.env.add_curiosity_module(curiosity_module)

        self.agent = PPO2Agent(self.env, subproc=subproc)

        if new_model:
            model_storage.new_model(model_name, description, model_path, r_plot_path, t_plot_path)
            if starting_model_path is None:
                self.agent.new_model(policy=custom_policy(policy_hidden), gamma=gamma, batch_size=batch_size)
            else:
                self.agent.load_model(starting_model_path)
        else:
            self.agent.load_model(model_path if starting_model_path is None else starting_model_path)

        learning_handler = LearningHandler(model_name, model_storage, max_points=150)
        self.agent.learn(timesteps,
                         learning_handler,
                         checkpoint_interval=checkpoint_interval,
                         path=model_path, learning_rate=learning_rate,
                         curiosity_path=curiosity_path,
                         batch_size=batch_size)

    def bo_search(self, parameter_bounds, n_iter=100, random_searches=10, random_state=1):
        """
        Assumes self.agent is a generalizing agent trained using train_rl

        :return:
        """
        agent_evaluator = self.evaluator_class(self.agent)
        optimizer = DesignOptimizer(agent_evaluator)
        self.bo_results = optimizer.search(parameter_bounds, n_iter, random_searches, random_state)

    def demo(self, model_name, design=None, timestep_sleep=0):
        model_path = f"{self.repo_path}/{self.env_name}/models/agent/{model_name}.pkl"
        self.env.do_render = True
        agent = PPO2Agent(self.env)
        if design is not None:
            self.env.evaluation_mode = True
            self.env.set_evaluation_design(design)
        agent.load_model(model_path)
        agent.demo(timestep_sleep)

    def train_rl_split(self):
        pass

    def bo_finetune(self):
        """
        Assumes self.agent is a generalizing agent trained using train_rl and that bo_search has been performed

        :return:
        """

        pass

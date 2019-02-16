from bayes_opt import BayesianOptimization
from stable_baselines.common.policies import MlpLstmPolicy, MlpPolicy
import numpy as np

from reacher.env import LinkMode, RewardMode
from reacher.run import train
from utils import ModelStorage

REPO_PATH = "/Users/fraser/Documents/University/Fourth Year/Dexterous Manipulation/Dexterous Manipulation"


class CustomMlpPolicy(MlpPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, **_kwargs):
        super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, layers=[32, 32], **_kwargs)


class CustomMlpLstmPolicy(MlpLstmPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, **_kwargs):
        super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, layers=[256, 256, 256],
                         **_kwargs)



def evaluate_fixed_agent(agent, link_1, link_2, n_episodes=200, render=False):
    rewards = []
    # variances = []
    agent.base_env.fixed_lengths = [link_1, link_2, 0]
    agent.base_env.link_mode = LinkMode.FIXED
    obs = agent.base_env.reset()
    for i in range(n_episodes):
        ep_rewards = []
        while True:
            action, _states = agent.model.predict(obs)
            obs, reward, done, _ = agent.base_env.step(action)
            if render:
                agent.base_env.render()
            ep_rewards.append(reward)
            if done:
                rewards.append(np.mean(ep_rewards))
                # variances.append(np.std(ep_rewards))
                obs = agent.base_env.reset()
                break
    return np.mean(rewards)


class AgentEvaluator:
    def __init__(self, agent):
        self.agent = agent

    def evaluate_fixed_agent(self, link_1, link_2, n_episodes=500, render=False):
        return evaluate_fixed_agent(self.agent, link_1, link_2, n_episodes, render)


class AdaptiveAgentOptimizer:
    def __init__(self, starting_agent, name="Adaptive Agent", desc="xd", temp_path="copied_model"):
        self.current_agent = starting_agent
        self.current_link_lengths = self.current_agent.base_env.fixed_lengths
        self.learned_agents = [(self.current_link_lengths, self.current_agent)]
        self.name = name
        self.desc = desc
        self.temp_path = "copied_model"
        self.current_agent.save_model(temp_path)

    def adapt_agent(self, link_1, link_2, n_timesteps=5000000):
        print(f"Adapting and training new agent with lengths [{link_1}, {link_2}]")
        self.current_link_lengths = [link_1, link_2, 0]

        closest_dist = None
        closest_agent = None
        for links, agent in self.learned_agents:
            dist = np.linalg.norm((self.current_link_lengths, links))
            if closest_dist is None or dist < closest_dist:
                closest_dist = dist
                closest_agent = agent
        print(f"Adapting from model with lengths [{closest_agent.base_env.link_lengths[0]}, {closest_agent.base_env.link_lengths[1]}]")
        closest_agent.save_model(self.temp_path)

        storage = ModelStorage()
        gran = self.current_agent.base_env.action_granularity
        model_name = f"{self.name}-{str(link_1).replace('.', '_')}-{str(link_2).replace('.', '_')}"

        print(f"Training {model_name}")
        self.current_agent = train(model_name, self.desc, n_timesteps, storage, new_model=True,
                                   action_granularity=gran, checkpoint_interval=10000, link_mode=LinkMode.FIXED,
                                   reward_mode=RewardMode.LINEAR, learning_rate=0.0001, gamma=0.95,
                                   link_lengths=self.current_link_lengths, starting_model_path=self.temp_path)
        self.learned_agents.append((self.current_link_lengths, self.current_agent))

    def train_and_eval_new_agent(self, link_1, link_2, n_episodes=500):
        self.adapt_agent(link_1, link_2)
        print(f"Evaluating...")
        return evaluate_fixed_agent(self.current_agent, self.current_link_lengths[0], self.current_link_lengths[1],
                                    n_episodes=n_episodes)

    def search(self, parameter_bounds=None, n_iter=10, random_searches=0, random_state=1):
        if parameter_bounds is None:
            parameter_bounds = {'link_1': (0., 0.3), 'link_2': (0., 0.3)}
        optimizer = BayesianOptimization(f=self.train_and_eval_new_agent,
                                         pbounds=parameter_bounds,
                                         random_state=random_state,
                                         verbose=2)
        optimizer.maximize(n_iter=n_iter, init_points=random_searches)

        return optimizer.max, self.learned_agents


class MultitaskAgentOptimizer:
    def __init__(self, multitask_agent):
        self.evaluator = AgentEvaluator(multitask_agent)

    def search(self, parameter_bounds=None, n_iter=100, random_searches=10, random_state=1):
        if parameter_bounds is None:
            parameter_bounds = {'link_1': (0., 0.3), 'link_2': (0., 0.3)}
        optimizer = BayesianOptimization(f=self.evaluator.evaluate_fixed_agent,
                                         pbounds=parameter_bounds,
                                         random_state=random_state,
                                         verbose=2)
        optimizer.maximize(n_iter=n_iter, init_points=random_searches)

        return optimizer.max

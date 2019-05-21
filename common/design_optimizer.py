from bayes_opt import BayesianOptimization
import numpy as np

REPO_PATH = "/Users/fraser/Documents/University/Fourth Year/Dexterous Manipulation/Dexterous Manipulation"


def evaluate_design(agent, design_parameters, n_episodes, render=False):
    rewards = []
    agent.base_env.set_evaluation_design(design_parameters)
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
                obs = agent.base_env.reset()
                break
    return np.mean(rewards)


class DesignOptimizer:
    """
    Optimizes design through use of trained agent which can make actions across design space. This uses bayesian
    optimization in order to sample designs; each design is evaluated, and best one is selected
    """

    def __init__(self, evaluator):
        self.evaluator = evaluator

    def search(self, parameter_bounds, n_iter=100, random_searches=10, random_state=1):
        optimizer = BayesianOptimization(f=self.evaluator.evaluate_design,
                                         pbounds=parameter_bounds,
                                         random_state=random_state,
                                         verbose=2)
        optimizer.maximize(n_iter=n_iter, init_points=random_searches)

        return optimizer.res
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy

# from simple_reacher.bayesian_optimization import AdaptiveAgentOptimizer, MultitaskAgentOptimizer
from simple_reacher.agent import Simple2DAgent
from simple_reacher.env import SimpleRobot2D, EnvType, LinkMode, AngleMode, RewardMode, FIXED_LINK_LENGTHS
from common.utils import ModelStorage, LearningHandler

REPO_PATH = "/Users/fraser/Documents/University/Fourth Year/Dexterous Manipulation/Dexterous Manipulation"


class CustomMlpPolicy(MlpPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, **_kwargs):
        super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, layers=[32, 32], **_kwargs)


class CustomMlpLstmPolicy(MlpLstmPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, **_kwargs):
        super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, layers=[256, 256, 256],
                         **_kwargs)


def train(model_name, description, timesteps, model_storage, new_model=False,
          action_granularity=None, starting_model_path=None, checkpoint_interval=None, link_mode=LinkMode.OPTIMAL,
          reward_mode=RewardMode.LINEAR, learning_rate=0.00025, gamma=0.99, link_lengths=FIXED_LINK_LENGTHS,
          env_type=EnvType.REACHER_3_DOF):
    r_plot_path = f"{REPO_PATH}/simple_reacher/figures/realtime/{model_name}"
    t_plot_path = f"{REPO_PATH}/simple_reacher/figures/timesteps/{model_name}"
    model_path = f"{REPO_PATH}/simple_reacher/models/{model_name}"

    env = SimpleRobot2D(action_granularity=action_granularity,
                        env_type=env_type, link_mode=link_mode, angle_mode=AngleMode.SIM,
                        reward_mode=reward_mode, link_lengths=link_lengths)
    ppo_agent = Simple2DAgent(env)

    if new_model:
        model_storage.new_model(model_name, description, model_path, r_plot_path, t_plot_path)
        if starting_model_path is None:
            ppo_agent.new_model(policy=CustomMlpPolicy, gamma=gamma)
        else:
            ppo_agent.load_model(starting_model_path)
    else:
        ppo_agent.load_model(model_path if starting_model_path is None else starting_model_path)

    learning_handler = LearningHandler(model_name, model_storage, max_points=150)
    ppo_agent.learn(timesteps,
                    learning_handler,
                    checkpoint_interval=checkpoint_interval,
                    path=model_path, learning_rate=learning_rate)

    return ppo_agent


def demo(model_name, action_granularity=360, timestep_sleep=0, link_mode=LinkMode.OPTIMAL,
         link_lengths=FIXED_LINK_LENGTHS, env_type=EnvType.REACHER_3_DOF):
    model_path = f"{REPO_PATH}/simple_reacher/models/{model_name}.pkl"

    env = SimpleRobot2D(action_granularity=action_granularity,
                        env_type=env_type, link_mode=link_mode, angle_mode=AngleMode.SIM,
                        reward_mode=RewardMode.LINEAR, random_target=False)
    ppo_agent = Simple2DAgent(env)

    ppo_agent.load_model(model_path)
    ppo_agent.demo(timestep_sleep)


def optimize_multitask(model_name, action_granularity=360, render=False):
    model_path = f"{REPO_PATH}/simple_reacher/models/{model_name}.pkl"

    env = SimpleRobot2D(action_granularity=action_granularity,
                        env_type=EnvType.MULTI_TARG_2_DOF, link_mode=LinkMode.FIXED, angle_mode=AngleMode.SIM,
                        reward_mode=RewardMode.LINEAR)
    ppo_agent = Simple2DAgent(env)

    ppo_agent.load_model(model_path)

    optimizer = MultitaskAgentOptimizer(ppo_agent)
    print(optimizer.search(n_iter=50))


def optimize_adaptive(model_name, action_granularity, start_lengths=[0.15, 0.3, 0], render=False):
    model_path = f"reacher/models/{model_name}.pkl"

    env = SimpleRobot2D(action_granularity=action_granularity,
                        env_type=EnvType.MULTI_TARG_2_DOF, link_mode=LinkMode.FIXED, angle_mode=AngleMode.SIM,
                        reward_mode=RewardMode.LINEAR, link_lengths=start_lengths)
    ppo_agent = Simple2DAgent(env)

    ppo_agent.load_model(model_path)

    optimizer = AdaptiveAgentOptimizer(ppo_agent)
    return optimizer.search(n_iter=10)


if __name__ == "__main__":
    NAME = "random_reacher"
    DESCRIPTION = "Desc: Fixed links, single target.\nNetwork size: 32, 32\nDOF: " \
                  "2\nReward: Negative sum of distances" \
                  "\nGranularity: 200"
    link_lengths = [0.04, 0.02, 0]

    storage = ModelStorage()

    # train(NAME, DESCRIPTION, 150000000, storage, new_model=True, action_granularity=200, checkpoint_interval=500000,
    #       link_mode=LinkMode.RANDOM, reward_mode=RewardMode.LINEAR, learning_rate=0.00025, gamma=0.95)

    demo(NAME, action_granularity=200, link_mode=LinkMode.RANDOM, env_type=EnvType.REACHER_3_DOF)

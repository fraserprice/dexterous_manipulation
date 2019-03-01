from stable_baselines.common.policies import MlpPolicy

from curiosity_module import CuriosityModule
from reacher_pymunk.agent import ReacherAgent
from reacher_pymunk.env import Reacher, SEG_LENGTH_RANGE, LENGTHS, LinkMode, TARGET_RANGE, MotorActionType
from utils import LearningHandler, ModelStorage

REPO_PATH = "/Users/fraser/Documents/University/Fourth Year/Dexterous Manipulation/Dexterous Manipulation"
CURIOSITY_HIDDEN = [32, 32, 32, 32]
POLICY_HIDDEN = [32, 32, 32, 32]


class CustomMlpPolicy(MlpPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, **_kwargs):
        super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, layers=POLICY_HIDDEN, **_kwargs)


def train(model_name, description, timesteps, model_storage, new_model=False,
          granularity=None, starting_model_path=None, checkpoint_interval=None,
          learning_rate=0.00025, gamma=0.99, lengths=LENGTHS, seg_length_range=SEG_LENGTH_RANGE,
          curiosity=False, subproc=False, sparse=True, link_mode=LinkMode.RANDOM,
          target_range=TARGET_RANGE, motor_action_type=MotorActionType.RATE_FORCE):
    r_plot_path = f"{REPO_PATH}/reacher_pymunk/figures/realtime/{model_name}"
    t_plot_path = f"{REPO_PATH}/reacher_pymunk/figures/timesteps/{model_name}"
    model_path = f"{REPO_PATH}/reacher_pymunk/models/agent/{model_name}"
    curiosity_path = f"{REPO_PATH}/reacher_pymunk/models/curiosity/{model_name}" if curiosity else None

    env = Reacher(granularity=granularity, seg_length_range=seg_length_range, arm_segment_lengths=lengths,
                  sparse=sparse, link_mode=link_mode, target_range=target_range, motor_action_type=motor_action_type)

    if curiosity:
        action_space = env.action_space.shape[0]
        state_space = env.observation_space.shape[0]
        curiosity_module = CuriosityModule(action_space + state_space, state_space, CURIOSITY_HIDDEN)
        if not new_model:
            curiosity_module.load_forward(curiosity_path)
        env.add_curiosity_module(curiosity_module)

    agent = ReacherAgent(env, subproc=subproc)

    if new_model:
        model_storage.new_model(model_name, description, model_path, r_plot_path, t_plot_path)
        if starting_model_path is None:
            agent.new_model(policy=CustomMlpPolicy, gamma=gamma)
        else:
            agent.load_model(starting_model_path)
    else:
        agent.load_model(model_path if starting_model_path is None else starting_model_path)

    learning_handler = LearningHandler(model_name, model_storage, max_points=150)
    agent.learn(timesteps,
                learning_handler,
                checkpoint_interval=checkpoint_interval,
                path=model_path, learning_rate=learning_rate,
                curiosity_path=curiosity_path)

    return agent


def demo(model_name, granularity=100, timestep_sleep=0, seg_length_range=SEG_LENGTH_RANGE, lengths=LENGTHS, sparse=True,
         link_mode=LinkMode.RANDOM, target_range=TARGET_RANGE, motor_action_type=MotorActionType.RATE_FORCE):
    model_path = f"{REPO_PATH}/reacher_pymunk/models/agent/{model_name}.pkl"

    env = Reacher(granularity=granularity, seg_length_range=seg_length_range, arm_segment_lengths=lengths,
                  render=True, sparse=sparse, link_mode=link_mode, target_range=target_range,
                  motor_action_type=motor_action_type)
    agent = ReacherAgent(env)

    agent.load_model(model_path)
    agent.demo(timestep_sleep)


if __name__ == "__main__":
    NAME = "sparse_general-optimal_randtarg_large_cont"
    DESCRIPTION = "Desc: Random links (20, 200), random target (50, 550)\n" \
                  "Network size: 32, 32, 32, 32\n" \
                  "DOF: 2 (1 x 2 seg)\n" \
                  "Reward: 100 if target else 0\n" \
                  "Granularity: Continuous"

    storage = ModelStorage()

    # train(NAME, DESCRIPTION, 500000000, storage, new_model=True, granularity=None,
    #       checkpoint_interval=50000, curiosity=False, subproc=True, sparse=True, link_mode=LinkMode.GENERAL_OPTIMAL,
    #       target_range=TARGET_RANGE, motor_action_type=MotorActionType.RATE_FORCE, learning_rate=0.0001)
    demo(NAME, sparse=True, granularity=None, link_mode=LinkMode.GENERAL_OPTIMAL, motor_action_type=MotorActionType.RATE_FORCE)

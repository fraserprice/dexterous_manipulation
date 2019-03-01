from stable_baselines.common.policies import MlpPolicy

from curiosity_module import CuriosityModule
from grabber_pymunk.agent import GrabberAgent
from grabber_pymunk.env import Grabber, LENGTHS, SEG_LENGTH_RANGE
from utils import LearningHandler, ModelStorage

REPO_PATH = "/Users/fraser/Documents/University/Fourth Year/Dexterous Manipulation/Dexterous Manipulation"
CURIOSITY_HIDDEN = [32, 32, 32]
POLICY_HIDDEN = [32, 32, 32]


class CustomMlpPolicy(MlpPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, **_kwargs):
        super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, layers=POLICY_HIDDEN, **_kwargs)


def train(model_name, description, timesteps, model_storage, new_model=False,
          granularity=None, starting_model_path=None, checkpoint_interval=None,
          learning_rate=0.00025, gamma=0.99, lengths=LENGTHS, seg_length_range=SEG_LENGTH_RANGE,
          curiosity=False, subproc=False, sparse=True):
    r_plot_path = f"{REPO_PATH}/grabber_pymunk/figures/realtime/{model_name}"
    t_plot_path = f"{REPO_PATH}/grabber_pymunk/figures/timesteps/{model_name}"
    model_path = f"{REPO_PATH}/grabber_pymunk/models/agent/{model_name}"
    curiosity_path = f"{REPO_PATH}/grabber_pymunk/models/curiosity/{model_name}" if curiosity else None

    env = Grabber(granularity=granularity, seg_length_range=seg_length_range, arm_segment_lengths=lengths,
                  sparse=sparse)

    if curiosity:
        action_space = env.action_space.shape[0]
        state_space = env.observation_space.shape[0]
        curiosity_module = CuriosityModule(action_space + state_space, state_space, CURIOSITY_HIDDEN)
        if not new_model:
            curiosity_module.load_forward(curiosity_path)
        env.add_curiosity_module(curiosity_module)

    agent = GrabberAgent(env, subproc=subproc)

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


def demo(model_name, granularity=100, timestep_sleep=0, seg_length_range=SEG_LENGTH_RANGE, lengths=LENGTHS):
    model_path = f"{REPO_PATH}/grabber_pymunk/models/agent/{model_name}.pkl"

    env = Grabber(granularity=granularity, seg_length_range=seg_length_range, arm_segment_lengths=lengths,
                  render=True, sparse=True)
    agent = GrabberAgent(env)

    agent.load_model(model_path)
    agent.demo(timestep_sleep)


if __name__ == "__main__":
    NAME = "non_curious_sparse_test"
    DESCRIPTION = "Desc: Fixed links (120 x 2), fixed target (300, 350), -100 drop\n" \
                  "Network size: 32, 32, 32\n" \
                  "DOF: 8 (2 x 2 seg)\n" \
                  "Reward: 0.01 * Negative distance\n" \
                  "Granularity: 100"

    NAME_CUR = "curious_sparse_test"
    DESCRIPTION_CUR = "Desc: Fixed links (120 x 2), fixed target (300, 350)\n" \
                      "Network size: 32, 32\n" \
                      "DOF: 8 (2 x 2 seg)\n" \
                      "Reward: 0.01 * curiosity, +100 obj, -10 drop\n" \
                      "Granularity: 100"

    storage = ModelStorage()

    # train(NAME_CUR, DESCRIPTION_CUR, 15000000, storage, new_model=False, granularity=100, seg_length_range=None,
    #       lengths=((120, 120), (120, 120)), checkpoint_interval=50000, curiosity=True, subproc=True, sparse=True)
    # train(NAME, DESCRIPTION, 10000000, storage, new_model=True, granularity=100, seg_length_range=None,
    #       lengths=((120, 120), (120, 120)), checkpoint_interval=100000, curiosity=False, subproc=True, sparse=True)
    demo(NAME_CUR, seg_length_range=None, lengths=((115, 115), (115, 115)),)

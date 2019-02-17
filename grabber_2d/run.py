from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy

from grabber_2d.agent import Grabber2DAgent
from grabber_2d.env import Grabber2D, LENGTHS, SEG_LENGTH_RANGE
from utils import LearningHandler, ModelStorage

REPO_PATH = "/Users/fraser/Documents/University/Fourth Year/Dexterous Manipulation/Dexterous Manipulation"


class CustomMlpPolicy(MlpPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, **_kwargs):
        super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, layers=[32, 32, 32], **_kwargs)


class CustomMlpLstmPolicy(MlpLstmPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, **_kwargs):
        super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, layers=[256, 256, 256],
                         **_kwargs)


def train(model_name, description, timesteps, model_storage, new_model=False,
          granularity=None, starting_model_path=None, checkpoint_interval=None,
          learning_rate=0.0005, gamma=0.99, lengths=LENGTHS, seg_length_range=SEG_LENGTH_RANGE):
    r_plot_path = f"{REPO_PATH}/grabber_2d/figures/realtime/{model_name}"
    t_plot_path = f"{REPO_PATH}/grabber_2d/figures/timesteps/{model_name}"
    model_path = f"{REPO_PATH}/grabber_2d/models/{model_name}"

    env = Grabber2D(granularity=granularity, seg_length_range=seg_length_range, arm_segment_lengths=lengths)
    agent = Grabber2DAgent(env)

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
                    path=model_path, learning_rate=learning_rate)

    return agent


def demo(model_name, granularity=100, timestep_sleep=0, seg_length_range=SEG_LENGTH_RANGE, lengths=LENGTHS):
    model_path = f"{REPO_PATH}/grabber_2d/models/{model_name}.pkl"

    env = Grabber2D(granularity=granularity, seg_length_range=seg_length_range, arm_segment_lengths=lengths, render=True)
    agent = Grabber2DAgent(env)

    agent.load_model(model_path)
    agent.demo(timestep_sleep)


if __name__ == "__main__":
    NAME = "grabber2d_32-32-32_100-gran_2dof"
    DESCRIPTION = "Desc: Fixed links (150 x 2), fixed target (300, 350).\n" \
                  "Network size: 32, 32, 32\nDOF: " \
                  "4 (2 x 2 seg)\nReward: Negative distance / 10" \
                  "\nGranularity: 100"

    storage = ModelStorage()

    train(NAME, DESCRIPTION, 10000000, storage, new_model=False, granularity=100, seg_length_range=None,
          lengths=((150, 150), (150, 150)), checkpoint_interval=100000)
    # demo(NAME, seg_length_range=None, lengths=((150, 150), (150, 150)))

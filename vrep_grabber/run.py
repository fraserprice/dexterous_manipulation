from stable_baselines.common.policies import MlpLstmPolicy, MlpPolicy

from vrep_grabber.agent import GrabberAgent, VREP_PORTS, DEFAULT_VREP_PORT
from vrep_grabber.env import EnvType
from common.utils import LearningHandler, ModelStorage

REPO_PATH = "/Users/fraser/Documents/University/Fourth Year/Dexterous Manipulation/Dexterous Manipulation"


class CustomMlpPolicy(MlpPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, **_kwargs):
        super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, layers=[64, 64], **_kwargs)


class CustomMlpLstmPolicy(MlpLstmPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, **_kwargs):
        super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, layers=[256, 256, 256],
                         **_kwargs)


def handle_docker(env_type, subproc=True):
    for port in VREP_PORTS if subproc else [DEFAULT_VREP_PORT]:
        print(f"Starting container on port {port}")
        if env_type == EnvType.DOF3:
            name = "dof3"
        elif env_type == EnvType.DOF6:
            name = "dof3"
        else:
            name = "dof9"
        os.system(f"./vrep_grabber/start_docker_{name} {port}")

    time.sleep(10)

    atexit.register(lambda: os.system("./kill_docker"))


def train(env_type, model_name, description, timesteps, model_storage, new_model=False,
          action_granularity=None, starting_model_path=None, subproc=True, checkpoint_interval=None):

    handle_docker(env_type, subproc)

    r_plot_path = f"{REPO_PATH}/vrep_grabber/figures/realtime/{model_name}"
    t_plot_path = f"{REPO_PATH}/vrep_grabber/figures/timesteps/{model_name}"
    model_path = f"{REPO_PATH}/vrep_grabber/models/{model_name}"

    ppo_agent = GrabberAgent(env_type, action_granularity=action_granularity, subproc=subproc)

    if new_model:
        model_storage.new_model(model_name, description, model_path, r_plot_path, t_plot_path)
        ppo_agent.new_model(policy=CustomMlpPolicy, gamma=0.99)
    else:
        ppo_agent.load_model(model_path if starting_model_path is None else starting_model_path)

    learning_handler = LearningHandler(model_name, model_storage, max_points=150)
    ppo_agent.learn(timesteps,
                    learning_handler,
                    checkpoint_interval=checkpoint_interval,
                    path=model_path)


def demo(env_type, model_name, action_granularity=None, timestep_sleep=0):
    model_path = f"vrep_grabber/models/{model_name}.pkl"

    ppo_agent = GrabberAgent(env_type, action_granularity=action_granularity, subproc=False)

    ppo_agent.load_model(model_path)
    ppo_agent.demo(timestep_sleep)


if __name__ == "__main__":
    NAME = "64_64_6dof_02height_no-pen_15gran"
    NAME_3DOF = "64_64_3dof_02height_no-pen_15gran"
    TYPE = EnvType.DOF6

    DESCRIPTION = "Network size: 64, 64\nDOF: 3\nReward: 0.2 z with no penalty for x,y\nGranularity: 15"

    storage = ModelStorage()

    # train(TYPE, NAME, DESCRIPTION, 3000000, storage, new_model=False, subproc=True, action_granularity=15,
    #       checkpoint_interval=50000)
    demo(TYPE, NAME, action_granularity=15)

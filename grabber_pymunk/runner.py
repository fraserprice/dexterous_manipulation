from common.constants import RewardType, LinkMode
from common.design_optimizer import evaluate_design
from grabber_pymunk.env import Grabber, LENGTHS

from common.base_runner import BaseRunner


class GrabberEvaluator:
    """
    Assume agent is OPTIMAL or GENERAL_OPTIMAL agent, i.e. has length inputs.
    """
    def __init__(self, agent):
        self.agent = agent
        self.agent.base_env.evaluation_mode = True

    def evaluate_design(self, l11, l12, l13, l21, l22, l23, n_episodes=10, render=False):
        design = {
            "l11": l11,
            "l12": l12,
            "l13": l13,
            "l21": l21,
            "l22": l22,
            "l23": l23,
        }
        return evaluate_design(self.agent, design, n_episodes, render)


class GrabberRunner(BaseRunner):
    def __init__(self, link_mode=LinkMode.FIXED, granularity=None, lengths=LENGTHS, reward_type=RewardType.SPARSE,
                 square_target=True):
        super().__init__()
        self.env = Grabber(link_mode=link_mode, granularity=granularity, arm_segment_lengths=lengths,
                           reward_type=reward_type, square_target=square_target)
        self.env_name = "grabber_pymunk"
        self.evaluator_class = GrabberEvaluator


if __name__ == "__main__":
    NAME = "test123"
    DESCRIPTION = "Desc: Fixed links ((150 * 3) * 2), random square target, random start angles, curious, 256 batch\n" \
                  "Network size: 4 * 32\n" \
                  "DOF: 12 (2 (force, desired speed) x 6 seg)\n" \
                  "Reward: 10 if target else 0\n" \
                  "Granularity: 10"

    grabber_runner = GrabberRunner(link_mode=LinkMode.FIXED, granularity=10, square_target=True)
    grabber_runner.train(model_name=NAME, description=DESCRIPTION,
                         new_model=True, policy_hidden=[32, 32, 32, 32], curiosity_hidden=[32, 32, 32, 32],
                         checkpoint_interval=50000,
                         learning_rate=0.00025, gamma=0.95, batch_size=256, curiosity=True)
    grabber_runner.demo(NAME)


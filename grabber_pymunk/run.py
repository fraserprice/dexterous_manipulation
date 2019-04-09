from common.constants import RewardType
from common.design_optimizer import evaluate_design
from grabber_pymunk.env import Grabber, LENGTHS, SEG_LENGTH_RANGE

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
    def __init__(self, granularity=None, lengths=LENGTHS, seg_length_range=SEG_LENGTH_RANGE,
                 reward_type=RewardType.SPARSE):
        super().__init__()
        self.env = Grabber(granularity=granularity, seg_length_range=seg_length_range, arm_segment_lengths=lengths,
                           reward_type=reward_type)
        self.env_name = "grabber_pymunk"
        self.evaluator_class = None


if __name__ == "__main__":
    NAME = "test123"
    DESCRIPTION = "Desc: Fixed links (200, 200), random target (50, 550), random start angles, 1024 batch-size\n" \
                  "Network size: 6 * 32\n" \
                  "DOF: 4 (2 (force, desired speed) x 2 seg)\n" \
                  "Reward: 10 if target else 0, sparse_distance = 0.06\n" \
                  "Granularity: Cont"

    grabber_runner = GrabberRunner(granularity=10)
    # grabber_runner.train(model_name=NAME, description=DESCRIPTION,
    #                      new_model=True, policy_hidden=[32, 32, 32], checkpoint_interval=50000,
    #                      learning_rate=0.00025, gamma=0.95, batch_size=256)
    #
    grabber_runner.demo(NAME)

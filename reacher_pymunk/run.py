from common.base_runner import BaseRunner
from common.design_optimizer import evaluate_design
from reacher_pymunk.env import Reacher, SEG_LENGTH_RANGE, LENGTHS, LinkMode, TARGET_RANGE, MotorActionType, RewardType,\
    SPARSE_DISTANCE


class ReacherEvaluator:
    """
    Assume agent is OPTIMAL or GENERAL_OPTIMAL agent, i.e. has length inputs.
    """
    def __init__(self, agent):
        self.agent = agent
        self.agent.base_env.evaluation_mode = True

    def evaluate_design(self, link_1, link_2, n_episodes=10, render=False):
        design = {
            "link_1": link_1,
            "link_2": link_2
        }
        return evaluate_design(self.agent, design, n_episodes, render)


class ReacherRunner(BaseRunner):
    def __init__(self, granularity=None, lengths=LENGTHS, seg_length_range=SEG_LENGTH_RANGE, link_mode=LinkMode.RANDOM,
                 target_range=TARGET_RANGE, motor_action_type=MotorActionType.RATE_FORCE, target_points=None,
                 reward_type=RewardType.SPARSE, sparse_distance=SPARSE_DISTANCE, render=False):
        super().__init__()
        self.env = Reacher(granularity=granularity,
                           seg_length_range=seg_length_range,
                           arm_segment_lengths=lengths,
                           reward_type=reward_type,
                           link_mode=link_mode,
                           target_range=target_range,
                           motor_action_type=motor_action_type,
                           target_points=target_points,
                           sparse_distance=sparse_distance,
                           render=render)
        self.env_name = "reacher_pymunk"
        self.evaluator_class = ReacherEvaluator


if __name__ == "__main__":
    NAME = "test123"
    DESCRIPTION = "Desc: Fixed links (200, 200), random target (50, 550), random start angles, 1024 batch-size\n" \
                  "Network size: 6 * 32\n" \
                  "DOF: 4 (2 (force, desired speed) x 2 seg)\n" \
                  "Reward: 10 if target else 0, sparse_distance = 0.06\n" \
                  "Granularity: Cont"

    reacher_runner = ReacherRunner(granularity=10, link_mode=LinkMode.GENERAL_OPTIMAL, sparse_distance=0.06)
    reacher_runner.train(timesteps=11000, model_name=NAME, description=DESCRIPTION,
                         new_model=True, policy_hidden=[32, 32, 32], checkpoint_interval=5000,
                         learning_rate=0.00025, gamma=0.95, batch_size=256)
    reacher_runner.bo_search({'link_1': (20, 200), 'link_2': (20, 200)}, n_iter=5)

    reacher_runner.demo(NAME, design={'link_1': 20, 'link_2': 20})

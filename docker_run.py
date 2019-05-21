from common.constants import LinkMode, MotorActionType, AgentType
from grabber_pymunk.runner import GrabberRunner
from reacher_pymunk.env import Reacher
from reacher_pymunk.runner import ReacherRunner, ReacherDesignRunner

NAME = "grabber-fixed-64x4-cont-64x8batch-0001lr"
DESCRIPTION = "Desc: Fixed links ((150 * 3) * 2), random square target, random start angles, 256 batch\n" \
              "Network size: 64x4\n" \
              "DOF: 12 (2 (force, desired speed) x 6 seg)\n" \
              "Reward: 10 if target else 0\n" \
              "Granularity: 10"

grabber_runner = GrabberRunner(link_mode=LinkMode.FIXED, granularity=None, square_target=False)
grabber_runner.train(model_name=NAME, description=DESCRIPTION,
                     new_model=True, policy_hidden=[64] * 4, curiosity_hidden=[64] * 4,
                     checkpoint_interval=10000,
                     learning_rate=0.00025, gamma=0.95, batch_size=8, curiosity=False)
# grabber_runner.demo(NAME)

DESIGN_NAME = "design-test"
NAME = "reacher-random-128x4-cont-64x4batch-0001lr"
DESCRIPTION = "Desc:Fixed, random target (0, 600), random start angles, 8 batch-size, 64 envs\n" \
              "Network size: 64x4\n" \
              "DOF: 4 (2 X 2 seg)\n" \
              "Reward: 100 if target else 0, sparse_distance = 0.08\n" \
              "Granularity: Cont"


reacher_runner = ReacherRunner(granularity=None, link_mode=LinkMode.FIXED, sparse_distance=0.08,
                               motor_action_type=MotorActionType.RATE_FORCE, agent_type=AgentType.PPO)
# reacher_runner.train(model_name=NAME, description=DESCRIPTION,
#                      new_model=True, policy_hidden=[128] * 4, checkpoint_interval=10000,
#                      learning_rate=0.0001, gamma=0.99, batch_size=4, envs=64, subproc=True)
# reacher_runner.demo(NAME)
reacher_runner.bo_search(NAME, {'link_1': (20, 200), 'link_2': (20, 200)})

# rdr = ReacherDesignRunner(NAME, render=True)
# rdr.train(DESIGN_NAME, "...", new_model=True, policy_hidden=[4] * 3, checkpoint_interval=500,
#           learning_rate=0.00025, gamma=0.99, batch_size=8, subproc=False)
# rdr.demo(DESIGN_NAME)

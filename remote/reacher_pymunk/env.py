import math
import random

import numpy as np
import pymunk
from pygame.color import THECOLORS
from pymunk import Circle
import pygame
from gym import spaces

from common.constants import MotorActionType, RewardType, LinkMode, ActionSpace
from common.pymunk_env import PymunkEnv, PymunkSimulation, ENV_SIZE

NO_COLLISION_TYPE = 10
GHOST_TYPE = 11
MAX_MOTOR_RATE = 4
MIN_MOTOR_FORCE = 2000000
MAX_MOTOR_FORCE = 10000000
EP_LENGTH = 120
TARGET_POS = (400, 400)
TARGET_MASS = 10
SEG_LENGTH_RANGE = (20, 200)
TARGET_RANGE = [(50, 550), (50, 550)]
ANCHOR_POINT = (300, 300)
LENGTHS = (150, 150)
TARGET_SIZE = 100
SPARSE_DISTANCE = 0.06


class Reacher(PymunkEnv):
    def __init__(self, arm_anchor_points=ANCHOR_POINT, arm_segment_lengths=LENGTHS,
                 target=TARGET_POS, target_range=None, target_points=None,
                 granularity=None, render=False, reward_type=RewardType.DENSE, link_mode=LinkMode.RANDOM,
                 motor_action_type=MotorActionType.RATE_FORCE, sparse_distance=SPARSE_DISTANCE):
        super().__init__(render, reward_type, granularity, link_mode, len(arm_segment_lengths), motor_action_type,
                         ReacherSimulation)
        self.link_mode = link_mode
        self.motor_action_type = motor_action_type
        self.sparse_distance = sparse_distance
        self.simulation = ReacherSimulation(arm_anchor_points, arm_segment_lengths, target, do_render=render,
                                            sparse=(self.reward_type != RewardType.DENSE),
                                            sparse_distance=self.sparse_distance)
        self.arm_anchor_points = arm_anchor_points
        self.arm_segment_lengths = arm_segment_lengths
        self.target_range = target_range
        self.target = target
        self.target_points = target_points

        self.min_seg = SEG_LENGTH_RANGE[0]
        self.max_seg = SEG_LENGTH_RANGE[1]
        self.min_force = MIN_MOTOR_FORCE
        self.max_force = MAX_MOTOR_FORCE
        self.max_rate = MAX_MOTOR_RATE

        self.observation_space = spaces.Box(low=0.,
                                            high=1.,
                                            shape=(4 + 3 * self.n_joints + (1 if self.length_actions else 0),),
                                            dtype=np.float32)

    def set_evaluation_design(self, design):
        """
        :param design: {'link_1': int, 'link_2': int}
        """
        self.arm_segment_lengths = (design['link_1'], design['link_2'])

    def reset(self):
        self.steps = 0
        if self.link_mode == LinkMode.RANDOM and not self.evaluation_mode:
            self.arm_segment_lengths = [random.randint(self.min_seg, self.max_seg)
                                        for _ in self.arm_segment_lengths]

        if self.target_points is not None:
            self.target = random.choice(self.target_points)
        elif self.target_range is not None:
            self.target = (random.randint(self.target_range[0][0], self.target_range[0][1]),
                           random.randint(self.target_range[1][0], self.target_range[1][1]))

        self.simulation = ReacherSimulation(self.arm_anchor_points, self.arm_segment_lengths, self.target,
                                            do_render=self.do_render, sparse=(self.reward_type == RewardType.SPARSE),
                                            sparse_distance=self.sparse_distance)

        return self.get_observation()

    def step(self, action):
        multiplier = 1 if self.action_space_type == ActionSpace.CONTINUOUS else 1. / self.granularity
        # noinspection PyTypeChecker
        denormalized_action = self.denormalize_action(action * multiplier)
        if not self.length_actions or self.steps > 0 or self.evaluation_mode:
            self.simulation.set_motor_rates(denormalized_action[0:self.n_joints])
            if self.motor_action_type == MotorActionType.RATE_FORCE:
                self.simulation.set_motor_max_forces(denormalized_action[self.n_joints:2 * self.n_joints])
            self.simulation.step()
        else:
            print(denormalized_action[-self.n_joints:])
            self.arm_segment_lengths = denormalized_action[-self.n_joints:]
            print(self.arm_segment_lengths)
            self.simulation = ReacherSimulation(self.arm_anchor_points, self.arm_segment_lengths, self.target,
                                                do_render=self.do_render, sparse=(self.reward_type == RewardType.SPARSE),
                                                sparse_distance=self.sparse_distance)

        obs = self.get_observation()

        end_effector_pos = np.array([coord / ENV_SIZE for coord in self.simulation.end_effector_body.position])
        distance = np.linalg.norm(np.array(obs[0:2]) - end_effector_pos)
        if self.reward_type == RewardType.SPARSE:
            reward = 100 if distance < self.sparse_distance else 0
        elif self.reward_type == RewardType.DENSE:
            reward = -distance
        else:
            reward = (100 if distance < self.sparse_distance else 0) - distance
        done = self.steps > EP_LENGTH

        reward = self.get_curiosity_reward(reward, action, obs)

        return obs, reward, done, {'curiosity_loss': self.curiosity_loss_history, 'ext_reward': self.ext_reward_history}

    def get_observation(self):
        targ_pos = [coord / ENV_SIZE for coord in self.simulation.target_position]
        end_pos = [coord / ENV_SIZE for coord in self.simulation.end_effector_body.position]
        segment_angles = [b.angle / (2 * math.pi) for b in self.simulation.segment_bodies]
        segment_angvs = [(seg.angular_velocity + 4) / 8 for seg in self.simulation.segment_bodies]
        segment_lengths = [(seg - SEG_LENGTH_RANGE[0]) / (SEG_LENGTH_RANGE[1] - SEG_LENGTH_RANGE[0])
                           for seg in self.arm_segment_lengths]

        # obs = targ_pos + segment_angles + segment_lengths + segment_angvs
        obs = targ_pos + end_pos + segment_angles + segment_lengths + segment_angvs
        if self.length_actions:
            if self.steps == 0:
                obs[-2 * len(segment_angvs):-len(segment_angvs)] = [0] * len(segment_angvs)
                obs.append(0)
                if self.link_mode == LinkMode.GENERAL_OPTIMAL:
                    obs[0:2] = [0] * 2
            else:
                obs.append(1)

        return np.array(obs)


class ReacherSimulation(PymunkSimulation):
    def __init__(self, arm_anchor_point, arm_segment_lengths, target_position, do_render=False, sparse=True,
                 sparse_distance=SPARSE_DISTANCE):
        super().__init__(do_render, sparse, MAX_MOTOR_FORCE)

        self.sparse_distance = sparse_distance
        self.target_position = target_position
        self.space.gravity = (0.0, 0.0)

        self.end_effector_body = None
        end_anchor, end_pos = self.add_arm(arm_segment_lengths, arm_anchor_point)

        self.end_effector_body = pymunk.Body(1, 1)
        self.end_effector_body.position = end_pos[0], end_pos[1]
        end_effector_shape = Circle(self.end_effector_body, 8)
        end_effector_shape.collision_type = NO_COLLISION_TYPE
        self.space.add(self.end_effector_body, end_effector_shape)

        pin = pymunk.PinJoint(end_anchor, self.end_effector_body, (-arm_segment_lengths[-1] / 2, 0), (0, 0))

        self.space.add(pin)

    def render(self):
        pygame.event.get()
        self.screen.fill(THECOLORS["white"])
        target_radius = ENV_SIZE * self.sparse_distance
        if self.sparse:
            within_target = (self.end_effector_body.position[0] - self.target_position[0]) ** 2 + \
                            (self.end_effector_body.position[1] - self.target_position[1]) ** 2 < \
                            target_radius ** 2
            colour = THECOLORS["green"] if within_target else THECOLORS["red"]
            self.draw_options.draw_circle(self.target_position, 0, target_radius, colour, colour)
        else:
            distance = np.linalg.norm(self.end_effector_body.position - self.target_position)
            col = max(0, 255 - distance / 2)
            colour = (255 - col, col, 0, 255)
            self.draw_options.draw_dot(target_radius, self.target_position, colour)
        self.space.debug_draw(self.draw_options)

        pygame.display.flip()
        self.clock.tick(50)
        pygame.display.set_caption("fps: " + str(self.clock.get_fps()))

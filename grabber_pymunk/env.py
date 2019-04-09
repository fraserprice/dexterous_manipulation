import math
import random

import numpy as np
import pymunk
from pygame.color import THECOLORS
from pymunk import Body, Poly
import pygame
from gym import spaces

from common.constants import RewardType, LinkMode, MotorActionType
from common.pymunk_env import PymunkEnv, PymunkSimulation

NO_COLLISION_TYPE = 10
GHOST_TYPE = 11
ENV_SIZE = 600
MAX_MOTOR_RATE = 4
MIN_MOTOR_FORCE = 2000000
MAX_MOTOR_FORCE = 10000000
EP_LENGTH = 300
TARGET_POS = (300, 350)
TARGET_MASS = 10
SEG_LENGTH_RANGE = (20, 200)
TARGET_RANGE = [(200, 400), (250, 450)]
ANCHOR_POINTS = ((400, 350), (200, 350))
LENGTHS = ((150, 150, 150), (150, 150, 150))
TARGET_SIZE = 100
SPARSE_DISTANCE = 0.3


class Grabber(PymunkEnv):
    def __init__(self, link_mode=LinkMode.RANDOM, arm_anchor_points=ANCHOR_POINTS, arm_segment_lengths=LENGTHS,
                 target=TARGET_POS, seg_length_range=SEG_LENGTH_RANGE, target_range=None, granularity=None,
                 render=False, reward_type=RewardType.SPARSE, motor_action_type=MotorActionType.RATE_FORCE):
        n_joints = sum(len(arm_segs) for arm_segs in arm_segment_lengths)
        super().__init__(render, reward_type, granularity, link_mode, n_joints, motor_action_type, GrabberSimulation)

        self.link_mode = link_mode
        self.sparse = reward_type == RewardType.SPARSE
        self.simulation = GrabberSimulation(arm_anchor_points, arm_segment_lengths, target, MAX_MOTOR_FORCE, do_render=render)
        self.motor_action_type = motor_action_type

        self.arm_anchor_points = arm_anchor_points
        self.arm_segment_lengths = arm_segment_lengths
        self.seg_length_range = seg_length_range
        self.target_range = target_range
        self.target = target

        self.observation_space = spaces.Box(low=0.,
                                            high=1.,
                                            shape=(6 + 2 * self.n_joints,),
                                            dtype=np.float32)

    def set_evaluation_design(self, design):
        self.arm_segment_lengths = ((design['l11'], design['l12'], design['l13']),
                                    (design['l21'], design['l22'], design['l23']))

    def reset(self):
        self.steps = 0
        if self.link_mode == LinkMode.RANDOM and not self.evaluation_mode:
            lengths = []
            for arm_segs in self.arm_segment_lengths:
                lengths.append([random.randint(self.seg_length_range[0], self.seg_length_range[1]) for _ in arm_segs])
            self.arm_segment_lengths = lengths

        if self.target_range is not None:
            self.target = (random.randint(TARGET_RANGE[0][0], TARGET_RANGE[0][1]),
                           random.randint(TARGET_RANGE[1][0], TARGET_RANGE[1][1]))

        self.simulation = GrabberSimulation(self.arm_anchor_points, self.arm_segment_lengths, self.target, MAX_MOTOR_FORCE,
                                            do_render=self.do_render, sparse=self.sparse)

        return self.get_observation()

    def step(self, action):
        obs = self.apply_action(action)

        distance = np.linalg.norm(np.array(self.target) - np.array((obs[0] * ENV_SIZE, obs[1] * ENV_SIZE))) * 0.01
        if self.sparse:
            reward = 100 if distance < SPARSE_DISTANCE else 0
        else:
            reward = -distance
        done = self.steps > EP_LENGTH

        reward = self.get_curiosity_reward(reward, action, obs)

        return obs, reward, done, {'curiosity_loss': self.curiosity_loss_history, 'ext_reward': self.ext_reward_history}

    def denormalize_action(self, action):
        denormalized_rates = [rate * MAX_MOTOR_RATE * 2 - MAX_MOTOR_RATE for rate in action[0:self.n_joints]]
        denormalized_forces = [force * (MAX_MOTOR_FORCE - MIN_MOTOR_FORCE) + MIN_MOTOR_FORCE
                               for force in action[self.n_joints:]]
        return denormalized_rates + denormalized_forces

    def get_observation(self):
        t_pos = self.simulation.target_body.position / ENV_SIZE
        t_ang = self.simulation.target_body.angle / (2 * math.pi)
        t_v = self.simulation.target_body.velocity + 10 / 20
        t_angv = self.simulation.target_body.angular_velocity + 3 / 6
        segment_positions = [b.angle / (2 * math.pi) for b in self.simulation.segment_bodies]

        segment_lengths = [(seg - SEG_LENGTH_RANGE[0]) / (SEG_LENGTH_RANGE[1] - SEG_LENGTH_RANGE[0])
                           for arm_segs in self.arm_segment_lengths for seg in arm_segs]

        return np.array([t_pos[0], t_pos[1], t_ang,
                         max(0, min(t_v[0], 1)), max(0, min(t_v[1], 1)),
                         t_angv] + segment_positions + segment_lengths)


class GrabberSimulation(PymunkSimulation):
    def __init__(self, arm_anchor_points, arm_segment_lengths, target_position, max_motor_force, do_render=False,
                 sparse=True, target_size=TARGET_SIZE):
        super().__init__(do_render, sparse, max_motor_force)

        self.target_size = target_size
        self.target_position = target_position
        self.space.gravity = (0.0, -900.0)

        floor = pymunk.Segment(self.space.static_body,
                               (int(-5 * ENV_SIZE), int(0.17 * ENV_SIZE)),
                               (int(5 * ENV_SIZE), int(0.17 * ENV_SIZE)),
                               0.017 * ENV_SIZE)
        floor.elasticity = 0.1
        floor.friction = 0.5
        floor.collision_type = NO_COLLISION_TYPE
        self.space.add(floor)

        for i, anchor_point in enumerate(arm_anchor_points):
            self.add_arm(arm_segment_lengths[i], anchor_point)

        target_size = (target_size, target_size)
        target_mass = TARGET_MASS
        target_moment = pymunk.moment_for_box(target_mass, target_size)
        self.target_body = Body(target_mass, target_moment)
        self.target_body.position = ENV_SIZE / 2, target_size[1] / 2 + int(0.187 * ENV_SIZE)
        target_shape = Poly.create_box(self.target_body, target_size)
        target_shape.elasticity = 0.1
        target_shape.friction = 1.0
        self.space.add(self.target_body, target_shape)

    def render(self):
        pygame.event.get()
        self.screen.fill(THECOLORS["white"])
        square_reach = math.sqrt(((self.target_size / 2) ** 2) * 2)
        target_radius = 100 * SPARSE_DISTANCE + square_reach
        if self.sparse:
            within_target = (self.target_body.position[0] - self.target_position[0]) ** 2 + \
                            (self.target_body.position[1] - self.target_position[1]) ** 2 < \
                            (target_radius - square_reach) ** 2
            colour = THECOLORS["green"] if within_target else THECOLORS["red"]
            self.draw_options.draw_circle(self.target_position, 0, target_radius, colour, colour)
        else:
            distance = np.linalg.norm(self.target_body.position - self.target_position)
            col = max(0, 255 - 1.5 * distance)
            colour = (255 - col, col, 0, 255)
            self.draw_options.draw_dot(target_radius, self.target_position, colour)
        self.space.debug_draw(self.draw_options)

        pygame.display.flip()
        self.clock.tick(50)
        pygame.display.set_caption("fps: " + str(self.clock.get_fps()))

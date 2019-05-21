import math
import random
from abc import ABC

import pygame
import pymunk
from gym import spaces
from pymunk import pygame_util, Space
import numpy as np

from common.constants import LinkMode, MotorActionType, ActionSpace
from common.normalized_env import NormalizedEnv

NO_COLLISION_TYPE = 10
GHOST_TYPE = 11
ENV_SIZE = 600


class PymunkEnv(NormalizedEnv, ABC):
    def __init__(self, do_render, reward_type, granularity, link_mode, n_joints, motor_action_type, simulation_class):
        super().__init__()
        self.do_render = do_render
        self.reward_type = reward_type
        self.granularity = granularity
        self.motor_action_type = motor_action_type
        self.simulation_class = simulation_class

        self.min_seg = None
        self.max_seg = None
        self.min_force = None
        self.max_force = None
        self.max_rate = None

        self.target = None
        self.arm_anchor_points = None
        self.arm_segment_lengths = None
        self.evaluation_mode = False
        self.simulation = None

        self.n_joints = n_joints
        self.length_actions = link_mode == LinkMode.OPTIMAL or link_mode == LinkMode.GENERAL_OPTIMAL
        self.seg_length_actions = self.n_joints if self.length_actions else 0
        self.joint_actions = self.n_joints * (2 if self.motor_action_type == MotorActionType.RATE_FORCE else 1)
        actions = self.joint_actions + self.seg_length_actions
        if self.granularity is None:
            self.action_space = spaces.Box(low=-1., high=1., shape=(actions,), dtype=np.float32)
            self.action_space_type = ActionSpace.CONTINUOUS
        else:
            self.action_space = spaces.MultiDiscrete([granularity] * actions)
            self.action_space_type = ActionSpace.DISCRETE

    def render(self, mode='human'):
        self.simulation.render()

    def denormalize_action(self, action):
        action = [(a + 1) / 2 for a in action]
        denormalized_action = [rate * self.max_rate * 2 - self.max_rate for rate in action[0:self.n_joints]]
        if self.motor_action_type == MotorActionType.RATE_FORCE:
            denormalized_action += [force * (self.max_force - self.min_force) + self.min_force
                                    for force in action[self.n_joints:2 * self.n_joints]]
        if self.length_actions and self.steps == 0:
            links = [l * (self.max_seg - self.min_seg) + self.min_seg for l in
                     action[-self.n_joints:]]
            denormalized_action += links
        return denormalized_action


class PymunkSimulation:
    def __init__(self, do_render, sparse, max_motor_force):
        self.do_render = do_render
        self.sparse = sparse
        self.max_motor_force = max_motor_force

        if self.do_render:
            pygame.init()
            self.screen = pygame.display.set_mode((ENV_SIZE, ENV_SIZE))
            self.draw_options = pygame_util.DrawOptions(self.screen)
            self.clock = pygame.time.Clock()
        self.motors = []
        self.segment_bodies = []

        self.space = Space()
        self.space.iterations = 20

        no_collision = self.space.add_collision_handler(NO_COLLISION_TYPE, NO_COLLISION_TYPE)
        no_collision.begin = lambda a, b, c: False
        ghost_collision = self.space.add_wildcard_collision_handler(GHOST_TYPE)
        ghost_collision.begin = lambda a, b, c: False

    def set_motor_rates(self, motor_rates):
        for i, motor in enumerate(self.motors):
            motor.rate = motor_rates[i]

    def set_motor_max_forces(self, motor_forces):
        for i, motor in enumerate(self.motors):
            motor.max_force = motor_forces[i]

    def step(self):
        dt = 0.03
        steps = 30
        for i in range(steps):
            self.space.step(dt / steps)

    def add_arm(self, segment_lengths, anchor_position):
        anchor = pymunk.Body(body_type=pymunk.Body.STATIC)
        anchor.position = anchor_position
        self.space.add(anchor)

        segment_anchor = anchor
        next_anchor_pos = anchor_position
        for i, segment_length in enumerate(segment_lengths):
            segment_size = segment_length, 10
            segment_body = pymunk.Body(10, pymunk.moment_for_box(10, segment_size))

            end_effector_shape = pymunk.Poly.create_box(segment_body, segment_size)
            end_effector_shape.collision_type = NO_COLLISION_TYPE
            end_effector_shape.friction = 1.0
            end_effector_shape.elasticity = 0.1

            alpha = random.random() * math.pi * 2
            dx = np.cos(alpha) * segment_length / 2
            dy = np.sin(alpha) * segment_length / 2
            segment_body.position = next_anchor_pos[0] - dx, next_anchor_pos[1] - dy
            next_anchor_pos = (next_anchor_pos[0] - 2 * dx, next_anchor_pos[1] - 2 * dy)
            segment_body.angle = alpha
            anchor_pin_pos = (0 if i == 0 else -segment_lengths[i - 1] / 2, 0)
            pin = pymunk.PinJoint(segment_anchor, segment_body, anchor_pin_pos, (segment_length / 2, 0))
            self.space.add(pin)
            self.space.add(segment_body, end_effector_shape)

            motor = pymunk.SimpleMotor(segment_anchor, segment_body, 0)
            motor.max_force = self.max_motor_force
            self.space.add(motor)
            self.motors.append(motor)
            self.segment_bodies.append(segment_body)
            segment_anchor = segment_body

        return segment_anchor, next_anchor_pos

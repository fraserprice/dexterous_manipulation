import math
import random
from abc import ABC, abstractmethod

import pygame
import pymunk
from gym import Env, spaces
from pymunk import pygame_util, Space
import numpy as np

from common.constants import LinkMode, MotorActionType, ActionSpace, RewardType

NO_COLLISION_TYPE = 10
GHOST_TYPE = 11
ENV_SIZE = 600


class PymunkEnv(Env, ABC):
    def __init__(self, do_render, reward_type, granularity, link_mode, n_joints, motor_action_type, simulation_class):
        self.do_render = do_render
        self.reward_type = reward_type
        self.granularity = granularity
        self.n_joints = n_joints
        self.motor_action_type = motor_action_type
        self.simulation_class = simulation_class
        self.target = None
        self.arm_anchor_points = None
        self.arm_segment_lengths = None

        self.steps = 0
        self.curiosity_module = None
        self.previous_observation = None
        self.evaluation_mode = False
        self.ext_reward_history = []
        self.curiosity_loss_history = []
        self.simulation = None

        self.length_actions = link_mode == LinkMode.OPTIMAL or link_mode == LinkMode.GENERAL_OPTIMAL
        self.seg_length_actions = self.n_joints if self.length_actions else 0
        self.joint_actions = self.n_joints * (2 if self.motor_action_type == MotorActionType.RATE_FORCE else 1)
        actions = self.joint_actions + self.seg_length_actions

        if self.granularity is None:
            self.action_space = spaces.Box(low=0., high=1., shape=(actions,), dtype=np.float32)
            self.action_space_type = ActionSpace.CONTINUOUS
        else:
            self.action_space = spaces.MultiDiscrete([granularity] * actions)
            self.action_space_type = ActionSpace.DISCRETE

    def render(self, mode='human'):
        self.simulation.render()

    def add_curiosity_module(self, curiosity_module):
        self.curiosity_module = curiosity_module

    def __track_info(self, curiosity_loss, ext_reward):
        self.curiosity_loss_history.append(curiosity_loss)
        self.ext_reward_history.append(ext_reward)

    def get_infos(self):
        info = (sum(self.curiosity_loss_history) / len(self.curiosity_loss_history),
                sum(self.ext_reward_history) / len(self.ext_reward_history))
        self.curiosity_loss_history = []
        self.ext_reward_history = []
        return info

    def get_curiosity_reward(self, reward, action, obs):
        curiosity_loss = None
        if self.curiosity_module is not None:
            if self.previous_observation is not None:
                curiosity_loss = 0.01 * self.curiosity_module.get_curiosity_loss(action, self.previous_observation,
                                                                                 obs).item()
                reward += curiosity_loss
            self.previous_observation = obs
        self.steps += 1

        ext_reward = reward if curiosity_loss is None else reward - curiosity_loss
        curiosity_loss = 0 if curiosity_loss is None else curiosity_loss
        self.__track_info(curiosity_loss, ext_reward)

        return reward

    @abstractmethod
    def denormalize_action(self, action):
        return NotImplemented

    @abstractmethod
    def get_observation(self):
        return NotImplemented

    def new_simulation(self, anchor, seg_lengths, target, render, sparse):
        self.simulation = self.simulation_class(anchor, seg_lengths, target, render, sparse)

    def apply_action(self, action):
        multiplier = 1 if self.action_space_type == ActionSpace.CONTINUOUS else 1. / self.granularity
        # noinspection PyTypeChecker
        denormalized_action = self.denormalize_action(action * multiplier)
        if not self.length_actions or self.steps > 0 or self.evaluation_mode:
            self.simulation.set_motor_rates(denormalized_action[0:self.n_joints])
            if self.motor_action_type == MotorActionType.RATE_FORCE:
                self.simulation.set_motor_max_forces(denormalized_action[self.n_joints:2 * self.n_joints])
            self.simulation.step()
        else:
            self.arm_segment_lengths = denormalized_action[-2:]
            self.new_simulation(self.arm_anchor_points, self.arm_segment_lengths, self.target, self.do_render,
                                (self.reward_type == RewardType.SPARSE))

        return self.get_observation()


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
        steps = 100
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

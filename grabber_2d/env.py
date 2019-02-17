import math
import random
from enum import Enum

import numpy as np
import pymunk
import torch
from pygame.color import THECOLORS
from pymunk import Space, Body, Poly
import pymunk.pygame_util as pygame_util
import pygame
from gym import Env, spaces

from curiosity_module import CuriosityModule

NO_COLLISION_TYPE = 10
GHOST_TYPE = 11
ENV_SIZE = 600
MAX_MOTOR_RATE = 15
EP_LENGTH = 100
TARGET_POS = (300, 350)
SEG_LENGTH_RANGE = (20, 200)
TARGET_RANGE = [(200, 400), (250, 450)]
ANCHOR_POINTS = ((400, 350), (200, 350))
LENGTHS = ((150, 150, 150), (150, 150, 150))


class ActionSpace(Enum):
    CONTINUOUS = 0
    DISCRETE = 1


class Grabber2D(Env):
    def __init__(self, arm_anchor_points=ANCHOR_POINTS, arm_segment_lengths=LENGTHS,
                 target=TARGET_POS, seg_length_range=SEG_LENGTH_RANGE, target_range=None,
                 granularity=None, render=False):
        self.do_render = render
        self.granularity = granularity
        self.simulation = GrabberSimulation(arm_anchor_points, arm_segment_lengths, target, do_render=render)
        self.arm_anchor_points = arm_anchor_points
        self.arm_segment_lengths = arm_segment_lengths
        self.seg_length_range = seg_length_range
        self.target_range = target_range
        self.target = target
        self.steps = 0
        self.curiosity_module = None
        self.previous_observation = None

        n_arms = sum(len(arm_segs) for arm_segs in arm_segment_lengths)
        if self.granularity is None:
            self.action_space = spaces.Box(low=0., high=1., shape=(n_arms,), dtype=np.float32)
            self.action_space_type = ActionSpace.CONTINUOUS
        else:
            self.action_space = spaces.MultiDiscrete([granularity] * 6)
            self.action_space_type = ActionSpace.DISCRETE
        self.observation_space = spaces.Box(low=0.,
                                            high=1.,
                                            shape=(6 + 3 * n_arms,),
                                            dtype=np.float32)
        self.ext_reward_history = []
        self.curiosity_loss_history = []

    def add_curiosity_module(self, curiosity_module):
        self.curiosity_module = curiosity_module

    def reset(self):
        self.steps = 0
        if self.seg_length_range is not None:
            lengths = []
            for arm_segs in self.arm_segment_lengths:
                lengths.append([random.randint(self.seg_length_range[0], self.seg_length_range[1]) for _ in arm_segs])
            self.arm_segment_lengths = lengths

        if self.target_range is not None:
            self.target = (random.randint(TARGET_RANGE[0][0], TARGET_RANGE[0][1]),
                           random.randint(TARGET_RANGE[1][0], TARGET_RANGE[1][1]))

        self.simulation = GrabberSimulation(self.arm_anchor_points, self.arm_segment_lengths, self.target, do_render=self.do_render)

        return self.__get_observation()

    def step(self, action):
        multiplier = 1 if self.action_space_type == ActionSpace.CONTINUOUS else 1. / self.granularity
        self.simulation.set_motor_rates(self.__denormalize_action(action * multiplier))
        self.simulation.step()
        obs = self.__get_observation()
        done = obs[0] < 0 or obs[0] > 1 or obs[1] < 0 or obs[1] > 1 or self.steps > EP_LENGTH
        reward = -np.linalg.norm(np.array(self.target) - np.array((obs[0] * ENV_SIZE, obs[1] * ENV_SIZE))) / 10
        curiosity_loss = None
        if self.curiosity_module is not None:
            if self.previous_observation is not None:
                curiosity_loss = self.curiosity_module.get_curiosity_loss(action, self.previous_observation, obs).item()
                reward += curiosity_loss
            self.previous_observation = obs
        self.steps += 1

        ext_reward = reward if curiosity_loss is None else reward - curiosity_loss
        curiosity_loss = 0 if curiosity_loss is None else curiosity_loss
        self.__track_info(curiosity_loss, ext_reward)

        return obs, reward, done, {'curiosity_loss': self.curiosity_loss_history, 'ext_reward': self.ext_reward_history}

    def render(self, mode='human'):
        self.simulation.render()

    @staticmethod
    def __denormalize_action(action):
        return [(rate - 0.5) * MAX_MOTOR_RATE * 2 for rate in action]

    def __get_observation(self):
        t_pos = self.simulation.target_body.position / ENV_SIZE
        t_ang = self.simulation.target_body.angle / (2 * math.pi)
        t_v = self.simulation.target_body.velocity + 10 / 20
        t_angv = self.simulation.target_body.angular_velocity + 3 / 6
        segment_positions = [p / 600 for b in self.simulation.segment_bodies for p in b.position]

        segment_lengths = [(seg - SEG_LENGTH_RANGE[0]) / (SEG_LENGTH_RANGE[1] - SEG_LENGTH_RANGE[0])
                           for arm_segs in self.arm_segment_lengths for seg in arm_segs]

        return np.array([t_pos[0], t_pos[1], t_ang,
                         max(0, min(t_v[0], 1)), max(0, min(t_v[1], 1)),
                         t_angv] + segment_positions + segment_lengths)

    def __track_info(self, curiosity_loss, ext_reward):
        self.curiosity_loss_history.append(curiosity_loss)
        self.ext_reward_history.append(ext_reward)
        if len(self.curiosity_loss_history) > 512:
            del self.curiosity_loss_history[0]
            del self.ext_reward_history[0]


class ObservationNormaliser:
    def __init__(self):
        n = 0
        t_pos_0_avg = 0
        t_pos_1_avg = 0
        t_ang_avg = 0
        t_v_0_avg = 0
        t_v_1_avg = 0
        t_angv_avg = 0


class GrabberSimulation:
    def __init__(self, arm_anchor_points, arm_segment_lengths, target_position, do_render=False):
        self.do_render = do_render
        if self.do_render:
            pygame.init()
            self.screen = pygame.display.set_mode((ENV_SIZE, ENV_SIZE))
            self.draw_options = pygame_util.DrawOptions(self.screen)
            self.clock = pygame.time.Clock()
        self.motors = []
        self.segment_bodies = []

        self.space = Space()
        self.space.iterations = 20
        self.space.gravity = (0.0, -900.0)

        floor = pymunk.Segment(self.space.static_body,
                               (int(0.417 * ENV_SIZE), int(0.17 * ENV_SIZE)),
                               (int(0.583 * ENV_SIZE), int(0.17 * ENV_SIZE)),
                               0.017 * ENV_SIZE)
        floor.elasticity = 0.99
        floor.friction = 1.0
        self.space.add(floor)

        no_collision = self.space.add_collision_handler(NO_COLLISION_TYPE, NO_COLLISION_TYPE)
        no_collision.begin = lambda a, b, c: False
        ghost_collision = self.space.add_wildcard_collision_handler(GHOST_TYPE)
        ghost_collision.begin = lambda a, b, c: False

        desired_target_body = Body(body_type=Body.STATIC)
        desired_target_body.position = target_position
        desired_target_size = (100, 100)
        desired_target_shape = Poly.create_box(desired_target_body, desired_target_size)
        desired_target_shape.collision_type = GHOST_TYPE
        desired_target_shape.color = (128, 255, 180, 255)
        self.space.add(desired_target_body, desired_target_shape)

        for i, anchor_point in enumerate(arm_anchor_points):
            self.__add_arm(arm_segment_lengths[i], anchor_point)

        target_size = (100, 100)
        target_mass = 50
        target_moment = pymunk.moment_for_box(target_mass, target_size)
        self.target_body = Body(target_mass, target_moment)
        self.target_body.position = ENV_SIZE / 2, target_size[1] / 2 + int(0.187 * ENV_SIZE)
        target_shape = Poly.create_box(self.target_body, target_size)
        target_shape.elasticity = 0.2
        target_shape.friction = 1.0
        self.space.add(self.target_body, target_shape)

    def set_motor_rates(self, motor_rates):
        for i, motor in enumerate(self.motors):
            motor.rate = motor_rates[i]

    def step(self):
        dt = 0.01
        steps = 100
        for i in range(steps):
            self.space.step(dt / steps)

    def render(self):
        pygame.event.get()
        self.screen.fill(THECOLORS["white"])
        self.space.debug_draw(self.draw_options)

        pygame.display.flip()
        self.clock.tick(50)
        pygame.display.set_caption("fps: " + str(self.clock.get_fps()))

    def __add_arm(self, segment_lengths, anchor_position):
        anchor = pymunk.Body(body_type=pymunk.Body.STATIC)
        anchor.position = anchor_position
        self.space.add(anchor)

        segment_anchor = anchor
        for i, segment_length in enumerate(segment_lengths):
            segment_size = segment_length, 10
            segment_body = pymunk.Body(10, pymunk.moment_for_box(10, segment_size))
            segment_body.angle = math.pi / 2

            segment_shape = pymunk.Poly.create_box(segment_body, segment_size)
            segment_shape.collision_type = NO_COLLISION_TYPE
            segment_shape.friction = 1.0

            if i == 0:
                segment_body.position = segment_anchor.position[0], segment_anchor.position[1] - segment_lengths[0] / 2
                pin = pymunk.PinJoint(segment_anchor, segment_body, (0, 0), (segment_lengths[0] / 2, 0))
                self.space.add(pin)
            else:
                segment_body.position = anchor.position[0], \
                                        segment_anchor.position[1] - segment_lengths[i - 1] / 2 - segment_length / 2
                pin = pymunk.PinJoint(
                    segment_anchor, segment_body, (-segment_lengths[i - 1] / 2, 0), (segment_length / 2, 0)
                )
                self.space.add(pin)

            self.space.add(segment_body, segment_shape)

            motor = pymunk.SimpleMotor(segment_anchor, segment_body, 0)
            motor.max_force = 100000000
            self.space.add(motor)
            self.motors.append(motor)
            self.segment_bodies.append(segment_body)
            segment_anchor = segment_body


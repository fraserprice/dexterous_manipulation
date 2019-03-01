import math
import random
from enum import Enum

import numpy as np
import pymunk
from pygame.color import THECOLORS
from pymunk import Space, Body, Poly, Circle
import pymunk.pygame_util as pygame_util
import pygame
from gym import Env, spaces

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


class ActionSpace(Enum):
    CONTINUOUS = 0
    DISCRETE = 1


class Grabber(Env):
    def __init__(self, arm_anchor_points=ANCHOR_POINTS, arm_segment_lengths=LENGTHS,
                 target=TARGET_POS, seg_length_range=SEG_LENGTH_RANGE, target_range=None,
                 granularity=None, render=False, sparse=False):
        self.do_render = render
        self.sparse = sparse
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

        self.n_joints = sum(len(arm_segs) for arm_segs in arm_segment_lengths)
        if self.granularity is None:
            self.action_space = spaces.Box(low=0., high=1., shape=(self.n_joints * 2,), dtype=np.float32)
            self.action_space_type = ActionSpace.CONTINUOUS
        else:
            self.action_space = spaces.MultiDiscrete([granularity] * (self.n_joints * 2))
            self.action_space_type = ActionSpace.DISCRETE
        self.observation_space = spaces.Box(low=0.,
                                            high=1.,
                                            shape=(6 + 2 * self.n_joints,),
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

        self.simulation = GrabberSimulation(self.arm_anchor_points, self.arm_segment_lengths, self.target,
                                            do_render=self.do_render, sparse=self.sparse)

        return self.__get_observation()

    def step(self, action):
        multiplier = 1 if self.action_space_type == ActionSpace.CONTINUOUS else 1. / self.granularity
        denormalized_action = self.__denormalize_action(action * multiplier)
        self.simulation.set_motor_rates(denormalized_action[0:self.n_joints])
        self.simulation.set_motor_max_forces(denormalized_action[self.n_joints:])
        self.simulation.step()

        obs = self.__get_observation()
        distance = np.linalg.norm(np.array(self.target) - np.array((obs[0] * ENV_SIZE, obs[1] * ENV_SIZE))) * 0.01
        if self.sparse:
            reward = 100 if distance < SPARSE_DISTANCE else 0
        else:
            reward = -distance
        done = self.steps > EP_LENGTH

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

        return obs, reward, done, {'curiosity_loss': self.curiosity_loss_history, 'ext_reward': self.ext_reward_history}

    def render(self, mode='human'):
        self.simulation.render()

    def __denormalize_action(self, action):
        denormalized_rates = [rate * MAX_MOTOR_RATE * 2 - MAX_MOTOR_RATE for rate in action[0:self.n_joints]]
        denormalized_forces = [force * (MAX_MOTOR_FORCE - MIN_MOTOR_FORCE) + MIN_MOTOR_FORCE
                               for force in action[self.n_joints:]]
        return denormalized_rates + denormalized_forces

    def __get_observation(self):
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

    def __track_info(self, curiosity_loss, ext_reward):
        self.curiosity_loss_history.append(curiosity_loss)
        self.ext_reward_history.append(ext_reward)

    def get_infos(self):
        info = (sum(self.curiosity_loss_history) / len(self.curiosity_loss_history),
                sum(self.ext_reward_history) / len(self.ext_reward_history))
        self.curiosity_loss_history = []
        self.ext_reward_history = []
        return info


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
    def __init__(self, arm_anchor_points, arm_segment_lengths, target_position, do_render=False, sparse=True,
                 target_size=TARGET_SIZE):
        self.do_render = do_render
        self.sparse = sparse
        self.target_size = target_size
        self.target_position = target_position
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
                               (int(-5 * ENV_SIZE), int(0.17 * ENV_SIZE)),
                               (int(5 * ENV_SIZE), int(0.17 * ENV_SIZE)),
                               0.017 * ENV_SIZE)
        floor.elasticity = 0.1
        floor.friction = 0.5
        self.space.add(floor)

        no_collision = self.space.add_collision_handler(NO_COLLISION_TYPE, NO_COLLISION_TYPE)
        no_collision.begin = lambda a, b, c: False
        ghost_collision = self.space.add_wildcard_collision_handler(GHOST_TYPE)
        ghost_collision.begin = lambda a, b, c: False

        for i, anchor_point in enumerate(arm_anchor_points):
            self.__add_arm(arm_segment_lengths[i], anchor_point)

        target_size = (target_size, target_size)
        target_mass = TARGET_MASS
        target_moment = pymunk.moment_for_box(target_mass, target_size)
        self.target_body = Body(target_mass, target_moment)
        self.target_body.position = ENV_SIZE / 2, target_size[1] / 2 + int(0.187 * ENV_SIZE)
        target_shape = Poly.create_box(self.target_body, target_size)
        target_shape.elasticity = 0.1
        target_shape.friction = 1.0
        self.space.add(self.target_body, target_shape)

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
            segment_shape.elasticity = 0.1

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
            motor.max_force = MAX_MOTOR_FORCE
            self.space.add(motor)
            self.motors.append(motor)
            self.segment_bodies.append(segment_body)
            segment_anchor = segment_body

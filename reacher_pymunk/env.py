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
TARGET_POS = (400, 400)
TARGET_MASS = 10
SEG_LENGTH_RANGE = (20, 200)
TARGET_RANGE = [(50, 550), (50, 550)]
ANCHOR_POINT = (300, 300)
LENGTHS = (150, 150)
TARGET_SIZE = 100
SPARSE_DISTANCE = 0.03


class ActionSpace(Enum):
    CONTINUOUS = 0
    DISCRETE = 1


class MotorActionType(Enum):
    RATE = 0
    RATE_FORCE = 1


class LinkMode(Enum):
    FIXED = 0
    RANDOM = 1
    OPTIMAL = 2
    GENERAL_OPTIMAL = 3


class Reacher(Env):
    def __init__(self, arm_anchor_points=ANCHOR_POINT, arm_segment_lengths=LENGTHS,
                 target=TARGET_POS, seg_length_range=SEG_LENGTH_RANGE, target_range=None,
                 granularity=None, render=False, sparse=False, link_mode=LinkMode.RANDOM,
                 motor_action_type=MotorActionType.RATE_FORCE):
        self.do_render = render
        self.sparse = sparse
        self.granularity = granularity
        self.link_mode = link_mode
        self.motor_action_type = motor_action_type
        self.simulation = ReacherSimulation(arm_anchor_points, arm_segment_lengths, target, do_render=render,
                                            sparse=self.sparse)
        self.arm_anchor_points = arm_anchor_points
        self.arm_segment_lengths = arm_segment_lengths
        self.seg_length_range = seg_length_range
        self.target_range = target_range
        self.target = target
        self.steps = 0
        self.curiosity_module = None
        self.previous_observation = None

        self.n_joints = len(arm_segment_lengths)
        self.length_actions = link_mode == LinkMode.OPTIMAL or link_mode == LinkMode.GENERAL_OPTIMAL
        seg_length_actions = self.n_joints if self.length_actions else 0
        joint_actions = 2 if self.motor_action_type == MotorActionType.RATE_FORCE else 1
        if self.granularity is None:
            self.action_space = spaces.Box(low=0., high=1., shape=(self.n_joints * joint_actions + seg_length_actions,),
                                           dtype=np.float32)
            self.action_space_type = ActionSpace.CONTINUOUS
        else:
            self.action_space = spaces.MultiDiscrete([granularity] * (self.n_joints * joint_actions + seg_length_actions))
            self.action_space_type = ActionSpace.DISCRETE
        self.observation_space = spaces.Box(low=0.,
                                            high=1.,
                                            shape=(4 + 3 * self.n_joints + (1 if self.length_actions else 0),),
                                            dtype=np.float32)
        self.ext_reward_history = []
        self.curiosity_loss_history = []

    def add_curiosity_module(self, curiosity_module):
        self.curiosity_module = curiosity_module

    def reset(self):
        self.steps = 0
        if self.seg_length_range is not None and self.link_mode == LinkMode.RANDOM:
            self.arm_segment_lengths = [random.randint(self.seg_length_range[0], self.seg_length_range[1])
                                        for _ in self.arm_segment_lengths]

        if self.target_range is not None:
            self.target = (random.randint(self.target_range[0][0], self.target_range[0][1]),
                           random.randint(self.target_range[1][0], self.target_range[1][1]))

        self.simulation = ReacherSimulation(self.arm_anchor_points, self.arm_segment_lengths, self.target,
                                            do_render=self.do_render, sparse=self.sparse)

        return self.__get_observation()

    def step(self, action):
        multiplier = 1 if self.action_space_type == ActionSpace.CONTINUOUS else 1. / self.granularity
        denormalized_action = self.__denormalize_action(action * multiplier)
        if not self.length_actions or self.steps > 0:
            self.simulation.set_motor_rates(denormalized_action[0:self.n_joints])
            if self.motor_action_type == MotorActionType.RATE_FORCE:
                self.simulation.set_motor_max_forces(denormalized_action[self.n_joints:2 * self.n_joints])
            self.simulation.step()
        else:
            self.arm_segment_lengths = denormalized_action[-2:]
            self.simulation = ReacherSimulation(self.arm_anchor_points, self.arm_segment_lengths, self.target,
                                                do_render=self.do_render, sparse=self.sparse)

        obs = self.__get_observation()
        distance = np.linalg.norm(np.array(obs[0:2]) - np.array(obs[2:4]))
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
        denormalized_action = [rate * MAX_MOTOR_RATE * 2 - MAX_MOTOR_RATE for rate in action[0:self.n_joints]]
        if self.motor_action_type == MotorActionType.RATE_FORCE:
            denormalized_action += [force * (MAX_MOTOR_FORCE - MIN_MOTOR_FORCE) + MIN_MOTOR_FORCE
                                    for force in action[self.n_joints:2 * self.n_joints]]
        if self.length_actions and self.steps == 0:
            links = [l * (SEG_LENGTH_RANGE[1] - SEG_LENGTH_RANGE[0]) + SEG_LENGTH_RANGE[0] for l in
                     action[-self.n_joints:]]
            denormalized_action += links
        return denormalized_action

    def __get_observation(self):
        targ_pos_x, targ_pos_y = [coord / ENV_SIZE for coord in self.simulation.target_position]
        end_pos_x, end_pos_y = [coord / ENV_SIZE for coord in self.simulation.end_effector_body.position]
        segment_positions = [b.angle / (2 * math.pi) for b in self.simulation.segment_bodies]
        segment_angvs = [(seg.angular_velocity + 4) / 8 for seg in self.simulation.segment_bodies]
        segment_lengths = [(seg - SEG_LENGTH_RANGE[0]) / (SEG_LENGTH_RANGE[1] - SEG_LENGTH_RANGE[0])
                           for seg in self.arm_segment_lengths]

        obs = [targ_pos_x, targ_pos_y, end_pos_x, end_pos_y] + segment_positions + segment_lengths + segment_angvs
        if self.length_actions:
            if self.steps == 0:
                obs[-2 * len(segment_angvs):-len(segment_angvs)] = [0] * len(segment_angvs)
                obs.append(0)
                if self.link_mode == LinkMode.GENERAL_OPTIMAL:
                    obs[0:2] = [0] * 2
            else:
                obs.append(1)

        return np.array(obs)

    def __track_info(self, curiosity_loss, ext_reward):
        self.curiosity_loss_history.append(curiosity_loss)
        self.ext_reward_history.append(ext_reward)

    def get_infos(self):
        info = (sum(self.curiosity_loss_history) / len(self.curiosity_loss_history),
                sum(self.ext_reward_history) / len(self.ext_reward_history))
        self.curiosity_loss_history = []
        self.ext_reward_history = []
        return info


class ReacherSimulation:
    def __init__(self, arm_anchor_point, arm_segment_lengths, target_position, do_render=False, sparse=True):
        self.do_render = do_render
        self.sparse = sparse
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
        self.space.gravity = (0.0, 0.0)

        self.end_effector_body = None
        self.__add_arm(arm_segment_lengths, arm_anchor_point)

        no_collision = self.space.add_collision_handler(NO_COLLISION_TYPE, NO_COLLISION_TYPE)
        no_collision.begin = lambda a, b, c: False

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
        target_radius = ENV_SIZE * SPARSE_DISTANCE
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

    def __add_arm(self, segment_lengths, anchor_position):
        anchor = pymunk.Body(body_type=pymunk.Body.STATIC)
        anchor.position = anchor_position
        self.space.add(anchor)

        segment_anchor = anchor
        for i, segment_length in enumerate(segment_lengths):
            segment_size = segment_length, 10
            segment_body = pymunk.Body(10, pymunk.moment_for_box(10, segment_size))
            segment_body.angle = math.pi / 2

            end_effector_shape = pymunk.Poly.create_box(segment_body, segment_size)
            end_effector_shape.collision_type = NO_COLLISION_TYPE
            end_effector_shape.friction = 1.0
            end_effector_shape.elasticity = 0.1

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

            self.space.add(segment_body, end_effector_shape)

            motor = pymunk.SimpleMotor(segment_anchor, segment_body, 0)
            motor.max_force = MAX_MOTOR_FORCE
            self.space.add(motor)
            self.motors.append(motor)
            self.segment_bodies.append(segment_body)
            segment_anchor = segment_body

        self.end_effector_body = pymunk.Body(1, 1)
        self.end_effector_body.position = anchor.position[0], segment_anchor.position[1] - segment_lengths[-1] / 2
        end_effector_shape = Circle(self.end_effector_body, 8)
        end_effector_shape.collision_type = NO_COLLISION_TYPE
        self.space.add(self.end_effector_body, end_effector_shape)

        pin = pymunk.PinJoint(segment_anchor, self.end_effector_body, (-segment_lengths[-1] / 2, 0), (0, 0))

        self.space.add(pin)

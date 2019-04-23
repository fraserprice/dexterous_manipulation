import math
import random
from abc import abstractmethod, ABC
from functools import reduce

import numpy as np
import pymunk
from pygame.color import THECOLORS
from pymunk import Body, Poly
import pygame
from gym import spaces

from common.constants import RewardType, LinkMode, MotorActionType, ActionSpace
from common.pymunk_env import PymunkEnv, PymunkSimulation

NO_COLLISION_TYPE = 10
GHOST_TYPE = 11
ENV_SIZE = 600
MAX_MOTOR_RATE = 4
MIN_MOTOR_FORCE = 2000000
MAX_MOTOR_FORCE = 10000000
EP_LENGTH = 300
TARGET_POS = (300, 300)
TARGET_MASS = 5
SEG_LENGTH_RANGE = (50, 250)
TARGET_RANGE = (50, 50)
ANCHOR_POINTS = ((450, 350), (150, 350))
LENGTHS = ((150, 150, 150), (150, 150, 150))
OBJECT_SIZE = 100
TARGET_SIZE = 115
SPARSE_DISTANCE = 0.3


class Grabber(PymunkEnv):
    def __init__(self, link_mode=LinkMode.RANDOM, arm_anchor_points=ANCHOR_POINTS, arm_segment_lengths=LENGTHS,
                 target=TARGET_POS, target_range=TARGET_RANGE, granularity=None,
                 render=False, reward_type=RewardType.SPARSE, motor_action_type=MotorActionType.RATE_FORCE,
                 square_target=True):
        n_joints = sum(len(arm_segs) for arm_segs in arm_segment_lengths)
        simulation_class = SquareGrabberSimulation if square_target else CircleGrabberSimulation
        super().__init__(render, reward_type, granularity, link_mode, n_joints, motor_action_type, simulation_class)

        self.link_mode = link_mode
        self.square_target = square_target
        self.sparse = reward_type == RewardType.SPARSE
        self.motor_action_type = motor_action_type
        self.simulation = None

        self.arm_anchor_points = arm_anchor_points
        self.arm_segment_lengths = arm_segment_lengths
        self.target_range = target_range
        self.target = target
        self.target_rotation = 0

        self.min_seg = SEG_LENGTH_RANGE[0]
        self.max_seg = SEG_LENGTH_RANGE[1]
        self.min_force = MIN_MOTOR_FORCE
        self.max_force = MAX_MOTOR_FORCE
        self.max_rate = MAX_MOTOR_RATE

        self.observation_space = spaces.Box(low=0.,
                                            high=1.,
                                            shape=(9 + 2 * self.n_joints,),
                                            dtype=np.float32)

    def set_evaluation_design(self, design):
        self.arm_segment_lengths = ((design['l11'], design['l12'], design['l13']),
                                    (design['l21'], design['l22'], design['l23']))

    def reset(self):
        self.steps = 0
        if self.link_mode == LinkMode.RANDOM and not self.evaluation_mode:
            lengths = []
            for arm_segs in self.arm_segment_lengths:
                lengths.append([random.randint(self.min_seg, self.max_seg) for _ in arm_segs])
            self.arm_segment_lengths = lengths

        if self.target_range is not None:
            self.target = (TARGET_POS[0] + random.randint(-self.target_range[0], self.target_range[0]),
                           TARGET_POS[1] + random.randint(-self.target_range[1], self.target_range[1]))

        target_rotation = random.random() * 2 * np.pi if self.square_target else 0
        self.simulation = self.simulation_class(self.arm_anchor_points, self.arm_segment_lengths, self.target,
                                                target_rotation, self.do_render)

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
            arm_size = 3
            self.arm_segment_lengths = denormalized_action[-self.n_joints:] if arm_size is None else \
                [denormalized_action[-self.n_joints + i:-self.n_joints + i + arm_size]
                 for i in range(int(self.n_joints / arm_size))]
            self.simulation = self.simulation_class(self.arm_anchor_points, self.arm_segment_lengths,
                                                    self.simulation.target_position,
                                                    self.simulation.target_rotation, self.do_render)

        obs = self.get_observation()

        distance = np.linalg.norm(np.array(self.target) - np.array((obs[0] * ENV_SIZE, obs[1] * ENV_SIZE))) * 0.01
        if self.sparse:
            reward = 10 if self.simulation.within_target() else 0
        else:
            reward = -distance
        done = self.steps > EP_LENGTH

        reward = self.get_curiosity_reward(reward, action, obs)

        return obs, reward, done, {'curiosity_loss': self.curiosity_loss_history, 'ext_reward': self.ext_reward_history}

    def get_observation(self):
        obj_pos = self.simulation.object_body.position
        obj_ang = self.simulation.object_body.angle
        obj_v = self.simulation.object_body.velocity
        obj_angv = self.simulation.object_body.angular_velocity
        segment_positions = [b.angle for b in self.simulation.segment_bodies]
        segment_lengths = [seg for arm_segs in self.arm_segment_lengths for seg in arm_segs]
        targ_pos = self.simulation.target_position
        targ_ang = self.simulation.target_rotation or 0

        observation = [obj_pos[0], obj_pos[1], obj_ang, obj_v[0], obj_v[1], obj_angv] \
                      + segment_positions + segment_lengths + [targ_pos[0], targ_pos[1], targ_ang]
        return self.standardize_observation(observation)


class GrabberSimulation(PymunkSimulation, ABC):
    def __init__(self, arm_anchor_points, arm_segment_lengths, target_position, target_rotation=0, do_render=False,
                 sparse=True, object_size=OBJECT_SIZE):
        super().__init__(do_render, sparse, MAX_MOTOR_FORCE)
        self.target_size = TARGET_SIZE
        self.target_position = target_position
        self.target_rotation = target_rotation
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

        self.object_size = (object_size, object_size)
        object_mass = TARGET_MASS
        object_moment = pymunk.moment_for_box(object_mass, self.object_size)
        self.object_body = Body(object_mass, object_moment)
        self.object_body.position = ENV_SIZE / 2, object_size / 2 + int(0.187 * ENV_SIZE)
        self.object_shape = Poly.create_box(self.object_body, self.object_size)
        self.object_shape.elasticity = 0.1
        self.object_shape.friction = 1.0
        self.space.add(self.object_body, self.object_shape)

    @abstractmethod
    def render(self):
        return NotImplemented

    @abstractmethod
    def within_target(self):
        return NotImplemented


class SquareGrabberSimulation(GrabberSimulation):
    def __init__(self, arm_anchor_points, arm_segment_lengths, target_position, target_rotation, do_render=False,
                 object_size=OBJECT_SIZE):
        super().__init__(arm_anchor_points, arm_segment_lengths, target_position, target_rotation, do_render, True,
                         object_size)

    def render(self):
        pygame.event.get()
        self.screen.fill(THECOLORS["white"])
        within_target = self.within_target()
        colour = THECOLORS["green"] if within_target else THECOLORS["red"]
        self.draw_options.draw_polygon(get_square_vertices(self.target_position, self.target_rotation, TARGET_SIZE),
                                       0, colour, colour)
        self.space.debug_draw(self.draw_options)

        pygame.display.flip()
        self.clock.tick(50)
        pygame.display.set_caption("fps: " + str(self.clock.get_fps()))

    def within_target(self):
        target_vertices = get_square_vertices(self.target_position, self.target_rotation, TARGET_SIZE)
        object_vertices = get_square_vertices(self.object_body.position, self.object_body.angle, self.object_size[0])
        for point in object_vertices:
            if not point_inside_square(point, target_vertices):
                return False
        return True


class CircleGrabberSimulation(GrabberSimulation):
    def __init__(self, arm_anchor_points, arm_segment_lengths, target_position, target_rotation, do_render=False,
                 object_size=OBJECT_SIZE):
        super().__init__(arm_anchor_points, arm_segment_lengths, target_position, target_rotation, do_render, True,
                         object_size)

    def render(self):
        pygame.event.get()
        self.screen.fill(THECOLORS["white"])
        square_reach = math.sqrt(((self.object_size[0] / 2) ** 2) * 2)
        target_radius = 100 * SPARSE_DISTANCE + square_reach
        if self.sparse:
            colour = THECOLORS["green"] if self.within_target() else THECOLORS["red"]
            self.draw_options.draw_circle(self.target_position, 0, target_radius, colour, colour)
        else:
            distance = np.linalg.norm(self.object_body.position - self.target_position)
            col = max(0, 255 - 1.5 * distance)
            colour = (255 - col, col, 0, 255)
            self.draw_options.draw_dot(target_radius, self.target_position, colour)
        self.space.debug_draw(self.draw_options)

        pygame.display.flip()
        self.clock.tick(50)
        pygame.display.set_caption("fps: " + str(self.clock.get_fps()))

    def within_target(self):
        square_reach = math.sqrt(((self.object_size[0] / 2) ** 2) * 2)
        target_radius = 100 * SPARSE_DISTANCE + square_reach
        return (self.object_body.position[0] - self.target_position[0]) ** 2 + \
               (self.object_body.position[1] - self.target_position[1]) ** 2 < \
               (target_radius - square_reach) ** 2


def point_inside_square(point, square_vertices):
    point = np.array(point)
    vs = np.array(square_vertices)
    am = np.array([point[0] - vs[0][0], point[1] - vs[0][1]])
    ab = np.array([vs[1][0] - vs[0][0], vs[1][1] - vs[0][1]])
    ad = np.array([vs[3][0] - vs[0][0], vs[3][1] - vs[0][1]])
    one = 0 <= np.dot(am, ab) <= np.dot(ab, ab)
    two = 0 <= np.dot(am, ad) <= np.dot(ad, ad)
    return one and two


def get_square_vertices(position, rotation, size):
    hh0 = np.ones(2) * size / 2
    vv = [np.asarray(position) + reduce(np.dot, [rotation_matrix(rotation), rotation_matrix(np.pi / 2 * c), hh0])
          for c in range(4)]
    return np.asarray(vv)


def rotation_matrix(alpha):
    """ Rotation Matrix for angle ``alpha`` """
    sa, ca = np.sin(alpha), np.cos(alpha)
    return np.array([[ca, -sa],
                     [sa, ca]])

# Python imports
import math
from enum import Enum

import numpy as np

from gym import Env, spaces


class ActionSpace(Enum):
    CONTINUOUS = 0
    DISCRETE = 1


class LinkMode(Enum):
    FIXED = 0
    RANDOM = 1
    ACTIONABLE = 2
    OPTIMAL = 3


class AngleMode(Enum):
    INSTANT = 0
    SIM = 1


class RewardMode(Enum):
    LINEAR = 0
    INVERSE = 1


class EnvType(Enum):
    REACHER_3_DOF = 0
    MULTI_TARG_2_DOF = 1


DOF = {
    EnvType.REACHER_3_DOF: 3,
    EnvType.MULTI_TARG_2_DOF: 2
}
FIXED_LINK_LENGTHS = {
    EnvType.REACHER_3_DOF: np.array([0.2, 0.15, 0.1]),
    EnvType.MULTI_TARG_2_DOF: np.array([0.2, 0.15])
}
MAX_LINK_LENGTH = 0.15
MAX_JOINT_VELOCITY = 2 * np.pi / 100
EPSILON = 2 * np.pi / 1000


class SimpleRobot2D(Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, action_granularity=360, target_distance=0.001, episode_length=100,
                 link_mode=LinkMode.FIXED,
                 angle_mode=AngleMode.INSTANT,
                 reward_mode=RewardMode.INVERSE,
                 env_type=EnvType.REACHER_3_DOF):
        super(SimpleRobot2D, self).__init__()

        self.link_mode = link_mode
        self.angle_mode = angle_mode
        self.reward_mode = reward_mode
        self.env_type = env_type
        self.action_granularity = action_granularity
        self.target_distance = target_distance
        self.episode_length = episode_length

        self.dof = DOF[self.env_type]

        self.n_steps = 0
        self.link_lengths = None
        self.current_joint_angles = None
        self.target_position = None
        self.target_2_position = None

        # Variables for sim mode
        self.joint_velocities = None
        self.max_joint_velocity = 2 * np.pi / 100

        length_actions = self.link_mode == LinkMode.ACTIONABLE or self.link_mode == LinkMode.OPTIMAL
        multiplier = 2 if length_actions else 1
        if action_granularity == -1:
            self.action_space = spaces.Box(low=0., high=1., shape=(self.dof * multiplier,), dtype=np.float32)
            self.action_space_type = ActionSpace.CONTINUOUS
        else:
            self.action_space = spaces.MultiDiscrete([action_granularity] * (self.dof * multiplier))
            self.action_space_type = ActionSpace.DISCRETE
        # self.observation_space = spaces.Box(low=0., high=1., shape=(8,), dtype=np.float32)  # positions
        self.observation_space = spaces.Box(low=0., high=1., shape=(18 if self.env_type == EnvType.REACHER_3_DOF else 15,),
                                            dtype=np.float32)  # angles + velocities + link lengths + positions

        # Rendering stuff
        self.screen_size = 500
        self.viewer = None
        self.j1_trans = None
        self.j2_trans = None
        self.j3_trans = None
        self.end_trans = None
        self.target_trans = None
        self.target_2_trans = None
        self.l1_trans = None
        self.l2_trans = None
        self.l3_trans = None
        self.link1 = None
        self.link2 = None
        self.link3 = None
        self.reset_viewer = True
        self.valid_space = None

        self.reset()

    def step(self, action):
        self.n_steps += 1

        multiplier = 1 if self.action_space_type == ActionSpace.CONTINUOUS else 1. / self.action_granularity

        if self.angle_mode == AngleMode.INSTANT:
            self.current_joint_angles = action[0:3] * 2 * np.pi * multiplier
        else:
            diff = 0.5 if self.action_space_type == ActionSpace.CONTINUOUS else self.action_granularity / 2
            self.joint_velocities = (action[0:3] - diff) * 2 * MAX_JOINT_VELOCITY * multiplier
            self.current_joint_angles = (self.joint_velocities + self.current_joint_angles) % (2 * np.pi)

        if self.link_mode == LinkMode.ACTIONABLE or self.link_mode == LinkMode.OPTIMAL and self.link_lengths is None:
            self.link_lengths = action[3:6] * MAX_LINK_LENGTH * multiplier

        _, j2, j3, end = self.__compute_joint_positions()
        if self.env_type == EnvType.REACHER_3_DOF:
            distance_to_target = np.linalg.norm(end - self.target_position)
        else:
            distance_to_target = np.linalg.norm(j3, self.target_2_position) + np.linalg.norm(j2, self.target_position)
        reward = -distance_to_target if self.reward_mode == RewardMode.LINEAR else min(1 / (20 * distance_to_target),
                                                                                        10)
        # if self.angle_mode == AngleMode.SIM:
        #     reward -= np.abs(self.joint_velocities).sum()
        done = self.n_steps >= self.episode_length

        return self.__get_observation(), reward, done, {"distance": distance_to_target}

    def reset(self):
        self.n_steps = 0
        if self.link_mode == LinkMode.RANDOM or self.link_mode == LinkMode.ACTIONABLE:
            self.link_lengths = np.random.uniform(0, MAX_LINK_LENGTH, size=(3,))
        elif self.link_mode == LinkMode.FIXED:
            self.link_lengths = FIXED_LINK_LENGTHS[self.env_type]
        elif self.link_mode == LinkMode.OPTIMAL:
            self.link_lengths = None

        if self.link_mode == LinkMode.ACTIONABLE or self.link_mode == LinkMode.OPTIMAL:
            self.target_position = uniform_sphere(MAX_LINK_LENGTH * 3)
            self.target_2_position = uniform_sphere(sum(self.link_lengths))
        else:
            self.target_position = uniform_sphere(sum(self.link_lengths))
            self.target_2_position = uniform_sphere(sum(self.link_lengths))

        if self.angle_mode == AngleMode.SIM:
            self.joint_velocities = np.array([0, 0, 0], dtype=np.float32)

        self.current_joint_angles = np.array([0, 0, 0], dtype=np.float32)
        self.reset_viewer = True

        return self.__get_observation()

    def render(self, mode='human', close=False):
        joint_1, joint_2, joint_3, robot_end = self.__compute_joint_positions()

        # Then, compute the coordinates of the joints in screen space
        joint_1_screen = [int((0.5 + joint_1[0]) * self.screen_size),
                          self.screen_size - int((0.5 - joint_1[1]) * self.screen_size)]
        joint_2_screen = [int((0.5 + joint_2[0]) * self.screen_size),
                          self.screen_size - int((0.5 - joint_2[1]) * self.screen_size)]
        joint_3_screen = [int((0.5 + joint_3[0]) * self.screen_size),
                          self.screen_size - int((0.5 - joint_3[1]) * self.screen_size)]
        robot_end_screen = [int((0.5 + robot_end[0]) * self.screen_size),
                            self.screen_size - int((0.5 - robot_end[1]) * self.screen_size)]
        target_screen = [int((0.5 + self.target_position[0]) * self.screen_size),
                         self.screen_size - int((0.5 - self.target_position[1]) * self.screen_size)]
        target_2_screen = [int((0.5 + self.target_2_position[0]) * self.screen_size),
                           self.screen_size - int((0.5 - self.target_2_position[1]) * self.screen_size)]

        from gym.envs.classic_control import rendering

        link_lengths = [0, 0, 0] if self.link_lengths is None else self.link_lengths

        if self.viewer is None:
            self.viewer = rendering.Viewer(self.screen_size, self.screen_size)

            self.j1_trans = rendering.Transform()
            joint1 = rendering.make_circle(4)
            joint1.add_attr(self.j1_trans)

            self.j2_trans = rendering.Transform()
            joint2 = rendering.make_circle(4)
            joint2.add_attr(self.j2_trans)

            self.j3_trans = rendering.Transform()
            if self.dof == 2:
                joint3 = rendering.make_circle(2)
                joint3.set_color(0, 0, 1)
            else:
                joint3 = rendering.make_circle(4)
            joint3.add_attr(self.j3_trans)

            self.end_trans = rendering.Transform()
            end = rendering.make_circle(2)
            end.set_color(0, 0, 1)
            end.add_attr(self.end_trans)

            self.target_trans = rendering.Transform()
            target = rendering.make_circle(self.screen_size * self.target_distance, filled=False)
            target.set_color(1, 0, 0)
            target.add_attr(self.target_trans)

            self.target_2_trans = rendering.Transform()
            target_2 = rendering.make_circle(self.screen_size * self.target_distance, filled=False)
            target_2.set_color(1, 0, 0)
            target_2.add_attr(self.target_2_trans)

            self.l1_trans = rendering.Transform()
            self.link1 = rendering.Line([0, 0], [0, link_lengths[0] * self.screen_size])
            self.link1.add_attr(self.l1_trans)

            self.l2_trans = rendering.Transform()
            self.link2 = rendering.Line([0, 0], [0, link_lengths[1] * self.screen_size])
            self.link2.add_attr(self.l2_trans)

            self.l3_trans = rendering.Transform()
            self.link3 = rendering.Line([0, 0], [0, link_lengths[2] * self.screen_size])
            self.link3.add_attr(self.l3_trans)

            self.viewer.add_geom(self.link1)
            self.viewer.add_geom(self.link2)
            if self.dof > 2:
                self.viewer.add_geom(self.link3)

            self.viewer.add_geom(joint1)
            self.viewer.add_geom(joint2)
            self.viewer.add_geom(joint3)
            if self.dof > 2:
                self.viewer.add_geom(end)

            self.viewer.add_geom(target)
            if self.dof == 2:
                self.viewer.add_geom(target_2)

        if self.reset_viewer:
            self.link1.start, self.link1.end = [0, 0], [0, link_lengths[0] * self.screen_size]
            self.link2.start, self.link2.end = [0, 0], [0, link_lengths[1] * self.screen_size]
            self.link3.start, self.link3.end = [0, 0], [0, link_lengths[2] * self.screen_size]

        self.j1_trans.set_translation(joint_1_screen[0], joint_1_screen[1])
        self.j2_trans.set_translation(joint_2_screen[0], joint_2_screen[1])
        self.j3_trans.set_translation(joint_3_screen[0], joint_3_screen[1])
        self.end_trans.set_translation(robot_end_screen[0], robot_end_screen[1])
        self.target_trans.set_translation(target_screen[0], target_screen[1])
        self.target_2_trans.set_translation(target_2_screen[0], target_2_screen[1])

        self.l1_trans.set_translation(joint_1_screen[0], joint_1_screen[1])
        self.l2_trans.set_translation(joint_2_screen[0], joint_2_screen[1])
        self.l3_trans.set_translation(joint_3_screen[0], joint_3_screen[1])

        self.l1_trans.set_rotation(self.current_joint_angles[0])
        self.l2_trans.set_rotation(self.current_joint_angles[1] + self.current_joint_angles[0])
        self.l3_trans.set_rotation(
            self.current_joint_angles[2] + self.current_joint_angles[1] + self.current_joint_angles[0])

        self.valid_space = rendering.make_circle(self.screen_size * sum(self.link_lengths), filled=False, res=50)
        translation = rendering.Transform()
        translation.set_translation(0.5 * self.screen_size, 0.5 * self.screen_size)
        self.valid_space.add_attr(translation)
        self.viewer.add_onetime(self.valid_space)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def __get_observation(self):
        # _, (j2x, j2y), (j3x, j3y), (endx, endy) = self.__compute_joint_positions()
        # tx, ty = self.target_position
        # return np.array([j2x, j2y, j3x, j3y, endx, endy, tx, ty])
        a1, a2, a3 = self.current_joint_angles / (2 * np.pi)
        v1, v2, v3 = (self.joint_velocities + MAX_JOINT_VELOCITY) / (2 * MAX_JOINT_VELOCITY)
        l1, l2, l3 = self.link_lengths / max(MAX_LINK_LENGTH,
                                             max(self.link_lengths)) if self.link_lengths is not None else (0, 0, 0)
        r = 1 if self.link_lengths is None else 3 * MAX_LINK_LENGTH
        f = lambda c: (r + c) / (2 * r)
        _, (j2x, j2y), (j3x, j3y), (endx, endy) = self.__compute_joint_positions()
        _, (j2x, j2y), (j3x, j3y), (endx, endy) = _, (f(j2x), f(j2y)), (f(j3x), f(j3y)), (f(endx), f(endy))
        tx, ty = self.target_position
        tx, ty = f(tx), f(ty)
        t2x, t2y = self.target_2_position
        t2x, t2y = f(t2x), f(t2y)
        joints_set = 0 if self.link_lengths is None else 1
        if self.env_type == EnvType.REACHER_3_DOF:
            obs = np.array([a1, a2, a3,
                            v1, v2, v3,
                            l1, l2, l3,
                            j2x, j2y, j3x, j3y, endx, endy,
                            tx, ty,
                            joints_set])
        else:
            obs = np.array([a1, a2,
                            v1, v2,
                            l1, l2,
                            j2x, j2y, j3x, j3y,
                            tx, ty,
                            t2x, t2y,
                            joints_set])
        return obs

    def __compute_joint_positions(self):
        if self.link_lengths is None:
            return tuple([np.array(point) for point in [[0, 0]] * 4])
        trans_10 = compute_transformation(0, self.current_joint_angles[0])
        trans_21 = compute_transformation(self.link_lengths[0], self.current_joint_angles[1])
        trans_32 = compute_transformation(self.link_lengths[1], self.current_joint_angles[2])

        # Then, compute the coordinates of the joints in world space
        joint_1 = np.dot(trans_10, np.array([0, 0, 1]))[0:2]
        joint_2 = np.dot(trans_10, (np.dot(trans_21, np.array([0, 0, 1]))))[0:2]
        joint_3 = np.dot(trans_10, (np.dot(trans_21, np.dot(trans_32, np.array([0, 0, 1])))))[0:2]
        robot_end = np.dot(trans_10, (np.dot(trans_21, np.dot(trans_32, np.array([0, self.link_lengths[2], 1])))))[0:2]

        return joint_1, joint_2, joint_3, robot_end

    def __add_angle_deltas(self, deltas):
        self.current_joint_angles += deltas
        self.current_joint_angles = np.remainder(self.current_joint_angles, 2 * np.pi)


# Function to compute the transformation matrix between two frames, based on the length (in y-direction) along the
# first frame, and the rotation of the second frame with respect to the first frame
def compute_transformation(length, rotation):
    cos_rotation = np.cos(rotation)
    sin_rotation = np.sin(rotation)
    transformation = np.array([[cos_rotation, -sin_rotation, 0], [sin_rotation, cos_rotation, length], [0, 0, 1]])
    return transformation


def uniform_sphere(radius):
    a = np.random.uniform(0, 2 * np.pi)
    r = radius * math.sqrt(np.random.uniform(0, 1))
    return np.array([r * np.cos(a), r * np.sin(a)])

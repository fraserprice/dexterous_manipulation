import time
from enum import Enum

import numpy as np

from gym import Env, spaces

from vrep_grabber.utils.vrep_interface import VrepInterface


class ActionType(Enum):
    CONTINUOUS = 0
    DISCRETE = 1


class EnvType(Enum):
    DOF3 = 3
    DOF6 = 6
    DOF9 = 9


SCENE_FOLDER = "/Users/fraser/Documents/University/Fourth Year/Dexterous Manipulation/Dexterous Manipulation/3D Model"
scene_paths = {
    EnvType.DOF3: f"{SCENE_FOLDER}/3dof_grabber.ttt",
    EnvType.DOF6: f"{SCENE_FOLDER}/6dof_grabber.ttt",
    EnvType.DOF9: f"{SCENE_FOLDER}/9dof_grabber.ttt"
}

VREP_OBJECT_NAMES = ["a0j0", "a0j1", "a0l0", "a0j2", "a0l1",
                     "a1j0", "a1j1", "a1l0", "a1j2", "a1l1",
                     "a2j0", "a2j1", "a2l0", "a2j2", "a2l1",
                     "base", "target"]
VREP_JOINT_NAMES = {
    EnvType.DOF3: ["a0j2", "a1j2", "a2j2"],
    EnvType.DOF6: ["a0j1", "a0j2", "a1j1", "a1j2", "a2j1", "a2j2"],
    EnvType.DOF9: ["a0j0", "a0j1", "a0j2", "a1j0", "a1j1", "a1j2", "a2j0", "a2j1", "a2j2"],
}

VREP_COLLECTION_NAMES = ['joints']
JOINT_COLLECTION = VREP_COLLECTION_NAMES[0]

MAX_JOINT_ANGLE = np.pi / 2


class VrepGrabber(Env):
    def __init__(self, grabber_type, vrep_port, action_granularity=None, episode_length=15):
        super(VrepGrabber, self).__init__()

        self.port = vrep_port
        self.action_granularity = action_granularity
        self.action_type = ActionType.CONTINUOUS if action_granularity is None else ActionType.DISCRETE
        self.grabber_type = grabber_type
        self.dof = self.grabber_type.value

        if self.action_type == ActionType.CONTINUOUS:
            self.action_space = spaces.Box(low=0., high=1., shape=(self.dof,), dtype="float32")
        else:
            self.action_space = spaces.MultiDiscrete([self.action_granularity] * self.dof)

        self.observation_space = spaces.Box(low=0., high=1., shape=(12 + self.dof,), dtype="float32")

        self.vrep_interface = VrepInterface()

        print(f"Attempting to make connection on port {vrep_port}")
        if not self.vrep_interface.connect_to_vrep(vrep_port):
            print(f"Failed to connect to port {vrep_port}")
        else:
            print(f"Successfully connected to VREP on port {vrep_port}. Client id: {self.vrep_interface.client_id}")

        print("Loading scene...")
        if not self.vrep_interface.load_scene(scene_paths[grabber_type], VREP_OBJECT_NAMES, VREP_COLLECTION_NAMES):
            print(f"Failed to load scene on port {vrep_port}")
        else:
            print(f"Successfully loaded scene on port {vrep_port}")

        self.episode_length = episode_length
        self.n_steps = 0

        self.time = 0

    def step(self, action):
        if self.n_steps == 0:
            self.time = time.time()
            self.vrep_interface.start_simulation()
        else:
            self.vrep_interface.step_simulation()
        self.n_steps += 1

        for i, joint_name in enumerate(VREP_JOINT_NAMES[self.grabber_type]):
            if self.action_type == ActionType.CONTINUOUS:
                denormalized_angle = denormalize_angle(action[i])
            else:
                denormalized_angle = denormalize_discrete_angle(action[i], self.action_granularity)
            self.vrep_interface.set_joint_target_angle(joint_name, denormalized_angle)

        obs = self.__get_observation()

        tx, ty, tz = [denormalize_position(coordinate) for coordinate in obs[0:3]]

        done = self.n_steps >= self.episode_length

        reward = 10 * -abs(0.2 - tz)  # - 0.05 * (abs(ty) + abs(tx)

        return obs, reward, done, {}

    def reset(self):
        if self.n_steps > 0:
            self.vrep_interface.stop_simulation()
            self.n_steps = 0

        return self.__get_observation()

    def render(self, mode='human', close=False):
        return False

    '''
    Observation space is target object position + orientation, joint angles + velocities
    obs1 = {
        targ_x, targ_y, targ_z, targ_a, targ_b, targ_g,
        j00_r, j01_r, j02_r,
        j10_r, jj11_r, j12_r,
        j20_r, j21_r, j22_r,
    }
    '''

    def __get_observation(self):
        try:
            tx, ty, tz = [normalize_position(p) for p in self.vrep_interface.get_object_position('target')]
            ta, tb, tg = [normalize_radian(a) for a in self.vrep_interface.get_object_orientation('target')]
            lv, av = self.vrep_interface.get_object_velocity('target')
            (vx, vy, vz) = [normalize_lin_velocity(v) for v in lv]
            (va, vb, vg) = [normalize_ang_velocity(v) for v in av]
            observation = [tx, ty, tz, ta, tb, tg, vx, vy, vz, va, vb, vg]

            joint_angles = self.vrep_interface.get_collection_joint_rotations(JOINT_COLLECTION)
            for joint_name in VREP_JOINT_NAMES[self.grabber_type]:
                observation.append(normalize_radian(joint_angles[joint_name]))

            return np.array(observation)
        except (KeyError, TypeError, ConnectionError):
            return self.__get_observation()


def denormalize_angle(normalized_angle):
    return 2 * (normalized_angle - 0.5) * MAX_JOINT_ANGLE


def denormalize_discrete_angle(discrete_action, granularity):
    return 2 * (discrete_action - granularity / 2) * MAX_JOINT_ANGLE / granularity


def normalize_ang_velocity(av):
    return (av + 180) / 360


def normalize_lin_velocity(lv):
    return (lv + 10) / 20


def normalize_radian(angle):
    return (angle + np.pi) / (2 * np.pi)


def normalize_position(pos):
    return (pos + 1) / 2


def denormalize_position(normalized_pos):
    return 2 * normalized_pos - 1


if __name__ == "__main__":
    print(denormalize_discrete_angle(360, 360))

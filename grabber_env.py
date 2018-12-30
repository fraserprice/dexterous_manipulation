import time

import numpy as np

from gym import Env, spaces

from vrep_interface import VrepInterface

VREP_OBJECT_NAMES = ["a0j0", "a0j1", "a0l0", "a0j2", "a0l1",
                     "a1j0", "a1j1", "a1l0", "a1j2", "a1l1",
                     "a2j0", "a2j1", "a2l0", "a2j2", "a2l1",
                     "base", "target"]
VREP_JOINT_NAMES = ["a0j0", "a0j1", "a0j2",
                    "a1j0", "a1j1", "a1j2",
                    "a2j0", "a2j1", "a2j2"]

VREP_COLLECTION_NAMES = ['joints']
JOINT_COLLECTION = VREP_COLLECTION_NAMES[0]
SCENE_PATH = "/Users/fraser/Documents/University/Fourth Year/Dexterous Manipulation/VREP Repo/3D Model/6dof_grabber.ttt"

MAX_JOINT_VELOCITY = 90


class VrepGrabber(Env):
    def __init__(self, scene_path, episode_length=15, headless=True):
        super(VrepGrabber, self).__init__()

        self.vrep_interface = VrepInterface()
        if not self.vrep_interface.connect_to_vrep(scene_path, VREP_OBJECT_NAMES, VREP_COLLECTION_NAMES):
             raise ConnectionError("Failed to connect to VREP")
        # if not self.vrep_interface.start_vrep(SCENE_PATH, VREP_OBJECT_NAMES, VREP_COLLECTION_NAMES, headless=headless):
        #     raise ConnectionError("Failed to start VREP")
        self.episode_length = episode_length
        self.n_steps = 0

        self.time = 0

        # Action space is joint target velocities
        self.action_space = spaces.Box(low=0., high=1., shape=(9,), dtype="float32")

        '''
        Observation space is target object position + orientation, joint angles + velocities
        obs1 = {
            targ_x, targ_y, targ_z, targ_a, targ_b, targ_g,
            j00_r, j01_r, j02_r,
            j10_v, j11_v, j12_v,
            j20_v, j21_v, j22_v,
        }
        '''
        self.observation_space = spaces.Box(low=0., high=1., shape=(21,), dtype="float32")

    def step(self, action):
        if self.n_steps == 0:
            self.time = time.time()
            self.vrep_interface.start_simulation()
        else:
            self.vrep_interface.step_simulation()
        self.n_steps += 1

        for i, joint_name in enumerate(VREP_JOINT_NAMES):
            self.vrep_interface.set_joint_target_velocity(joint_name, denormalizej_velocity(action[i]))

        obs = self.__get_observation()
        tx, ty, tz = [denormalize_position(coordinate) for coordinate in obs[0:3]]
        reward = -abs((0.35 - tz)) #- 0.1 * (abs(ty) + abs(tx))  # if done else 0

        done = self.n_steps >= self.episode_length

        return obs, reward, done, {}

    def reset(self):
        print(self.vrep_interface.vrep.simxGetPingTime(self.vrep_interface.client_id))
        if self.n_steps > 0:
            self.vrep_interface.stop_simulation()
            self.n_steps = 0

        print(time.time() - self.time)

        print("Resetting...")

        return self.__get_observation()

    def render(self, mode='human', close=False):
        return False

    def __get_observation(self):
        try:
            tx, ty, tz = [normalize_position(p) for p in self.vrep_interface.get_object_position('target')]
            ta, tb, tg = [normalize_radian(a) for a in self.vrep_interface.get_object_orientation('target')]
            lv, av = self.vrep_interface.get_object_velocity('target')
            (vx, vy, vz) = [normalize_lin_velocity(v) for v in lv]
            (va, vb, vg) = [normalize_ang_velocity(v) for v in av]
            observation = [tx, ty, tz, ta, tb, tg, vx, vy, vz, va, vb, vg]

            joint_angles = self.vrep_interface.get_collection_joint_rotations(JOINT_COLLECTION)
            for joint_name in VREP_JOINT_NAMES:
                observation.append(normalize_radian(joint_angles[joint_name]))

            return np.array(observation)
        except (KeyError, TypeError):
            self.n_steps = self.episode_length
            self.vrep_interface.reload_scene(SCENE_PATH, VREP_OBJECT_NAMES, VREP_COLLECTION_NAMES)
            return np.array([0 for _ in range(21)])


def denormalizej_velocity(normalized_velocity):
    return 2 * (normalized_velocity - 0.5) * MAX_JOINT_VELOCITY


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
    vri = VrepInterface()
    vg = VrepGrabber()
    vg.step([0.5, 0.5, 0.75, 0.5, 0.5, 0.75, 0.5, 0.5, 0.75])
    time.sleep(0.3)
    vg.step([0.5, 0.5, 0.75, 0.5, 0.5, 0.75, 0.5, 0.5, 0.75])
    time.sleep(0.3)
    vg.step([0.5, 0.5, 0.75, 0.5, 0.5, 0.75, 0.5, 0.5, 0.75])
    time.sleep(0.3)
    vg.step([0.5, 0.5, 0.75, 0.5, 0.5, 0.75, 0.5, 0.5, 0.75])
    time.sleep(0.3)
    vg.step([0.5, 0.5, 0.75, 0.5, 0.5, 0.75, 0.5, 0.5, 0.75])
    time.sleep(0.3)
    vg.step([0.5, 0.5, 0.75, 0.5, 0.5, 0.75, 0.5, 0.5, 0.75])
    time.sleep(0.3)
    vg.step([0.5, 0.5, 0.75, 0.5, 0.5, 0.75, 0.5, 0.5, 0.75])
    time.sleep(0.3)
    vg.step([0.5, 0.5, 0.75, 0.5, 0.5, 0.75, 0.5, 0.5, 0.75])
    time.sleep(0.3)
    vg.step([0.5, 0.5, 0.75, 0.5, 0.5, 0.75, 0.5, 0.5, 0.75])
    time.sleep(0.3)

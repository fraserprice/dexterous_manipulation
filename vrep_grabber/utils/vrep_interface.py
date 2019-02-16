import os
import time
from multiprocessing import Process
from vrep_grabber.utils import vrepConst, vrep

PORT = 19997
VREP_LOCATION = "/Applications/V-REP_PRO_EDU_V3_5_0_Mac/vrep.app/Contents/MacOS/vrep"
BLOCKING = vrepConst.simx_opmode_blocking
OK = vrepConst.simx_return_ok

class VrepInterface:
    def __init__(self):
        self.vrep = vrep
        self.client_id = None
        self.object_handles = {}
        self.object_names = {}
        self.collection_handles = {}
        self.vrep_process = None

    def connect_to_vrep(self, port=PORT):
        if self.__connect(port):
            self.vrep.simxSynchronous(self.client_id, enable=True)
            return True
        return False

    def start_vrep(self, scene_path, object_names, collection_names, port=PORT, timeout=10, headless=True):
        def vrep_process():
            h = " -h" if headless else ""
            os.system(f"{VREP_LOCATION} -gREMOTEAPISERVERSERVICE_{PORT}_FALSE_TRUE{h} {scene_path}")

        self.vrep_process = Process(target=vrep_process)
        self.vrep_process.start()

        for _ in range(timeout):
            if self.__connect(port):
                return self.load_scene(scene_path, object_names, collection_names)
            time.sleep(1)

        return False

    def start_simulation(self):
        self.vrep.simxSynchronous(self.client_id, True)
        return self.vrep.simxStartSimulation(self.client_id, BLOCKING) == OK

    def step_simulation(self):
        return self.vrep.simxSynchronousTrigger(self.client_id)

    def stop_simulation(self):
        if self.vrep.simxStopSimulation(self.client_id, BLOCKING) == OK:
            while True:
                self.vrep.simxGetIntegerSignal(self.client_id, 'asdf', BLOCKING)
                _, res = self.vrep.simxGetInMessageInfo(self.client_id, vrepConst.simx_headeroffset_server_state)
                if not res & 1:
                    return True
        return False

    def load_scene(self, filepath, object_names, collection_names):
        res = self.vrep.simxLoadScene(self.client_id, filepath, 0xFF, BLOCKING)
        if res != OK:
            print("Could not load scene")
            return False
        for collection_name in collection_names:
            res, handle = self.vrep.simxGetCollectionHandle(self.client_id, collection_name,
                                                            BLOCKING)
            if res != OK:
                print(f"Could not find collection handle for {collection_name}")
                return False
            self.collection_handles[collection_name] = handle
        for object_name in object_names:
            res, handle = self.vrep.simxGetObjectHandle(self.client_id, object_name, BLOCKING)
            if res != OK:
                print(f"Could not find obbject handle for {object_name}")
                return False
            self.object_handles[object_name] = handle
            self.object_names[handle] = object_name
        return True

    def get_object_position(self, object_name, reference_object_name=None):
        return self.__get_relative_object_data(object_name, reference_object_name, self.vrep.simxGetObjectPosition)

    def get_object_orientation(self, object_name, reference_object_name=None):
        return self.__get_relative_object_data(object_name, reference_object_name, self.vrep.simxGetObjectOrientation)

    def get_object_velocity(self, object_name):
        res, linear, angular = self.vrep.simxGetObjectVelocity(self.client_id,
                                                               self.object_handles[object_name],
                                                               BLOCKING)
        if res != OK:
            print("Could not retrieve object data for " + object_name)
            return None, None
        else:
            return linear, angular

    def set_joint_target_angle(self, joint_name, target_angle):
        res = self.vrep.simxSetJointTargetPosition(self.client_id,
                                                   self.object_handles[joint_name],
                                                   target_angle,
                                                   BLOCKING)
        if res != OK:
            print(f"Failed to set {joint_name} to {target_angle}")
        return res == OK

    def get_joint_angle(self, joint_name):
        res, position = self.vrep.simxGetJointPosition(self.client_id,
                                                       self.object_handles[joint_name],
                                                       BLOCKING)

        return position if res == OK else -1

    def get_collection_positions(self, collection_name):
        res, handles, _, positions, _ = self.__get_collection_data(collection_name, 3)
        if res == OK:
            results = {}
            for i, handle in enumerate(handles):
                index = i * 3
                x, y, z = positions[index], positions[index + 1], positions[index + 2]
                results[self.object_names[handle]] = (x, y, z)
            return results
        return {}

    def get_collection_orientations(self, collection_name):
        res, handles, _, orientations, _ = self.__get_collection_data(collection_name, 5)
        if res == OK:
            results = {}
            for i, handle in enumerate(handles):
                index = i * 3
                a, b, g = orientations[index], orientations[index + 1], orientations[index + 2]
                results[self.object_names[handle]] = (a, b, g)
            return results
        return {}

    def check_connection(self):
        ret, _ = self.vrep.simxGetPingTime(self.client_id)
        return ret == OK

    def get_collection_joint_rotations(self, collection_name):
        res, handles, _, rotations, _ = self.__get_collection_data(collection_name, 15)
        if res == OK:
            results = {}
            for i, handle in enumerate(handles):
                r = rotations[i * 2]
                results[self.object_names[handle]] = r
            return results
        return {}

    def __get_collection_data(self, collection_name, data_type):
        collection_handle = self.collection_handles[collection_name]
        return self.vrep.simxGetObjectGroupData(self.client_id,
                                                collection_handle,
                                                data_type, BLOCKING)

    def __get_relative_object_data(self, object_name, reference_object_name, obj_data_function):
        reference_object_handle = -1 if reference_object_name is None else self.object_handles[reference_object_name]
        res, (x, y, z) = obj_data_function(self.client_id,
                                           self.object_handles[object_name],
                                           reference_object_handle,
                                           BLOCKING)
        if res != OK:
            print(f"Could not retrieve object data for {object_name}. Ret code: {res}")
            raise ConnectionError
        else:
            return x, y, z

    def __connect(self, port):
        # self.vrep.simxFinish(-1)
        print(f"Making connection on port {port}...")
        self.client_id = self.vrep.simxStart('127.0.0.1', port, True, True, 1000, 5)
        return self.client_id != -1

    def __disconnect(self):
        return self.vrep.simxFinish(self.client_id)

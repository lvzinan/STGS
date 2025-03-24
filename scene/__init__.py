#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks, readColmapBlock, readColmapBlockBatch
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import numpy as np
import random


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2]])


def Scene_init(args: ModelParams):
    txt_path = os.path.join(args.source_path, "sparse/0/images.txt")

    max_x = -1000000
    min_x = 1000000
    max_y = -1000000
    min_y = 1000000
    max_z = -1000000
    min_z = 1000000

    with open(txt_path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                pose = np.concatenate(
                    [np.concatenate([qvec2rotmat(qvec), tvec.reshape(3, 1)], 1), np.array([0, 0, 0, 1]).reshape(1, 4)],
                    0)
                pose = np.linalg.inv(pose)
                r0 = pose[:3, :3]
                xyz = pose[:3, 3]
                max_x = max(float(xyz[0]), max_x)
                max_y = max(float(xyz[1]), max_y)
                max_z = max(float(xyz[2]), max_z)
                min_x = min(float(xyz[0]), min_x)
                min_y = min(float(xyz[1]), min_y)
                min_z = min(float(xyz[2]), min_z)
                line = fid.readline()

    return max_x, min_x, max_y, min_y, max_z, min_z


def Scene_init_point(args: ModelParams, R):
    txt_path = os.path.join(args.source_path, "sparse/0/Points3D.txt")

    max_z = -1000000
    min_z = 1000000

    with open(txt_path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                xyz = np.array(tuple(map(float, elems[5:8])))
                xyz1 = np.dot(R, xyz)
                max_z = max(float(xyz1[2]), max_z)
                min_z = min(float(xyz1[2]), min_z)

    return max_z, min_z


def generate_random_sequence(N):
    sequence = list(range(0, N))
    random.shuffle(sequence)
    return sequence


def split_sequence(sequence, batch_size=250):
    yu_min = 0
    for i in range(batch_size - 50, batch_size):
        yu = len(sequence) % i
        if yu > yu_min:
            batch_size = i
            yu_min = yu
    return [sequence[i:i + batch_size] for i in range(0, len(sequence), batch_size)]


class Scene:
    gaussians: GaussianModel

    def __init__(self, args: ModelParams, gaussians: GaussianModel, load_iteration=None, shuffle=True,
                 resolution_scales=[1.0], train=True, x_min=None, x_max=None, y_min=None, y_max=None, block=None,
                 vis_img=None, R_append=0.0,
                 image_H=0, image_W=0, cam_append=0.25, batch_size=batch_size):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.block = block
        self.batch_size = batch_size

        if not train:

            if load_iteration:
                if load_iteration == -1:
                    self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
                else:
                    self.loaded_iter = load_iteration
                print("Loading trained model at iteration {}".format(self.loaded_iter))

            self.train_cameras = {}
            self.test_cameras = {}

            if os.path.exists(os.path.join(args.source_path, "sparse")):
                scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, train=False)
            elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
                print("Found transforms_train.json file, assuming Blender data set!")
                scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
            else:
                assert False, "Could not recognize scene type!"

            if not self.loaded_iter:
                with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply"),
                                                                       'wb') as dest_file:
                    dest_file.write(src_file.read())
                json_cams = []
                camlist = []
                if scene_info.test_cameras:
                    camlist.extend(scene_info.test_cameras)
                if scene_info.train_cameras:
                    camlist.extend(scene_info.train_cameras)
                for id, cam in enumerate(camlist):
                    json_cams.append(camera_to_JSON(id, cam))
                with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                    json.dump(json_cams, file)

            if shuffle:
                random.shuffle(scene_info.train_cameras)
                random.shuffle(scene_info.test_cameras)

            self.cameras_extent = scene_info.nerf_normalization["radius"]

            for resolution_scale in resolution_scales:
                print("Loading Training Cameras")
                self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras,
                                                                                resolution_scale, args)
                print("Loading Test Cameras")
                self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras,
                                                                               resolution_scale, args)

            if self.loaded_iter:
                self.gaussians.load_ply(os.path.join(self.model_path,
                                                     "point_cloud",
                                                     "iteration_" + str(self.loaded_iter),
                                                     "point_cloud.ply"))
            else:
                self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
        else:

            if load_iteration:
                if load_iteration == -1:
                    self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
                else:
                    self.loaded_iter = load_iteration
                print("Loading trained model at iteration {}".format(self.loaded_iter))

            self.train_cameras = {}
            self.test_cameras = {}
            print("scene_info begin")

            scene_info, uid = readColmapBlock(args.source_path, args.images, args.eval, x_min, x_max, y_min, y_max,
                                              block, train=True, R_append=R_append, cam_append=cam_append)
            print("scene_info end")
            self.uid = uid

            if not self.loaded_iter:
                with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply"),
                                                                       'wb') as dest_file:
                    dest_file.write(src_file.read())
                json_cams = []
                camlist = []
                if scene_info.test_cameras:
                    camlist.extend(scene_info.test_cameras)
                if scene_info.train_cameras:
                    camlist.extend(scene_info.train_cameras)
                for id, cam in enumerate(camlist):
                    json_cams.append(camera_to_JSON(id, cam))
                with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                    json.dump(json_cams, file)

            if shuffle:
                random.shuffle(scene_info.train_cameras)
                random.shuffle(scene_info.test_cameras)

            self.cameras_extent = scene_info.nerf_normalization["radius"]

            self.scene_info = scene_info
            self.train_cam_num = len(self.scene_info.train_cameras)
            print(self.train_cam_num)

            if image_H != 0:
                self.gaussians.create_cam(self.train_cam_num, image_H, image_W)

            random_sequence = generate_random_sequence(self.train_cam_num)
            self.chunks = split_sequence(random_sequence, batch_size=self.batch_size)
            self.num_chunks = 0

            if vis_img == None:
                self.vis_img = []
                for _ in range(self.train_cam_num):
                    self.vis_img.append(True)
            else:
                self.vis_img = vis_img

            if self.loaded_iter:
                self.gaussians.load_ply(os.path.join(self.model_path,
                                                     "point_cloud",
                                                     "iteration_" + str(self.loaded_iter),
                                                     "point_cloud.ply"))
            else:
                self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

            self.num_chunks_test = 0

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud" + str(self.block) + ".ply"))

    def scene_batch(self, args: ModelParams, resolution_scales=[1.0], init=False):
        self.train_cameras = {}
        self.test_cameras = {}
        if self.num_chunks < len(self.chunks):
            pass
        else:
            random_sequence = generate_random_sequence(self.train_cam_num)
            self.chunks = split_sequence(random_sequence, batch_size=self.batch_size)
            self.num_chunks = 0
        scene_info = readColmapBlockBatch(self.scene_info, self.chunks[self.num_chunks], init=init)
        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras,
                                                                            resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras,
                                                                           resolution_scale, args)
        self.num_chunks += 1

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTrainCameras_all(self, args: ModelParams, scale=1.0, resolution_scales=[1.0]):
        self.train_cameras_test = {}
        if self.num_chunks_test < len(self.chunks):
            pass
        else:
            self.num_chunks_test = 0
        scene_info = readColmapBlockBatch(self.scene_info, self.chunks[self.num_chunks_test])
        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras,
                                                                            resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras,
                                                                           resolution_scale, args)
        self.num_chunks_test += 1

        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

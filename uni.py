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
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel, Scene_init, Scene_init_point
# from scene.camera_optimise import optimise_cam
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.system_utils import searchForMaxIteration
from plyfile import PlyData, PlyElement
import numpy as np
from scene.cameras import Camera

import cv2

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def read_plyblock(ply_path, x_min, x_max, y_min, y_max):
    plydata = PlyData.read(ply_path)
    num1 = len(plydata.elements[0].data)

    data = np.stack((np.asarray(plydata.elements[0]["x"]),
                     np.asarray(plydata.elements[0]["y"]),
                     np.asarray(plydata.elements[0]["z"])), axis=1)

    for p in plydata.elements[0].properties:
        if p.name == "x" or p.name == "y" or p.name == "z":
            continue

        data = np.hstack((data, np.asarray(plydata.elements[0][p.name]).reshape(num1, 1)))

    print(data.shape)

    data1 = data[data[:, 0] >= x_min]
    data1 = data1[data1[:, 0] < x_max]
    data1 = data1[data1[:, 2] >= y_min]
    data1 = data1[data1[:, 2] < y_max]
    print(data1.shape)

    return data1


def create_output(data, filename):
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2']
    for i in range(45):
        stri = 'f_rest_' + str(i)
        l += [stri]
    l += ['opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']
    # print(l)
    dtype_full = [(attribute, 'f4') for attribute in l]

    elements = np.empty(data.shape[0], dtype=dtype_full)
    attributes = data
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(filename)


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from,
             num_block=[1, 2]):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    max_x, min_x, max_y, min_y, max_z, min_z = Scene_init(dataset)
    print(max_x, min_x, max_y, min_y, max_z, min_z)

    max_x_train = max_x + (max_x - min_x) * opt.spatial_append
    max_y_train = max_z + (max_z - min_z) * opt.spatial_append
    min_x_train = min_x - (max_x - min_x) * opt.spatial_append
    min_y_train = min_z - (max_z - min_z) * opt.spatial_append

    print(max_x_train, min_x_train, max_y_train, min_y_train)

    blocks = []

    num_block1, num_block2 = num_block
    for i in range(num_block1 * num_block2):
        x_limit_min = min_x_train + (max_x_train - min_x_train) / num_block1 * (i % num_block1)
        x_limit_max = min_x_train + (max_x_train - min_x_train) / num_block1 * (i % num_block1 + 1)
        y_limit_min = min_y_train + (max_y_train - min_y_train) / num_block2 * (i // num_block1)
        y_limit_max = min_y_train + (max_y_train - min_y_train) / num_block2 * (i // num_block1 + 1)
        blocks.append((x_limit_min, x_limit_max, y_limit_min, y_limit_max))
        print(x_limit_min, x_limit_max, y_limit_min, y_limit_max)

    dataset.model_path = ""
    iteration = searchForMaxIteration(os.path.join(dataset.model_path, "point_cloud"))
    point_cloud_path = os.path.join(dataset.model_path, "point_cloud/iteration_{}".format(iteration))
    for i in range(num_block1 * num_block2):
        x_min = blocks[i][0]
        x_max = blocks[i][1]
        y_min = blocks[i][2]
        y_max = blocks[i][3]
        ply_path = os.path.join(point_cloud_path, "point_cloud" + str(i) + ".ply")
        if not os.path.exists(ply_path):
            continue
        ply = read_plyblock(ply_path, x_min, x_max, y_min, y_max)
        if i == 0:
            plys = ply
        else:
            plys = np.vstack([plys, ply])
    ply_path = os.path.join(point_cloud_path, "point_cloud" + ".ply")
    create_output(plys, ply_path)


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[3_000, 7_000, 30_000, 50_000, 70_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[10_000, 30_000, 70_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--num_block", nargs=2, type=int, default=[2, 2], help="Specify the number of blocks as n*m")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations,
             args.checkpoint_iterations, args.start_checkpoint, args.debug_from, num_block=args.num_block)

    # All done
    print("\nTraining complete.")

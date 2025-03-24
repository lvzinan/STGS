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
from utils.loss_utils import l1_loss, ssim, structure_loss, caculate_struct
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel, Scene_init, Scene_init_point
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

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import StructuralSimilarityIndexMeasure

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def read_plyblock(ply_path, x_min, x_max, y_min, y_max):
    plydata = PlyData.read(ply_path)
    num = len(plydata.elements[0].data)

    data = np.stack((np.asarray(plydata.elements[0]["x"]),
                     np.asarray(plydata.elements[0]["y"]),
                     np.asarray(plydata.elements[0]["z"])), axis=1)

    for p in plydata.elements[0].properties:
        if p.name == "x" or p.name == "y" or p.name == "z":
            continue

        data = np.hstack((data, np.asarray(plydata.elements[0][p.name]).reshape(num, 1)))

    print("original block shape", data.shape)

    data_split = data[data[:, 0] >= x_min]
    data_split = data_split[data_split[:, 0] < x_max]
    data_split = data_split[data_split[:, 2] >= y_min]
    data_split = data_split[data_split[:, 2] < y_max]
    print("final block shape", data_split.shape)

    return data_split


def create_output(data, filename):
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2']
    for i in range(45):
        stri = 'f_rest_' + str(i)
        l += [stri]
    l += ['opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']
    dtype_full = [(attribute, 'f4') for attribute in l]

    elements = np.empty(data.shape[0], dtype=dtype_full)
    attributes = data
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(filename)


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from,
             num_block=[1, 2]):
    tb_writer = prepare_output_and_logger(dataset)

    max_x, min_x, max_y, min_y, max_z, min_z = Scene_init(dataset)
    print("max_x, min_x, max_y, min_y, max_z, min_z", max_x, min_x, max_y, min_y, max_z, min_z)

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

    for i in range(len(blocks)):
        first_iter = 0
        x_min = blocks[i][0]
        x_max = blocks[i][1]
        y_min = blocks[i][2]
        y_max = blocks[i][3]
        print(x_min, x_max, y_min, y_max)

        # optimise_cam0 = optimise_cam().cuda()

        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, train=True, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, block=i,
                      R_append=0.1, cam_append=0.25, batch_size=opt.batch_size)
        gaussians.training_setup(opt)
        if checkpoint:
            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, opt)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        for _ in range(len(scene.chunks)):
            scene.scene_batch(dataset)
            for viewpoint in scene.getTrainCameras():
                image_H, image_W = viewpoint.original_image.shape[1], viewpoint.original_image.shape[2]
                render_pkg = render(viewpoint, gaussians, pipe, bg, uid=scene.uid, iteration=0,
                                    trans=True, iteration_cam=opt.iteration_cam)
                visibility_filter = render_pkg["visibility_filter"]
                if visibility_filter.sum() < opt.vis_points:
                    scene.vis_img[scene.uid[viewpoint.colmap_id]] = False

        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, train=True, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
                      block=i, vis_img=scene.vis_img, R_append=opt.r_append, image_H=image_H,
                      image_W=image_W, cam_append=opt.cam_append, batch_size=opt.batch_size)

        gaussians.training_setup(opt)
        if checkpoint:
            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, opt)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end = torch.cuda.Event(enable_timing=True)

        viewpoint_stack = None
        ema_loss_for_log = 0.0
        progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
        first_iter += 1
        loss_batch = []
        for iteration in range(first_iter, opt.iterations + 1):
            if iteration % 1000 == 1:
                scene.scene_batch(dataset)

            iter_start.record()

            gaussians.update_learning_rate(iteration)

            if iteration % 1000 == 0:
                gaussians.oneupSHdegree()

            # Pick a random Camera
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
            while not scene.vis_img[scene.uid[viewpoint_cam.colmap_id]]:
                try:
                    if not viewpoint_stack:
                        viewpoint_stack = scene.getTrainCameras().copy()
                    viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
                except:
                    scene.scene_batch(dataset)

            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True

            bg = torch.rand((3), device="cuda") if opt.random_background else background

            gt_image = viewpoint_cam.original_image.cuda()

            render_pkg = render(viewpoint_cam, gaussians, pipe, bg, uid=scene.uid, iteration=iteration, trans=True,
                                iteration_cam=opt.iteration_cam)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg[
                "viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            canny_image = caculate_struct(image)
            canny_ori = caculate_struct(gt_image)

            loss_structure = structure_loss(image, gt_image, canny_image, canny_ori)

            # Loss
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (
                    1.0 - ssim(image, gt_image)) + loss_structure * 3.0

            if len(loss_batch) < opt.batch_size:
                loss_batch.append(float(loss_structure))
            else:
                if iteration % 1000 < opt.batch_size:
                    loss_batch[iteration % 1000] = float(loss_structure)

            loss.backward()

            iter_end.record()

            with torch.no_grad():
                # if iteration > opt.iteration_cam:
                #     gaussians.train_cam_r[scene.uid[viewpoint_cam.colmap_id]] = torch.nn.functional.normalize(gaussians.train_cam_r[scene.uid[viewpoint_cam.colmap_id]].reshape(1, 4), p=2,
                #                                   dim=1).reshape(4, )

                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()

                # Log and save
                training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                                testing_iterations, scene, render, (pipe, background), dataset.model_path, opt)
                if (iteration in saving_iterations):
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.save(iteration)

                # Densification
                if iteration < opt.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                         radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > opt.densify_from_iter and iteration % 100 == 0 and iteration <= opt.s_loss_iter1:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent,
                                                    size_threshold, x_min, x_max, y_min, y_max, iteration,
                                                    r_append=opt.r_append, clone_from_iter=opt.clone_from_iter,
                                                    max_points=opt.max_points)
                    elif iteration > opt.s_loss_iter1 and iteration % 10 == 0 and float(loss_structure) - sum(
                            loss_batch) / len(
                        loss_batch) >= opt.s_loss_th1 and iteration < opt.s_loss_iter2:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent,
                                                    size_threshold, x_min, x_max, y_min, y_max, iteration,
                                                    r_append=opt.r_append, clone_from_iter=opt.clone_from_iter,
                                                    max_points=opt.max_points)
                    elif iteration > opt.s_loss_iter2 + 5000 and float(loss_structure) - sum(loss_batch) / len(
                            loss_batch) >= opt.s_loss_th2 and iteration % 1000 > 1000 - opt.batch_size:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent,
                                                    size_threshold, x_min, x_max, y_min, y_max, iteration,
                                                    enhance=True, r_append=opt.r_append,
                                                    clone_from_iter=opt.clone_from_iter, max_points=opt.max_points)

                    if iteration % opt.opacity_reset_interval == 0 or (
                            dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()

                if iteration % 1000 == 0:
                    gaussians.report_size()

                # Optimizer step
                if iteration < opt.iterations:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)

                if (iteration in checkpoint_iterations):
                    print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

                if (iteration in checkpoint_iterations):
                    training_report_all(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                                        testing_iterations, scene, render, (pipe, background), dataset.model_path,
                                        dataset, opt)

    iteration = searchForMaxIteration(os.path.join(dataset.model_path, "point_cloud"))
    point_cloud_path = os.path.join(dataset.model_path, "point_cloud/iteration_{}".format(iteration))
    for i in range(len(blocks)):
        x_min = blocks[i][0]
        x_max = blocks[i][1]
        y_min = blocks[i][2]
        y_max = blocks[i][3]
        ply_path = os.path.join(point_cloud_path, "point_cloud" + str(i) + ".ply")
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


def ssim_eval(img1, img2):
    metric = StructuralSimilarityIndexMeasure(data_range=1.0)
    return metric(img1, img2)


def lpips(img1, img2):
    metric = LearnedPerceptualImagePatchSimilarity(net_type='vgg')
    return metric(img1, img2)


import matplotlib.pyplot as plt
from PIL import Image


def training_report_all(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                        renderArgs, model_path, dataset, opt):
    # Report test and samples of training set
    torch.cuda.empty_cache()
    l1_test = 0.0
    psnr_test = 0.0
    num_test = 0

    psnr_all = 0.0
    lpips_all = 0.0
    ssim_all = 0.0
    num_all = 0

    for i in range(len(scene.chunks)):
        test_cameras = scene.getTrainCameras_all(dataset)

        for idx, viewpoint in enumerate(test_cameras):
            if not scene.vis_img[scene.uid[viewpoint.colmap_id]]:
                continue

            image = torch.clamp(renderFunc(viewpoint, scene.gaussians, uid=scene.uid, iteration=iteration,
                                           iteration_cam=opt.iteration_cam, *renderArgs)["render"], 0.0, 1.0)
            # image = torch.clamp(renderFunc(viewpoint, scene.gaussians,mode=1, *renderArgs)["render"], 0.0, 1.0)
            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

            if viewpoint.colmap_id % 50 == 0:
                distance = lpips(
                    image.cpu().reshape([1, image.shape[0], image.shape[1], image.shape[2]]),
                    gt_image.cpu().reshape([1, image.shape[0], image.shape[1], image.shape[2]]))
                lpips_all += distance

                ssim_score = ssim_eval(
                    image.cpu().reshape([1, image.shape[0], image.shape[1], image.shape[2]]),
                    gt_image.cpu().reshape([1, image.shape[0], image.shape[1], image.shape[2]]))
                ssim_all += ssim_score

                psnr_score = psnr(image, gt_image).mean().double()
                psnr_all += psnr_score

                num_all += 1

            l1_test += l1_loss(image, gt_image).mean().double()
            psnr_test += psnr(image, gt_image).mean().double()
            num_test += 1
    psnr_test /= num_test
    l1_test /= num_test
    lpips_all /= (num_all + 1)
    ssim_all /= (num_all + 1)
    psnr_all /= (num_all + 1)
    print("\n[ITER {}] : L1 {} PSNR {}".format(iteration, l1_test, psnr_test))
    print("\n[ITER {}] : TEST SET num {} LPIPS {} PSNR {} SSIM {}".format(iteration, num_all, lpips_all, psnr_all,
                                                                          ssim_all))
    print("num_test", num_test)
    with open(model_path + "/log.txt", "a") as f:
        f.writelines(["test ", str(iteration), str(l1_test), str(psnr_test), "\n"])
        f.writelines(["test_all ", str(iteration), str(num_all), str(lpips_all), str(ssim_all), str(psnr_all), "\n"])

    torch.cuda.empty_cache()


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                    renderArgs, model_path, opt):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 90, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                num_test = 0
                for idx, viewpoint in enumerate(config['cameras']):
                    if not scene.vis_img[scene.uid[viewpoint.colmap_id]]:
                        continue
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, uid=scene.uid, iteration=iteration,
                                                   iteration_cam=opt.iteration_cam, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    num_test += 1
                psnr_test /= num_test
                l1_test /= num_test
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                print(model_path)
                with open(model_path + "/log.txt", "a") as f:
                    f.writelines([str(iteration), str(l1_test), str(psnr_test), "\n"])
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


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

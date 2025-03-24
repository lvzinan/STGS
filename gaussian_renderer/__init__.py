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

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.graphics_utils import getWorld2View2, getProjectionMatrix


def rotation_matrix(angles):
    """
    Generate a rotation matrix for given angles (roll, pitch, yaw).

    :param angles: A tuple of three angles in radians (roll, pitch, yaw)
    :return: A 3x3 rotation matrix
    """
    Rx = torch.tensor([[1, 0, 0],
                       [0, torch.cos(angles[0]), -torch.sin(angles[0])],
                       [0, torch.sin(angles[0]), torch.cos(angles[0])]], requires_grad=True).cuda()

    Ry = torch.tensor([[torch.cos(angles[1]), 0, torch.sin(angles[1])],
                       [0, 1, 0],
                       [-torch.sin(angles[1]), 0, torch.cos(angles[1])]], requires_grad=True).cuda()

    Rz = torch.tensor([[torch.cos(angles[2]), -torch.sin(angles[2]), 0],
                       [torch.sin(angles[2]), torch.cos(angles[2]), 0],
                       [0, 0, 1]], requires_grad=True).cuda()

    R = Rz @ Ry @ Rx
    return R


def rotate_point(point, angles):
    alpha, beta, gamma = angles[0], angles[1], angles[2]

    x_rotated = point[0]
    y_rotated = point[1] * torch.cos(alpha) - point[2] * torch.sin(alpha)
    z_rotated = point[1] * torch.sin(alpha) + point[2] * torch.cos(alpha)

    point_rotated = torch.tensor([x_rotated, y_rotated, z_rotated])

    x_rotated = point_rotated[0] * torch.cos(beta) + point_rotated[2] * torch.sin(beta)
    y_rotated = point_rotated[1]
    z_rotated = -point_rotated[0] * torch.sin(beta) + point_rotated[2] * torch.cos(beta)

    point_rotated = torch.tensor([x_rotated, y_rotated, z_rotated])

    x_rotated = point_rotated[0] * torch.cos(gamma) - point_rotated[1] * torch.sin(gamma)
    y_rotated = point_rotated[0] * torch.sin(gamma) + point_rotated[1] * torch.cos(gamma)
    z_rotated = point_rotated[2]

    return torch.tensor([x_rotated, y_rotated, z_rotated])


def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0, override_color=None,
           uid=None, iteration=0, trans=False, iteration_cam=50000):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # print(uid)

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    if iteration > iteration_cam:
        t0 = pc.get_xyz
        pc.train_cam_r.retain_grad()
        pc.train_cam_xyz.retain_grad()
        q = torch.nn.functional.normalize(pc.train_cam_r[uid[viewpoint_camera.colmap_id]].reshape(1, 4), p=2,
                                          dim=1).reshape(4, )

        w, x, y, z = q[0], q[1], q[2], q[3]

        R = torch.stack((1 - 2 * (y * y) - 2 * (z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                         2 * (x * y + w * z), 1 - 2 * (x * x) - 2 * (z * z), 2 * (y * z - w * x),
                         2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x) - 2 * (y * y)), -1)
        R = R.reshape(3, 3)

        t0 = R @ t0.transpose(0, 1)

        T = pc.train_cam_xyz[uid[viewpoint_camera.colmap_id]].reshape(3, 1)
        t0 = (t0 + T).transpose(0, 1)
        if iteration % 100 == 0:
            print(R, T, pc._xyz[0], q)
    else:
        t0 = pc.get_xyz

    screenspace_points = torch.zeros_like(t0, dtype=t0.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    means3D = t0
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        if iteration > iteration_cam:
            cov3D_precomp = pc.get_covariance(scaling_modifier, R)
        else:
            cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = (t0 - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            if iteration > iteration_cam:
                sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized, R, trans=True)
            else:
                sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)

    if iteration > 0:
        exposure = pc._exposure[uid[viewpoint_camera.colmap_id]]
        rendered_image_expose = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0,
                                                                                                        1) + exposure[
                                                                                                             :3, 3,
                                                                                                             None, None]
    else:
        rendered_image_expose = rendered_image

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image_expose,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii}

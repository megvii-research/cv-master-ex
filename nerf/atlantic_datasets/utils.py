import numpy as np
import cv2
from pyobb.obb import OBB


def depths_to_pointclouds(depth, intrinsics):
    n, h, w = depth.shape
    fx, fy, cx, cy = intrinsics
    u = np.arange(w)[None, None, :].repeat(n, axis=0).repeat(h, axis=1)
    v = np.arange(h)[None, :, None].repeat(n, axis=0).repeat(w, axis=2)
    # u = np.expand_dims(np.arange(w), 0).repeat(h, axis=0)
    # v = np.expand_dims(np.arange(h), 1).repeat(w, axis=1)
    Z = depth
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy  ## n, h, w
    points = np.stack(
        [X.reshape(n, -1), Y.reshape(n, -1), Z.reshape(n, -1)], axis=-1
    )  # n, h*w, 3
    # points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
    return points


def normalize_cameras_by_pointcloud(c2w, depth, intrinsics, depth_clip=10.0):
    """
    Normalize the cameras by pointcloud.

    Calculate the OBB box of the scene's pointcloud, align it with xyz-axis,
    and normalize the pointcloud into the unit cube.

    :param c2w: (N, 3)
    :param depth: (N, H, W)
    :param intrinsics: (float, float, float, float)

    :return T (4,4), scale (float), grid_radius(float, float, float)
    """
    t = c2w[:, :3, 3]
    R = c2w[:, :3, :3]

    # (1) Get the pointclouds
    if depth_clip > 0.0:
        depth = depth.clip(0.0, depth_clip)
    ptclouds = depths_to_pointclouds(depth, intrinsics)  # (N, hw, 3)
    n, hw, _ = ptclouds.shape
    ptclouds = (np.expand_dims(R, 1).repeat(hw, 1) @ ptclouds[..., None])[
        ..., 0
    ] + np.expand_dims(t, 1).repeat(hw, 1)
    ptclouds = ptclouds.reshape(-1, 3)

    # (2) Get the OBB of pointclouds
    obb = OBB.build_from_points(ptclouds)

    # (3) Get the up direction
    R = obb.rotation @ R
    scene_up = np.around(calc_scene_up(R))
    world_up = np.array([0.0, -1.0, 0.0])
    R_align = get_rotation_from_two_vec(scene_up, world_up)
    R_obb = R_align @ obb.rotation

    # (4) Get scene center and radius
    pts = np.array(obb.points)
    pts = (R_obb @ pts[..., None])[..., 0]
    center = pts.mean(0)
    grid_size = (pts.max(0) - pts.min(0)) / 2.0
    scale = 1.0 / max(grid_size)
    grid_radius = grid_size * scale

    # (5) Get the transformation
    transform = np.eye(4)
    transform[:3, :3] = R_obb
    transform[:3, 3] = -center

    return transform, scale, grid_radius


def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


def calc_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    return fm


def calc_scene_up(R):
    """
    Calculate the up direction of the given rotation matrix.

    Assume -y is the up axis.
    """
    ups = np.sum(R * np.array([0.0, -1.0, 0.0]), axis=-1)
    world_up = np.mean(ups, axis=0)
    world_up /= np.linalg.norm(world_up)
    return world_up


def get_rotation_from_two_vec(srcv, dstv):
    c = (dstv * srcv).sum()
    cross = np.cross(srcv, dstv)
    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ]
    )
    assert c > -1, "Invalid coordinate convention!"
    R = np.eye(3) + skew + (skew @ skew) * 1 / (1 + c)
    return R

import argparse
import json
import os
import pickle
import sys

import cv2
import numpy as np
import open3d as o3d
import scipy

camera_scale = 0.1


def pose_inverse(pose):
    rotation = np.transpose(
        scipy.spatial.transform.Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
    )
    translation = -np.dot(rotation, pose[:3])
    rotation = scipy.spatial.transform.Rotation.from_matrix(rotation).as_quat()[
        [3, 0, 1, 2]
    ]
    return np.concatenate([translation, rotation])


def get_matrix_from_pose(pose):
    transformation = np.eye(4)
    rotation = o3d.geometry.get_rotation_matrix_from_quaternion(pose[3:])
    transformation[:3, :3] = rotation
    transformation[:3, 3] = pose[:3]
    return transformation


def get_inverse_matrix_from_pose(pose):
    transformation = np.eye(4)
    rotation = np.transpose(o3d.geometry.get_rotation_matrix_from_quaternion(pose[3:]))
    transformation[:3, :3] = rotation
    transformation[:3, 3] = -np.dot(rotation, pose[:3])
    return transformation


def depth_to_point_cloud(image, depth, intrinsics, native=False):
    h, w = depth.shape
    fx, fy, cx, cy = intrinsics
    u = np.expand_dims(np.arange(w), 0).repeat(h, axis=0)
    v = np.expand_dims(np.arange(h), 1).repeat(w, axis=1)
    Z = depth
    if native:
        u_u0_by_fx = (u - cx) / fx
        v_v0_by_fy = (v - cy) / fy
        Z /= np.sqrt(u_u0_by_fx**2 + v_v0_by_fy**2 + 1)
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
    colors = image.reshape(-1, 3) / 255.0
    valid_indices = np.where(np.linalg.norm(points, axis=1))[0]
    return points[valid_indices], colors[valid_indices]


def create_camera_actor(color, scale=0.01):
    CAM_POINTS = np.array(
        [
            [0, 0, 0],
            [-1, -1, 1.5],
            [1, -1, 1.5],
            [1, 1, 1.5],
            [-1, 1, 1.5],
            [-0.5, 1, 1.5],
            [0.5, 1, 1.5],
            [0, 1.2, 1.5],
        ]
    )
    CAM_LINES = np.array(
        [[1, 2], [2, 3], [3, 4], [4, 1], [1, 0], [0, 2], [3, 0], [0, 4], [5, 7], [7, 6]]
    )
    camera_actor = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(scale * CAM_POINTS),
        lines=o3d.utility.Vector2iVector(CAM_LINES),
    )
    color = (color * 1.0, 0.5 * (1 - color), 0.9 * (1 - color))
    camera_actor.paint_uniform_color(color)
    return camera_actor


def show_cameras_and_point_clouds(
    images,
    depths,
    poses,
    intrinsics,
    pose_mode="c2w",
    outlier_removal=0,
    native_depth=False,
):
    """
    :param images: numpy array of N×H×W×C (uint8, RGB)
    :param depths: numpy array of N×H×W (float32)
    :param poses: numpy array of N×4×4 (t1, t2, t3, w + xi + yj + zk)
    :param intrinsics: [fx, fy, cx, cy] or np.array([fx, fy, cx, cy])
    :param outlier_removal: typical value: 0.1 ~ 10; the lower this number the more aggressive the filter will be
    """
    result = []
    show_cameras_and_point_clouds.i = 0

    def add_camera(vis):
        image = images[show_cameras_and_point_clouds.i]
        depth = depths[show_cameras_and_point_clouds.i]
        pose = poses[show_cameras_and_point_clouds.i]
        points, colors = depth_to_point_cloud(
            image, depth, intrinsics, native=native_depth
        )
        if pose_mode == "c2w":
            transformation = pose
        if pose_mode == "w2c":
            transformation = np.linalg.inv(pose)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd = pcd.transform(transformation)
        if outlier_removal > 0:
            pcd, _ = pcd.remove_statistical_outlier(
                nb_neighbors=40, std_ratio=outlier_removal
            )
        if show_cameras_and_point_clouds.i == 0:
            vis.add_geometry(pcd, True)
        else:
            vis.add_geometry(pcd, False)
        cam_actor = create_camera_actor(1, camera_scale)
        cam_actor.transform(transformation)
        vis.add_geometry(cam_actor, False)
        show_cameras_and_point_clouds.i += 1
        return False

    key_to_callback = {}
    key_to_callback[ord(" ")] = add_camera
    o3d.visualization.draw_geometries_with_key_callbacks(result, key_to_callback)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--path", type=str, default="./ngp_0200.pkl", help="pickle path"
    )
    parser.add_argument(
        "-o",
        "--outlier_removal",
        type=float,
        default=0.5,
        help="when > 0, removes points that are further away from their neighbors",
    )
    parser.add_argument("-s", "--stride", type=int, default=1, help="index stride")
    parser.add_argument(
        "-d", "--depth_gt", action="store_true", help="point cloud use depth gt"
    )
    parser.add_argument(
        "-i", "--image_gt", action="store_true", help="point cloud use image gt"
    )
    parser.add_argument(
        "-n",
        "--native_depth",
        action="store_false",
        help="depth means z-axis coordinates or distance",
    )

    args = parser.parse_args()
    print(args)

    with open(args.path, "rb") as f:
        info = pickle.load(f)
    intrinsics = [
        info["intrinsics"][0, 0],
        info["intrinsics"][1, 1],
        info["intrinsics"][0, 2],
        info["intrinsics"][1, 2],
    ]
    depths = info["depths_gt" if args.depth_gt else "depths"]
    images = info["images_gt" if args.image_gt else "images"]
    poses = info["poses"]
    if "weights" in info:
        for i in range(len(depths)):
            depths[i] *= info["weights"][i][..., 0] > 200
    show_cameras_and_point_clouds(
        images[:: args.stride],
        depths[:: args.stride],
        poses[:: args.stride],
        intrinsics,
        pose_mode="c2w",
        outlier_removal=args.outlier_removal,
        native_depth=args.native_depth,
    )

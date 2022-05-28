import numpy as np
import torch
import scipy.spatial


class CameraPoseTransform:
    ## function here is support for nerf torch3d, cameras as torch tensor type.
    ## If you just want to change data type, you can use "from scipy.spatial.transform import Rotation"
    ## .as_quat(), .as_matrix(), link: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.as_matrix.html

    @staticmethod
    def get_matrix_from_pose(pose):
        transformation = np.eye(4)
        rotation = scipy.spatial.transform.Rotation.from_quat(
            pose[[4, 5, 6, 3]]
        ).as_matrix()
        transformation[:3, :3] = rotation
        transformation[:3, 3] = pose[:3]
        return transformation

    @staticmethod
    def get_inverse_matrix_from_pose(pose):
        transformation = np.eye(4)
        rotation = np.transpose(
            scipy.spatial.transform.Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
        )
        transformation[:3, :3] = rotation
        transformation[:3, 3] = -np.dot(rotation, pose[:3])
        return transformation

    @staticmethod
    def get_pose_from_matrix(matrix):
        # matrix: 3×4 or 4×4
        rotation = scipy.spatial.transform.Rotation.from_matrix(
            matrix[:3, :3]
        ).as_quat()[[3, 0, 1, 2]]
        translation = matrix[:3, 3]
        return np.concatenate([translation, rotation])

    @staticmethod
    def get_pose_from_inverse_matrix(matrix):
        # matrix: 3×4 or 4×4
        rotation_matrix = np.transpose(matrix[:3, :3])
        rotation = scipy.spatial.transform.Rotation.from_matrix(
            rotation_matrix[:3, :3]
        ).as_quat()[[3, 0, 1, 2]]
        translation = -np.dot(rotation_matrix, matrix[:3, 3])
        return np.concatenate([translation, rotation])

    @staticmethod
    def pose_inverse(pose):
        rotation = np.transpose(
            scipy.spatial.transform.Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
        )
        translation = -np.dot(rotation, pose[:3])
        rotation = scipy.spatial.transform.Rotation.from_matrix(rotation).as_quat()[
            [3, 0, 1, 2]
        ]
        return np.concatenate([translation, rotation])

    @staticmethod
    def rt_inverse(R, T):
        # input: rt pose
        # w2c->c2w or c2w->w2c
        R = R.transpose(1, 0)
        T = -R @ T
        return R, T

    @staticmethod
    def inverse_transform(transform):
        """
        Inverse a given transform.

        Args:
            transform: np.array of shape (4, 4), col-major.
        """
        R = transform[:3, :3]
        T = transform[:3, 3]
        inv_R, inv_T = CameraPoseTransform.rt_inverse(R, T)
        inv_transform = np.eye(4, dtype=transform.dtype)
        inv_transform[:3, :3] = inv_R
        inv_transform[:3, 3] = inv_T
        return inv_transform

    @staticmethod
    def get_transform_between_2cameras(camera_A, camera_B):
        """
        Get relative tranform between two cameras.

        camera_A @ transform = camera_B

        Args:
            camera_A: np.array of shape(4, 4), col-major, source camera.
            camera_B: np.array of shape(4, 4), col-major, target camera.
        """
        return CameraPoseTransform.inverse_transform(camera_A) @ camera_B

    @staticmethod
    def row_column_conversion(R):
        ## PyTorch3D采用row-major vector，这意味着点坐标的表示是行向量而不是通常情况下的列向量。
        R = R.transpose(0, 1)
        return R

    @staticmethod
    def rot180_zaxis(R, T):
        ## Pytorch3D的相机坐标系假定+X向左，+Y向上，+Z向外，
        ## 这与一般的+X向右，+Y向下恰好相反，也就意味着图片的遍历顺序是相反的，等价于在camera坐标系上绕z轴旋转180度
        R[:2] *= -1
        T[:2] *= -1
        return R, T

    @staticmethod
    def euler2rt(sequence="zyx", angles=[0, 0, 0], degrees=True, extrinsic=False):
        if extrinsic:
            sequence = sequence.lower()
        else:
            sequence = sequence.upper()
        r = scipy.spatial.transform.Rotation.from_euler(
            sequence, angles, degrees=degrees
        )
        return r.as_matrix()

    @staticmethod
    def quaternion2rt(pose, data_type, focal_length, principal_point):
        # pose order: xyz, wxyz
        # if your pose is not in this order, please use shuff_order(poses, data_type) function
        # to obtain the right order
        # output: torch3d camera
        from pytorch3d.renderer import PerspectiveCameras
        from pytorch3d.transforms import quaternion_to_matrix

        R = quaternion_to_matrix(torch.tensor(pose[3:]))
        T = torch.tensor(pose[:3])

        if data_type == "w2c":
            R, T = CameraPoseTransform.rt_inverse(R, T)

        R, T = CameraPoseTransform.rot180_zaxis(R, T)
        R = CameraPoseTransform.row_column_conversion(R)

        camera = PerspectiveCameras(
            focal_length=[focal_length],
            principal_point=[principal_point],
            R=R[None, ...],
            T=T[None, :],
        )
        return camera

    @staticmethod
    def rt2quaternion(R, T):
        # input: torch3d camera R, T
        # output order: T & quan in numpy, in default order: xyz, wxyz
        from pytorch3d.transforms import matrix_to_quaternion

        R = CameraPoseTransform.row_column_conversion(R)
        R, T = CameraPoseTransform.rot180_zaxis(R, T)
        quan = matrix_to_quaternion(R)

        pose = np.zeros((7,))
        pose[:3] = T.cpu().numpy()
        pose[3:] = quan.cpu().numpy()

        return pose

    @staticmethod
    def pose2torch3d(pose, data_type, focal_length, principal_point):
        camera = CameraPoseTransform.quaternion2rt(
            pose, data_type.split("_")[-1], focal_length, principal_point
        )
        return camera

    @staticmethod
    def transform_output(pose, output_pose_type, intrinsics=None, image_size=None):
        if output_pose_type.split("_")[0] == "quat":
            if output_pose_type.split("_")[-1] == "w2c":
                pose = CameraPoseTransform.pose_inverse(pose)
            new_order = np.zeros((7,), dtype=np.int16)
            translation = output_pose_type.split("_")[1]
            rotation = output_pose_type.split("_")[2]
            for i in range(3):
                if translation[i] == "x":
                    new_order[i] = 0
                elif translation[i] == "y":
                    new_order[i] = 1
                elif translation[i] == "z":
                    new_order[i] = 2
            for i in range(4):
                if rotation[i] == "w":
                    new_order[i + 3] = 3
                elif rotation[i] == "x":
                    new_order[i + 3] = 4
                elif rotation[i] == "y":
                    new_order[i + 3] = 5
                elif rotation[i] == "z":
                    new_order[i + 3] = 6
            pose = pose[new_order]
            return pose
        if output_pose_type.split("_")[0] == "rt":
            if output_pose_type.split("_")[-1] == "w2c":
                return CameraPoseTransform.get_inverse_matrix_from_pose(pose)
            if output_pose_type.split("_")[-1] == "c2w":
                return CameraPoseTransform.get_matrix_from_pose(pose)
        if output_pose_type.split("_")[0] == "pytorch3d":
            camera_intrinsics = CameraIntrinsics(intrinsics, image_size)
            focal_length, principal_point = camera_intrinsics.get_fp()
            return CameraPoseTransform.pose2torch3d(
                pose, output_pose_type, focal_length, principal_point
            )

    @staticmethod
    def convert_camera_pose_between_opencv_and_opengl(camera_pose):
        """
        Opencv camera coordinate system: x--->right, y--->down, z--->scene.
        To convert camera poses between Opencv and Opengl conventions, the following code can be used for both Opengl2Opencv and Opencv2Opengl.
        """
        return camera_pose @ np.diag([1.0, -1.0, -1.0, 1.0])


class CameraIntrinsics:
    def __init__(self, intrinsics, image_size):
        self.intrinsics = intrinsics
        self.image_size = image_size

    def get_fp(self):
        fx, fy, px, py = self.intrinsics
        h, w = self.image_size
        focal_length = [fx / w * 2, fy / h * 2]
        principal_point = [-(px - w / 2) / (w / 2), -(py - h / 2) / (h / 2)]
        return focal_length, principal_point

    @property
    def intrinsic_matrix(self):
        fx, fy, cx, cy = self.intrinsics
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    @property
    def intrinsic_matrix_4x4(self):
        fx, fy, cx, cy = self.intrinsics
        return np.array(
            [
                [fx, 0, cx, 0],
                [0, fy, cy, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

    @property
    def inv_intrinsic_matrix(self):
        fx, fy, cx, cy = self.intrinsics
        matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        return np.linalg.inv(matrix)

    @property
    def inv_intrinsic_matrix_4x4(self):
        fx, fy, cx, cy = self.intrinsics
        matrix = np.array(
            [
                [fx, 0, cx, 0],
                [0, fy, cy, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        return np.linalg.inv(matrix)

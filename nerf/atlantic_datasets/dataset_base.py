import numpy as np
import io
import cv2
import torch
import abc
from torch.utils.data import Dataset
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface
from .camera import CameraPoseTransform, CameraIntrinsics


class DatasetBase(Dataset, metaclass=abc.ABCMeta):
    def __init__(
        self,
        scene=None,
        pose_order=[0, 1, 2, 3, 4, 5, 6],
        has_depth=False,
        depth_scale=1.0,
        depth_clip: float = 0.0,
        image_size=None,
        output_pose_type="quat_xyz_wxyz_c2w",
        indices=None,
        sample_stride=1,
    ):
        """
        Dataset基类
        每个Dataset包含多个scene, 如果指定scene, 那么只包含这一个scene.
        @param depth_clip: depth_clip > 0: 直接clip;
                           depth_clip == 0: 使用_search_depth_clip() clip;
                           depth_clip < 0: 不clip.
        @param image_size: 可选__getitem__()输出的image_size, 默认为原图大小.
        @param indices, sample_stride: 仅在__getitem__()中使用, 不影响getitem()和image_stream().
        """
        self.scenes = scene
        self.depth_scale = depth_scale
        self.pose_order = pose_order
        self.has_depth = has_depth
        self.depth_clip = depth_clip
        self.image_size = image_size
        self.output_pose_type = output_pose_type
        self.indices = indices
        self.sample_stride = sample_stride

        self.scene_info = self._build_dataset()

    @abc.abstractmethod
    def _build_dataset(self):
        """
        初始化, 需要每个派生类自定义
        """
        pass

    def image_stream(
        self,
        scene,
        get_image=True,
        get_depth=False,
        get_flow=False,
        get_pose=False,
        get_imu=False,
        get_annotation=False,
        stereo=False,
        undistort=False,
        to_tensor=False,
        image_size=None,
        stride=1,
        keyframe_indices=None,
    ):
        """
        输出某个scene中每帧数据的迭代器.
        返回frame_data字典, 默认包含["intrinsics", "tstamp", "image"]项, 可选包含["image", "depth", "pose", "imu"]项.
        @param stereo: 是否返回双目数据, 不包含["image", "depth", "pose"]项,
                        改为["image_left", "image_right", "depth_left", "depth_right", "pose_left", "pose_right"]项.
        @param undistort: 是否去畸变
        @param to_tensor: 对于["image", "depth", "pose", "intrinsics"], to_tensor=False时为numpy,
                        to_tensor=True时为torch tensor, 且"image"会被扩充为4维tensor.
        @param image_size: 可指定返回的图片大小, 这会影响["image", "depth", "intrinsics"].
        @param stride: int
        @param keyframe_indices: list, 指定返回帧的index(index和self.indices无关).
        """
        if keyframe_indices is None:
            keyframe_indices = range(0, self.scene_info[scene]["n_frames"], stride)
        for i in keyframe_indices:
            yield self.getitem(
                scene,
                index=i,
                get_image=get_image,
                get_depth=get_depth,
                get_flow=get_flow,
                get_pose=get_pose,
                get_imu=get_imu,
                get_annotation=get_annotation,
                stereo=stereo,
                undistort=undistort,
                to_tensor=to_tensor,
                image_size=image_size,
            )

    def getitem(
        self,
        scene,
        index,
        get_image=True,
        get_depth=False,
        get_flow=False,
        get_pose=False,
        get_imu=False,
        get_annotation=False,
        stereo=False,
        undistort=False,
        to_tensor=False,
        image_size=None,
    ):
        """
        输出某个scene中的某帧数据
        index指原数据集中的index, 而不是self.indices[index]
        """
        frame_data = {}
        interpolation = cv2.INTER_LINEAR

        if stereo:
            if "intrinsics_left" in self.scene_info[scene]:
                intrinsics_left = self.scene_info[scene]["intrinsics_left"].copy()
            else:
                intrinsics_left = self.scene_info[scene]["intrinsics"].copy()
            if "intrinsics_right" in self.scene_info[scene]:
                intrinsics_right = self.scene_info[scene]["intrinsics_right"].copy()
            else:
                intrinsics_right = intrinsics_left
            if get_image:
                image_left = self._image_read(
                    self.scene_info[scene]["images_left"][index]
                )
                image_right = self._image_read(
                    self.scene_info[scene]["images_right"][index]
                )
                if undistort:
                    image_left, intrinsics_left = self._undistort(
                        image_left, intrinsics_left, "left"
                    )
                    image_right, intrinsics_right = self._undistort(
                        image_right, intrinsics_right, "right"
                    )
                if image_size is not None:
                    intrinsics_left[0] *= image_size[1] / image_left.shape[1]
                    intrinsics_left[1] *= image_size[0] / image_left.shape[0]
                    intrinsics_left[2] *= image_size[1] / image_left.shape[1]
                    intrinsics_left[3] *= image_size[0] / image_left.shape[0]
                    intrinsics_right[0] *= image_size[1] / image_right.shape[1]
                    intrinsics_right[1] *= image_size[0] / image_right.shape[0]
                    intrinsics_right[2] *= image_size[1] / image_right.shape[1]
                    intrinsics_right[3] *= image_size[0] / image_right.shape[0]
                    image_left = cv2.resize(
                        image_left,
                        (image_size[1], image_size[0]),
                        interpolation=interpolation,
                    )
                    image_right = cv2.resize(
                        image_right,
                        (image_size[1], image_size[0]),
                        interpolation=interpolation,
                    )
                if to_tensor:
                    intrinsics_left = torch.as_tensor(intrinsics_left)
                    intrinsics_right = torch.as_tensor(intrinsics_right)
                    image_left = torch.as_tensor(image_left).permute(2, 0, 1)[None]
                    image_right = torch.as_tensor(image_right).permute(2, 0, 1)[None]
                frame_data["image"] = image_left
                frame_data["image_left"] = image_left
                frame_data["image_right"] = image_right
                frame_data["intrinsics"] = intrinsics_left
                frame_data["intrinsics_left"] = intrinsics_left
                frame_data["intrinsics_right"] = intrinsics_right
            # TODO: undistort后depth/flow也要变
            if get_depth:
                depth_left = self._depth_clip(
                    self._depth_read(self.scene_info[scene]["depths_left"][index])
                )
                depth_right = self._depth_clip(
                    self._depth_read(self.scene_info[scene]["depths_right"][index])
                )
                if image_size is not None:
                    depth_left = cv2.resize(
                        depth_left,
                        (image_size[1], image_size[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    depth_right = cv2.resize(
                        depth_right,
                        (image_size[1], image_size[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                if to_tensor:
                    depth_left = torch.as_tensor(depth_left)
                    depth_right = torch.as_tensor(depth_right)
                frame_data["depth"] = depth_left
                frame_data["depth_left"] = depth_left
                frame_data["depth_right"] = depth_right
            if get_flow:
                flow_left = self._flow_read(self.scene_info[scene]["flows_left"][index])
                flow_right = self._flow_read(
                    self.scene_info[scene]["flows_right"][index]
                )
                if image_size is not None:
                    flow_left = cv2.resize(
                        flow_left,
                        (image_size[1], image_size[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    flow_right = cv2.resize(
                        flow_right,
                        (image_size[1], image_size[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                if to_tensor:
                    flow_left = torch.as_tensor(flow_left)
                    flow_right = torch.as_tensor(flow_right)
                frame_data["flow"] = flow_left
                frame_data["flow_left"] = flow_left
                frame_data["flow_right"] = flow_right
            if get_annotation:
                annotation_left = self.scene_info[scene]["annotations_left"][index]
                annotation_right = self.scene_info[scene]["annotations_right"][index]
                for annotation_dict in annotation_left:
                    if "read_function" in annotation_dict:
                        frame_data[annotation_dict["key"]] = annotation_dict[
                            "read_function"
                        ](annotation_dict["data"])
                    else:
                        frame_data[annotation_dict["key"]] = annotation_dict["data"]
                    frame_data[annotation_dict["key"] + "_left"] = frame_data[
                        annotation_dict["key"]
                    ]
                for annotation_dict in annotation_right:
                    if "read_function" in annotation_dict:
                        frame_data[annotation_dict["key"] + "_right"] = annotation_dict[
                            "read_function"
                        ](annotation_dict["data"])
                    else:
                        frame_data[annotation_dict["key"] + "_right"] = annotation_dict[
                            "data"
                        ]
            if get_pose:
                pose_left = self.scene_info[scene]["poses_left"][index]
                pose_left = CameraPoseTransform.transform_output(
                    pose_left,
                    self.output_pose_type,
                    intrinsics_left,
                    image_size
                    if image_size is not None
                    else self.scene_info[scene]["image_size"],
                )
                pose_right = self.scene_info[scene]["poses_right"][index]
                pose_right = CameraPoseTransform.transform_output(
                    pose_right,
                    self.output_pose_type,
                    intrinsics_right,
                    image_size
                    if image_size is not None
                    else self.scene_info[scene]["image_size"],
                )
                if to_tensor:
                    pose_left = torch.as_tensor(pose_left)
                    pose_right = torch.as_tensor(pose_right)
                frame_data["pose"] = pose_left
                frame_data["pose_left"] = pose_left
                frame_data["pose_right"] = pose_right
        else:
            if "intrinsics_left" in self.scene_info[scene]:
                intrinsics = self.scene_info[scene]["intrinsics_left"].copy()
            else:
                intrinsics = self.scene_info[scene]["intrinsics"].copy()
            if get_image:
                image = self._image_read(self.scene_info[scene]["images_left"][index])
                if undistort:
                    image, intrinsics = self._undistort(image, intrinsics)
                if image_size is not None:
                    intrinsics[0] *= image_size[1] / image.shape[1]
                    intrinsics[1] *= image_size[0] / image.shape[0]
                    intrinsics[2] *= image_size[1] / image.shape[1]
                    intrinsics[3] *= image_size[0] / image.shape[0]
                    image = cv2.resize(
                        image,
                        (image_size[1], image_size[0]),
                        interpolation=interpolation,
                    )
                if to_tensor:
                    intrinsics = torch.as_tensor(intrinsics)
                    image = torch.as_tensor(image).permute(2, 0, 1)[None]
                frame_data["image"] = image
                frame_data["intrinsics"] = intrinsics
            if get_depth:
                depth = self._depth_clip(
                    self._depth_read(self.scene_info[scene]["depths_left"][index])
                )
                if image_size is not None:
                    depth = cv2.resize(
                        depth,
                        (image_size[1], image_size[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                if to_tensor:
                    depth = torch.as_tensor(depth)
                frame_data["depth"] = depth
            if get_flow:
                flow = self._flow_read(self.scene_info[scene]["flows_left"][index])
                if image_size is not None:
                    flow = cv2.resize(
                        flow,
                        (image_size[1], image_size[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                if to_tensor:
                    flow = torch.as_tensor(flow)
                frame_data["flow"] = flow
            if get_annotation:
                annotations = self.scene_info[scene]["annotations_left"][index]
                for annotation_dict in annotations:
                    if "read_function" in annotation_dict:
                        frame_data[annotation_dict["key"]] = annotation_dict[
                            "read_function"
                        ](annotation_dict["data"])
                    else:
                        frame_data[annotation_dict["key"]] = annotation_dict["data"]
            if get_pose:
                pose = self.scene_info[scene]["poses_left"][index]
                pose = CameraPoseTransform.transform_output(
                    pose,
                    self.output_pose_type,
                    intrinsics,
                    image_size
                    if image_size is not None
                    else self.scene_info[scene]["image_size"],
                )
                if to_tensor:
                    pose = torch.as_tensor(pose)
                frame_data["pose"] = pose

        frame_data["tstamp"] = self.scene_info[scene]["tstamps"][index]
        if get_imu:
            frame_data["imu"] = self.scene_info[scene]["imu"][index]

        return frame_data

    def __len__(self):
        if len(self.scenes) != 1:
            raise NotImplementedError
        if self.indices is None:
            return self.get_length(self.scenes[0])
        return len(self.indices)

    def __getitem__(self, index):
        if len(self.scenes) != 1:
            raise NotImplementedError
        scene = self.scenes[0]
        idx = index if self.indices is None else self.indices[index]
        data = self.getitem(
            scene,
            idx,
            get_image=True,
            get_depth=self.has_depth,
            get_pose=True,
            get_imu=False,
            get_annotation=False,
            stereo=False,
            undistort=False,
            to_tensor=False,
            image_size=self.image_size,
        )
        image = torch.tensor(data["image"][..., ::-1].astype("float32") / 255.0)
        camera = data["pose"]
        camera_hash = int(idx / self.sample_stride)

        if self.has_depth:
            depth = torch.tensor(data["depth"].astype("float32"))[..., None]
            return {
                "image": image,  # (h, w, 3)
                "depth": depth,  # (h, w, 1)
                "camera": camera,  # Camera
                "camera_hash": camera_hash,  # int
            }
        else:
            return {
                "image": image,  # (h, w, 3)
                "camera": camera,  # Camera
                "camera_hash": camera_hash,  # int
            }

    def _search_depth_clip(self, depth, max=100, step=10, downsample=True):
        """
        Search depth clipping threshold.
        """
        if downsample:
            h, w = depth.shape[-2:]
            depth = torch.nn.functional.interpolate(
                depth,
                size=(h // 2, w // 2),
                mode="nearest",
            )

        l, r = 0, max

        def is_valid(thred):
            return (depth < thred).sum() == (depth < thred + step).sum()

        while l < r:
            m = (l + r) // 2
            if is_valid(m):
                r = m
            else:
                l = m + 1

        return r

    def _sync_tstamps(self, tstamps, tstamps1):
        """images, depths, poses的时间可能不一一对应, 需要匹配"""
        matching_indices = []
        for tstamp in tstamps:
            diffs = np.abs(tstamps1 - tstamp)
            matching_indices.append(int(np.argmin(diffs)))
        return matching_indices

    def _image_read(self, img_name):
        img = cv2.imread(img_name)
        return img

    def _flow_read(self, data_id):
        return self._image_read(data_id)

    def _depth_clip(self, depth):
        if self.depth_clip < 0:
            return depth
        if self.depth_clip > 0:
            return depth.clip(0.0, self.depth_clip)
        if self.depth_clip == 0:
            depth_clip = self._search_depth_clip(depth, downsample=False)
            return depth.clip(0.0, depth_clip)

    def _poses_read(self, file_path):
        raw_mat = file_interface.csv_read_matrix(file_path, delim=" ", comment_str="#")
        mat = np.array(raw_mat).astype(float)
        n_rows = mat.shape[1]
        if n_rows == 7:
            tstamps = 1.0 * np.array(list(range(mat.shape[0])))
            poses = mat
        if n_rows == 8:
            tstamps = mat[:, 0]
            poses = mat[:, 1:]
        poses = poses[:, self.pose_order]
        return tstamps, poses

    def _undistort(self, image, intrinsics_original, camera="left"):
        """要求去畸变之后image size不变"""
        raise NotImplementedError

    def get_length(self, scene=None):
        if len(self.scenes) == 1:
            scene = self.scenes[0]
        elif scene is None:
            raise NotImplementedError
        return self.scene_info[scene]["n_frames"]

    def get_image_size(self, scene=None):
        if len(self.scenes) == 1:
            scene = self.scenes[0]
        elif scene is None:
            raise NotImplementedError
        if self.image_size is None:
            return self.scene_info[scene]["image_size"]
        else:
            return self.image_size

    def get_intrinsics(self, scene=None, undistort=False):
        if len(self.scenes) == 1:
            scene = self.scenes[0]
        elif scene is None:
            raise NotImplementedError
        intrinsics = self.scene_info[scene]["intrinsics"]
        if self.image_size is None and not undistort:
            return intrinsics
        image = self._image_read(self.scene_info[scene]["images_left"][0])
        if undistort:
            image, intrinsics = self._undistort(image, intrinsics)
        if self.image_size is not None:
            intrinsics[0] *= self.image_size[1] / image.shape[1]
            intrinsics[1] *= self.image_size[0] / image.shape[0]
            intrinsics[2] *= self.image_size[1] / image.shape[1]
            intrinsics[3] *= self.image_size[0] / image.shape[0]
        return intrinsics

    def get_stereo_baseline(self, scene=None):
        if len(self.scenes) == 1:
            scene = self.scenes[0]
        elif scene is None:
            raise NotImplementedError
        pose_left = self.scene_info[scene]["poses_left"][0]
        pose_right = self.scene_info[scene]["poses_right"][0]
        baseline = np.linalg.norm(pose_left[:3, 3] - pose_right[:3, 3])
        return baseline

    def filename_to_index(self, scene):
        """一些用于训练的split需要从 image_path 到 scene/index 的索引"""
        return NotImplementedError

    def get_gt_trajectory(self, scene):
        """返回为evo库的PoseTrajectory3D类型"""
        if "poses_left_original" in self.scene_info[scene]:
            poses = self.scene_info[scene]["poses_left_original"]
            tstamps = self.scene_info[scene]["poses_tstamps"]
        else:
            poses = self.scene_info[scene]["poses_left"]
            tstamps = self.scene_info[scene]["tstamps"]
        xyz = poses[:, 0:3]  # n x 3
        quat = poses[:, 3:]  # n x 4
        return PoseTrajectory3D(xyz, quat, tstamps)

    def export_gt_las(self, scene, destination="test.las", stride=10):
        from ci.save import export_las

        images = []
        depths = []
        poses = []
        tstamps = []
        for fr in self.image_stream(
            scene,
            get_image=True,
            get_depth=True,
            get_pose=True,
            image_size=None,
            stride=stride,
        ):
            intrinsics = fr["intrinsics"]
            images.append(fr["image"])
            depths.append(fr["depth"])
            poses.append(fr["pose"])
            tstamps.append(fr["tstamp"])
        images = np.stack(images)
        depths = np.stack(depths)
        poses = np.stack(poses)

        export_las(
            destination,
            images,
            depths,
            poses,
            intrinsics,
            tstamps=tstamps,
            pose_mode="c2w",
            voxel_downsample=True,
        )

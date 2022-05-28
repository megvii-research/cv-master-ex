import importlib
from collections import defaultdict
from typing import Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .selector import selectors
from .utils import normalize_cameras_by_pointcloud
from .camera import CameraPoseTransform as CPT


def _class_to_module(name):
    return name.lower()


def bgr2rgb(image):
    if image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    return image


def sky2rgba(image, classSegmentation, class_colors):
    if "sky" in class_colors:
        sky_color = np.array(class_colors["sky"])
    elif "Sky" in class_colors:
        sky_color = np.array(class_colors["Sky"])
    else:
        return image
    if image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
    image[:, :, 3] = (np.linalg.norm(classSegmentation - sky_color, axis=2) != 0) * 255
    return image


def depth2distance(depth, intrinsics):
    """
    Depth to distance.

    Args:
        depth: np.array of shape(..., h, w)
        intrinsics: [fx, fy, cx, cy]
    """
    h, w = depth.shape[-2:]
    fx, fy, cx, cy = intrinsics
    u = np.expand_dims(np.arange(w), 0).repeat(h, axis=0)
    v = np.expand_dims(np.arange(h), 1).repeat(w, axis=1)
    u_u0_by_fx = (u - cx) / fx
    v_v0_by_fy = (v - cy) / fy
    distance = depth.copy()
    distance *= np.sqrt(u_u0_by_fx**2 + v_v0_by_fy**2 + 1)
    return distance


class AtlanticDataset(Dataset):
    def __init__(
        self,
        dataset_name: str,
        scene: str,
        selector: str,
        split="train",
        image_size: Tuple[int, int] = None,
        scene_scale: int = 1.0,
        bound: int = 1.0,
        normalization_mode: str = "camera",
        depth_clip: float = 10.0,
        stereo: str = "mono",
        annotation=False,
        mask_away_sky=False,
        preload=True,
    ):
        """
        Args:
            type: One of ["train", "val", "test"]
            image_size: Output image size, (height, width). If None, use actual size.
            scene_scale: Scale of the scene.
            bound: Half length of the boxel bounding box.
            normalization_mode: One of ["camera", "pointcloud"]
            depth_clip: Only valid when `normalization_mode` == "pointcloud". The threshold of depth clipping.
            stereo: Whether to use stereo data.
            preload: Whether to preload data into GPU.
        """
        super().__init__()
        # path: the json file path.

        self.split = split
        self.image_size = image_size
        self.images = defaultdict(list)
        self.poses = defaultdict(list)
        self.depths = defaultdict(list)
        self.pose_offsets = defaultdict(list)
        self.timestamps = defaultdict(list)
        self.camera_indices = defaultdict(list)
        self.n_views = 0
        self.depth_scale = depth_clip
        self.intrinsics = None
        self.annotation = annotation

        if annotation:
            self.instance_segs = defaultdict(list)
            self.class_segs = defaultdict(list)

        dataset_module = _class_to_module(dataset_name)
        mod = importlib.import_module(f".dataset_{dataset_module}", package=__name__)
        Dataset = getattr(mod, dataset_name)
        dataset = Dataset(scene=scene, output_pose_type="rt_c2w", depth_clip=depth_clip)
        if hasattr(dataset, "class_colors"):
            self.class_colors = dataset.class_colors
        else:
            self.class_colors = None

        if stereo == "spider":
            scene_path, default_eye = scene.split("/")
            extra_eyes = [
                "clone",
                "15-deg-left",
                "15-deg-right",
                "30-deg-left",
                "30-deg-right",
            ]
            extra_eyes.remove(default_eye)
            extra_datasets = [
                Dataset(scene=scene_path + "/" + eye, output_pose_type="rt_c2w")
                for eye in extra_eyes
            ]
        self.has_depth = dataset.has_depth

        if type(selector) is str:
            selector = selectors[selector]
        indices = {
            split: selector[split](len(dataset)) for split in ("train", "val", "test")
        }

        # get data
        for split in ("train", "val", "test"):
            camera_index = -1
            for index in indices[split]:
                camera_index += 1
                frame_data = dataset.getitem(
                    scene,
                    index,
                    get_image=True,
                    get_depth=self.has_depth,
                    get_pose=True,
                    get_imu=False,
                    stereo=(stereo != "mono"),
                    get_annotation=annotation,
                    undistort=False,
                    to_tensor=False,
                    image_size=image_size,
                )
                self.intrinsics = frame_data["intrinsics"]
                actual_image_size = frame_data["image"].shape[:2]
                image = bgr2rgb(frame_data["image"])
                reference_pose = frame_data["pose"]
                self.poses[split].append(reference_pose)
                self.pose_offsets[split].append(np.eye(4, dtype=np.float32))
                self.camera_indices[split].append(camera_index)
                self.timestamps[split].append(frame_data["tstamp"])
                if self.has_depth:
                    self.depths[split].append(frame_data["depth"])
                if annotation:
                    self.instance_segs[split].append(frame_data["instanceSegmentation"])
                    self.class_segs[split].append(frame_data["classSegmentation"])
                    if mask_away_sky:
                        image = sky2rgba(
                            image, frame_data["classSegmentation"], self.class_colors
                        )
                self.images[split].append(image)

                if stereo != "mono" and split == "train":
                    self.intrinsics_right = frame_data["intrinsics_right"]
                    image = bgr2rgb(frame_data["image_right"])
                    self.poses[split].append(reference_pose)
                    self.pose_offsets[split].append(
                        CPT.get_transform_between_2cameras(
                            reference_pose, frame_data["pose_right"]
                        )
                    )
                    self.camera_indices[split].append(camera_index)
                    self.timestamps[split].append(frame_data["tstamp"])
                    if self.has_depth:
                        self.depths[split].append(frame_data["depth_right"])
                    if annotation:
                        self.instance_segs[split].append(
                            frame_data["instanceSegmentation_right"]
                        )
                        self.class_segs[split].append(
                            frame_data["classSegmentation_right"]
                        )
                        if mask_away_sky:
                            image = sky2rgba(
                                image,
                                frame_data["classSegmentation_right"],
                                self.class_colors,
                            )
                    self.images[split].append(image)

                if stereo == "spider" and split == "train":
                    for i in range(len(extra_eyes)):
                        frame_data = extra_datasets[i].getitem(
                            scene_path + "/" + extra_eyes[i],
                            index,
                            get_image=True,
                            get_depth=self.has_depth,
                            get_pose=True,
                            get_imu=False,
                            stereo=True,
                            get_annotation=annotation,
                            undistort=False,
                            to_tensor=False,
                            image_size=image_size,
                        )
                        image = bgr2rgb(frame_data["image"])
                        self.poses[split].append(reference_pose)
                        self.pose_offsets[split].append(
                            CPT.get_transform_between_2cameras(
                                reference_pose, frame_data["pose"]
                            )
                        )
                        self.camera_indices[split].append(camera_index)
                        self.timestamps[split].append(frame_data["tstamp"])
                        if self.has_depth:
                            self.depths[split].append(frame_data["depth"])
                        if annotation:
                            self.instance_segs[split].append(
                                frame_data["instanceSegmentation"]
                            )
                            self.class_segs[split].append(
                                frame_data["classSegmentation"]
                            )
                            if mask_away_sky:
                                image = sky2rgba(
                                    image,
                                    frame_data["classSegmentation"],
                                    self.class_colors,
                                )
                        self.images[split].append(image)
                        image = bgr2rgb(frame_data["image_right"])
                        self.poses[split].append(reference_pose)
                        self.pose_offsets[split].append(
                            CPT.get_transform_between_2cameras(
                                reference_pose, frame_data["pose_right"]
                            )
                        )
                        self.camera_indices[split].append(camera_index)
                        self.timestamps[split].append(frame_data["tstamp"])
                        if self.has_depth:
                            self.depths[split].append(frame_data["depth_right"])
                        if annotation:
                            self.instance_segs[split].append(
                                frame_data["instanceSegmentation_right"]
                            )
                            self.class_segs[split].append(
                                frame_data["classSegmentation_right"]
                            )
                            if mask_away_sky:
                                image = sky2rgba(
                                    image,
                                    frame_data["classSegmentation_right"],
                                    self.class_colors,
                                )
                        self.images[split].append(image)
            self.images[split] = np.stack(self.images[split], axis=0).astype(
                np.float32
            )  # n, h, w, 3
            self.poses[split] = np.stack(self.poses[split], axis=0).astype(
                np.float32
            )  # n, 4, 4
            self.pose_offsets[split] = np.stack(
                self.pose_offsets[split], axis=0
            ).astype(
                np.float32
            )  # n, 4, 4
            if self.has_depth:
                self.depths[split] = np.stack(self.depths[split], axis=0).astype(
                    np.float32
                )  # n, h,

        self.image_size = actual_image_size
        self.n_views = max(self.camera_indices["train"]) + 1

        # Normalize Camera
        all_poses = np.concatenate(
            [self.poses[split] for split in ("train", "val")], axis=0
        )
        if self.has_depth:
            all_depths = np.concatenate(
                [self.depths[split] for split in ("train", "val")], axis=0
            )

        if normalization_mode == "camera":
            center = np.mean(all_poses[:, :3, 3], axis=0)
            all_poses[:, :3, 3] -= center[None]
            scale = 1.0 / np.mean(np.linalg.norm(all_poses[:, :3, 3], axis=-1), axis=0)
            scene_scale *= scale
            for split in ("train", "val", "test"):
                self.poses[split][:, :3, 3] -= center[None]

        elif normalization_mode == "pointcloud":
            assert self.has_depth, "Need depth to normalize by pointcloud."
            T, sscale, sradius = normalize_cameras_by_pointcloud(
                c2w=all_poses,
                depth=all_depths,
                intrinsics=self.intrinsics,
                depth_clip=depth_clip,
            )
            scene_scale *= sscale * bound
            sradius *= bound
            print(f"Scene radius = {[f'{r:.3f}' for r in sradius.tolist()]}")
            all_poses = T @ all_poses
            for split in ("train", "val", "test"):
                self.poses[split] = T @ self.poses[split]
        else:
            raise Exception("Invalid normalization mode!")

        # Scene scale
        all_poses[:, :, 3] *= scene_scale
        self.depth_scale *= scene_scale
        if self.depth_scale <= 0.0:
            self.depth_scale = bound
        if self.has_depth:
            all_depths *= scene_scale
            self.depth_scale = all_depths.max()
        for split in ("train", "val", "test"):
            self.poses[split][:, :3, 3] *= scene_scale
            self.pose_offsets[split][:, :3, 3] *= scene_scale
            if self.has_depth:
                self.depths[split] *= scene_scale
        print(f"Scene scale = {scene_scale:.3f}.")

        if self.has_depth:
            # 将depth最大值的部分改为0，作为mask使用
            for split in ("train", "val", "test"):
                self.depths[split][
                    self.depths[split]
                    == np.max(self.depths[split], axis=(1, 2))[:, None, None]
                ] = 0.0
            # depth to distance
            for split in ("train", "val", "test"):
                self.depths[split] = depth2distance(self.depths[split], self.intrinsics)

        # uint8 -> float32
        for split in ["train", "test", "val"]:
            self.images[split] = self.images[split].astype(np.float32) / 255.0
            self.poses[split] = self.poses[split].astype(np.float32)
            self.pose_offsets[split] = self.pose_offsets[split].astype(np.float32)
            if self.has_depth:
                self.depths[split] = self.depths[split].astype(np.float32)

        # intrinsic vector -> intrinsic matrix
        intrinsics = np.eye(3, dtype=np.float32)
        intrinsics[0, 0] = self.intrinsics[0]
        intrinsics[1, 1] = self.intrinsics[1]
        intrinsics[0, 2] = self.intrinsics[2]
        intrinsics[1, 2] = self.intrinsics[3]
        self.intrinsics = intrinsics

        if preload:
            # Preload data to GPU
            self.images[self.split] = (
                torch.from_numpy(self.images[self.split]).to(torch.float32).cuda()
            )
            self.poses[self.split] = (
                torch.from_numpy(self.poses[self.split]).to(torch.float32).cuda()
            )
            self.pose_offsets[self.split] = (
                torch.from_numpy(self.pose_offsets[self.split]).to(torch.float32).cuda()
            )
            self.intrinsics = torch.from_numpy(self.intrinsics).to(torch.float32).cuda()
            if self.has_depth:
                self.depths[self.split] = (
                    torch.from_numpy(self.depths[self.split]).to(torch.float32).cuda()
                )

    def __len__(self):
        return len(self.poses[self.split])

    def __getitem__(self, index):
        import pdb; pdb.set_trace()
        data = {
            "image": self.images[self.split][index],
            "pose": self.poses[self.split][index],
            "pose_offset": self.pose_offsets[self.split][index],
            "camera_index": self.camera_indices[self.split][index],
            "timestamp": self.timestamps[self.split][index],
            "intrinsic": self.intrinsics,
            "index": index,
            "H": str(self.image_size[0]),
            "W": str(self.image_size[1]),
        }
        if self.has_depth:
            data["depth"] = self.depths[self.split][index]
        if self.annotation:
            data["instance_seg"] = self.instance_segs[self.split][index]
            data["class_seg"] = self.class_segs[self.split][index]
        return data


class NGPDataset(Dataset):
    def __init__(
        self,
        images,
        poses,
        pose_offsets,
        camera_indices,
        timestamps,
        intrinsics,
        image_size,
        depth_scale,
        depths=None,
        instance_segs=None,
        class_segs=None,
        class_colors=None,
        n_views=0,
    ):
        super().__init__()

        self.images = images
        self.poses = poses
        self.pose_offsets = pose_offsets
        self.camera_indices = camera_indices
        self.timestamps = timestamps
        self.intrinsics = intrinsics
        self.image_size = image_size
        self.depth_scale = depth_scale
        self.depths = depths
        self.instance_segs = instance_segs
        self.class_segs = class_segs
        self.class_colors = class_colors
        self.n_views = n_views

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, index):
        data = {
            "image": self.images[index],
            "pose": self.poses[index],
            "pose_offset": self.pose_offsets[index],
            "camera_index": self.camera_indices[index],
            "timestamp": self.timestamps[index],
            "intrinsic": self.intrinsics,
            "index": index,
            "H": str(self.image_size[0]),
            "W": str(self.image_size[1]),
        }
        if self.depths is not None:
            data["depth"] = self.depths[index]
        if self.instance_segs is not None:
            data["instance_seg"] = self.instance_segs[index]
        if self.class_segs is not None:
            data["class_seg"] = self.class_segs[index]
        return data


def get_dataset(**kwargs):
    dataset = AtlanticDataset(**kwargs)

    datasets = []
    for split in ["train", "val", "test"]:
        datasets.append(
            NGPDataset(
                images=dataset.images[split],
                poses=dataset.poses[split],
                pose_offsets=dataset.pose_offsets[split],
                camera_indices=dataset.camera_indices[split],
                timestamps=dataset.timestamps[split],
                intrinsics=dataset.intrinsics,
                image_size=dataset.image_size,
                depth_scale=dataset.depth_scale,
                depths=dataset.depths[split] if dataset.has_depth else None,
                instance_segs=dataset.instance_segs[split]
                if dataset.annotation
                else None,
                class_segs=dataset.class_segs[split] if dataset.annotation else None,
                class_colors=dataset.class_colors,
                n_views=dataset.n_views if split == "train" else 0,
            )
        )

    return datasets

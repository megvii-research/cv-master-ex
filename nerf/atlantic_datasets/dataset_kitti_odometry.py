import numpy as np
import cv2
from evo.tools import file_interface
from .dataset_base import DatasetBase
from .camera import CameraPoseTransform
import os
import glob

datapath = "./datasets/kitti/"
files = {}
files["KITTI_Odometry"] = [
    "00",
    "01",
    "02",
    "03",
    "04",
    "05",
    "06",
    "07",
    "08",
    "09",
    "10",
]


class KITTI_Odometry(DatasetBase):
    def __init__(
        self,
        scene=None,
        **kwargs,
    ):
        super().__init__(
            scene=[scene] if scene else files["KITTI_Odometry"],
            pose_order=None,
            has_depth=False,
            depth_scale=None,
            **kwargs,
        )
        class_colors = {
            "road": [128, 64, 128],
            "sidewalk": [244, 35, 232],
            "building": [70, 70, 70],
            "wall": [102, 102, 156],
            "fence": [190, 153, 153],
            "pole": [153, 153, 153],
            "traffic light": [250, 170, 30],
            "traffic sign": [220, 220, 0],
            "vegetation": [107, 142, 35],
            "terrain": [152, 251, 152],
            "sky": [70, 130, 180],
            "person": [220, 20, 60],
            "rider": [255, 0, 0],
            "car": [0, 0, 142],
            "truck": [0, 0, 70],
            "bus": [0, 60, 100],
            "train": [0, 80, 100],
            "motorcycle": [0, 0, 230],
            "bicycle": [119, 11, 32],
        }  # rgb
        self.class_colors = {k: v[::-1] for k, v in class_colors.items()}

    def _build_dataset(self):
        scene_info = {}

        for scene in self.scenes:
            calib_path = os.path.join(datapath, scene, "calib.txt")
            pose_path = os.path.join(datapath, scene, "poses.txt")
            image_left_path = sorted(glob.glob(datapath + "/"+ scene +  "/image_left/*.png"))
            image_right_path = sorted(glob.glob(datapath + "/"+ scene +  "/image_right/*.png"))
            tstamps_path = os.path.join(datapath, scene, "times.txt")

            with open(calib_path, "r") as fr:
                calib = np.loadtxt(fr, usecols=(1, 6, 3, 7, 4, 8, 12))
            with open(tstamps_path, "r") as fr:
                tstamps = fr.readlines()
                tstamps = np.array(list(map(lambda x: float(x.strip("\n")), tstamps)))
            with open(pose_path, "r") as fr:
                raw_mat = file_interface.csv_read_matrix(fr, delim=" ", comment_str="#")
                poses = np.array(raw_mat).astype(float).reshape((-1, 3, 4))
                poses_left = poses.copy()
                poses_right = poses.copy()
                # 参考 https://github.com/utiasSTARS/pykitti/blob/master/pykitti/odometry.py
                baseline_left = np.append(-calib[2, 4:] / calib[2, 0], 1.0)
                baseline_right = np.append(-calib[3, 4:] / calib[3, 0], 1.0)
                poses_left[:, :, 3] = poses @ baseline_left
                poses_right[:, :, 3] = poses @ baseline_right
                poses_left = np.array(
                    [
                        CameraPoseTransform.get_pose_from_matrix(pose)
                        for pose in poses_left
                    ]
                )
                poses_right = np.array(
                    [
                        CameraPoseTransform.get_pose_from_matrix(pose)
                        for pose in poses_right
                    ]
                )

            n_frames = len(image_left_path)
            intrinsics_left = calib[2, :4]
            intrinsics_right = calib[3, :4]
            image_size = self._image_read(image_left_path[0]).shape[:2]

            scene_info[scene] = {
                "n_frames": n_frames,
                "intrinsics_left": intrinsics_left,
                "intrinsics_right": intrinsics_right,
                "image_size": image_size,
                "tstamps": tstamps,
                "images_left": image_left_path,
                "images_right": image_right_path,
                "poses_left": poses_left,
                "poses_right": poses_right,
                "annotations_left": None,
                "annotations_right": None,
            }

        return scene_info



if __name__ == "__main__":
    scene = "01"
    evalset = KITTI_Odometry(scene=scene)
    frame_data = evalset.getitem(
        scene,
        333,
        get_image=True,
        get_depth=False,
        get_annotation=True,
        get_pose=True,
        stereo=True,
    )
    print(frame_data.keys())
    print(frame_data["classSegmentation"])

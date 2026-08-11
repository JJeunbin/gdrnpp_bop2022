# encoding: utf-8
"""This file includes necessary params, info."""
import os.path as osp
import mmcv
import numpy as np

# ---------------------------------------------------------------- #
# ROOT PATH INFO
# ---------------------------------------------------------------- #
cur_dir = osp.abspath(osp.dirname(__file__))
root_dir = osp.normpath(osp.join(cur_dir, ".."))
output_dir = osp.join(root_dir, "output")  # directory storing experiment data (result, model checkpoints, etc).

data_root = osp.join(root_dir, "datasets")
bop_root = osp.join(data_root, "BOP_DATASETS/")

# ---------------------------------------------------------------- #
# Z_BRACKET (custom reflective metal bracket, single object) DATASET
# ---------------------------------------------------------------- #
dataset_root = osp.join(bop_root, "z_bracket")
test_dir = osp.join(dataset_root, "test")

model_dir = osp.join(dataset_root, "models")
# no separate models_eval dir was generated for this dataset; reuse models/
model_eval_dir = osp.join(dataset_root, "models")
vertex_scale = 0.001  # models are in mm, convert to m

# object info: single object, id 1
objects = ["z_bracket"]
id2obj = {1: "z_bracket"}

obj_num = len(id2obj)
obj2id = {_name: _id for _id, _name in id2obj.items()}

model_paths = [osp.join(model_dir, "obj_{:06d}.ply").format(_id) for _id in id2obj]
texture_paths = None
model_colors = [((i + 1) * 5, (i + 1) * 5, (i + 1) * 5) for i in range(obj_num)]  # for renderer

# diameter from models_info.json (mm) / 1000 -> m
diameters = np.array([107.54068921096159]) / 1000.0

# Camera info (Doosan M1509 top-down, 0.45m WD, Unity 60deg vertical FOV)
width = 1920
height = 1080
zNear = 0.1
zFar = 2.0
camera_matrix = np.array([[935.31, 0.0, 960.0], [0.0, 935.31, 540.0], [0, 0, 1]])


def get_models_info():
    """key is str(obj_id)"""
    models_info_path = osp.join(model_dir, "models_info.json")
    assert osp.exists(models_info_path), models_info_path
    models_info = mmcv.load(models_info_path)  # key is str(obj_id)
    return models_info


# ref core/gdrn_modeling/tools/z_bracket/z_bracket_1_compute_fps.py
def get_fps_points():
    fps_points_path = osp.join(model_dir, "fps_points.pkl")
    assert osp.exists(fps_points_path), fps_points_path
    fps_dict = mmcv.load(fps_points_path)
    return fps_dict


# ref core/gdrn_modeling/tools/z_bracket/z_bracket_1_compute_keypoints_3d.py
def get_keypoints_3d():
    keypoints_3d_path = osp.join(model_dir, "keypoints_3d.pkl")
    assert osp.exists(keypoints_3d_path), keypoints_3d_path
    kpts_dict = mmcv.load(keypoints_3d_path)
    return kpts_dict

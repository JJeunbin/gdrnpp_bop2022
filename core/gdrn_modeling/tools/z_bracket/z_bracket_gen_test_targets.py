"""Generate test_targets_bop19.json for the z_bracket held-out test scenes.

Scans every scene under datasets/BOP_DATASETS/z_bracket/test/ and emits one
target entry per (image, obj_id) pair, matching the BOP19 targets format
consumed by lib/pysixd/scripts/eval_pose_results_more.py.
"""
import os.path as osp
import sys

cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(cur_dir, "../../../../"))

import mmcv
from lib.pysixd import inout

test_root = osp.join(cur_dir, "../../../../datasets/BOP_DATASETS/z_bracket/test")
out_file = osp.join(cur_dir, "../../../../datasets/BOP_DATASETS/z_bracket/test_targets_bop19.json")

targets = []
scenes = sorted(
    d for d in __import__("os").listdir(test_root) if len(d) == 6 and d.isdigit() and osp.isdir(osp.join(test_root, d))
)

for scene in scenes:
    scene_id = int(scene)
    gt_path = osp.join(test_root, scene, "scene_gt.json")
    gt_dicts = mmcv.load(gt_path)
    for im_id, annos in gt_dicts.items():
        # count instances per obj_id in this image
        obj_counts = {}
        for anno in annos:
            obj_counts[anno["obj_id"]] = obj_counts.get(anno["obj_id"], 0) + 1
        for obj_id, inst_count in obj_counts.items():
            targets.append(
                {
                    "im_id": int(im_id),
                    "inst_count": inst_count,
                    "obj_id": obj_id,
                    "scene_id": scene_id,
                }
            )

inout.save_json(out_file, targets)
print(f"wrote {len(targets)} targets ({len(scenes)} scenes) to {out_file}")

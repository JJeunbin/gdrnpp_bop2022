# demo_zbracket_yolo11.py — z_bracket 실사 이미지 추론 (GT 없음)
# YOLOv11(ultralytics, bbox만 검출) + 학습된 GDRN(포즈 회귀) 조합.
# demo_gdrn.py의 YoloPredictor는 YOLOX 전용이라 여기서는 쓰지 않고,
# ultralytics YOLO 결과를 GdrnPredictor.preprocessing()이 기대하는
# [x1,y1,x2,y2,obj_conf,cls_conf,category_id] 포맷으로 직접 변환한다.
#
# 사용법:
#   python demo_zbracket_yolo11.py ~/Downloads/data_001.png
#   python demo_zbracket_yolo11.py ~/Downloads/data_001.png --conf 0.3
import os
import os.path as osp
import sys

cur_dir = osp.dirname(osp.abspath(__file__))
PROJ_ROOT = osp.normpath(osp.join(cur_dir, ".."))  # gdrnpp_bop2022/
sys.path.insert(0, PROJ_ROOT)
sys.path.insert(0, osp.join(PROJ_ROOT, "core/gdrn_modeling/demo"))
os.chdir(PROJ_ROOT)

import argparse
import numpy as np
import torch
import cv2

from predictor_gdrn import GdrnPredictor

# ==== 경로 설정 (다른 이미지/모델로 바꿀 땐 여기만 수정) ====
IMAGE_PATH = osp.join(PROJ_ROOT, "datasets/Unity_images/data_001.png")  # 입력 이미지
YOLO_CKPT_PATH = osp.join(PROJ_ROOT, "yolox/YOLOv11xlarge_Pose(0713).pt")  # YOLO 체크포인트
GDRN_CONFIG_PATH = osp.join(PROJ_ROOT, "configs/gdrn/z_bracket/z_bracket.py")  # GDRN config
GDRN_CKPT_PATH = osp.join(PROJ_ROOT, "output/gdrn/z_bracket/8300img_15epoch/model_final.pth")  # GDRN 체크포인트

# 3D 박스 12개 모서리 (vis_bop.py / vis_pred.py와 동일한 코너 생성 순서)
EDGES = [(0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7), (6, 7), (0, 4), (1, 5), (2, 6), (3, 7)]


def load_model_bbox_corners_mm(ply_path):
    from lib.pysixd import inout

    model = inout.load_ply(ply_path, vertex_scale=1.0)  # 원본 단위(mm) 그대로
    pts = model["pts"]
    mn, mx = pts.min(0), pts.max(0)
    corners = np.array(
        [[x, y, z] for x in (mn[0], mx[0]) for y in (mn[1], mx[1]) for z in (mn[2], mx[2])], dtype=np.float64
    )
    return corners


def project(pts3d, R, t, K):
    cam = R @ pts3d.T + t.reshape(3, 1)
    proj = K @ cam
    proj = proj[:2] / proj[2]
    return proj.T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?", default=IMAGE_PATH)
    parser.add_argument("--yolo-ckpt", default=YOLO_CKPT_PATH)
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    # 실사(=GT 없는) 이미지 추론 결과 저장 위치
    parser.add_argument("--out", default=osp.join(PROJ_ROOT, "pred/image_without_GT"))
    args = parser.parse_args()

    from ultralytics import YOLO

    print(f"YOLO 모델 로드: {args.yolo_ckpt}")
    yolo = YOLO(args.yolo_ckpt)

    image = cv2.imread(args.image)  # BGR, 알파는 자동으로 버려짐
    assert image is not None, f"이미지를 읽을 수 없습니다: {args.image}"
    print(f"이미지: {args.image}  shape={image.shape}")

    yolo_res = yolo.predict(source=image, conf=args.conf, verbose=False)[0]
    n_det = len(yolo_res.boxes)
    print(f"YOLO 검출 개수: {n_det}")
    if n_det == 0:
        print("검출된 물체가 없습니다. --conf 값을 낮춰보세요.")
        return

    # GdrnPredictor.preprocessing()이 기대하는 형식: outputs[0] = Nx7 텐서
    # [x1, y1, x2, y2, obj_conf, cls_conf, category_id]
    # NOTE: category_id는 실제 BOP obj_id가 아니라 self.objs dict의 0-based 순번이다
    # (postprocessing에서 self.obj_ids[category_id]로 실제 obj_id를 찾음). objs={1:"z_bracket"}
    # 하나뿐이므로 0으로 채운다 -- yolo 모델 자체의 클래스 인덱스(0="robot_part")와는 무관.
    xyxy = yolo_res.boxes.xyxy.cpu().numpy()
    conf = yolo_res.boxes.conf.cpu().numpy()
    n = xyxy.shape[0]
    outputs_tensor = torch.zeros((n, 7), dtype=torch.float32)
    outputs_tensor[:, 0:4] = torch.from_numpy(xyxy)
    outputs_tensor[:, 4] = 1.0
    outputs_tensor[:, 5] = torch.from_numpy(conf)
    outputs_tensor[:, 6] = 0
    yolo_outputs = [outputs_tensor]

    config_path = GDRN_CONFIG_PATH
    ckpt_path = GDRN_CKPT_PATH
    camera_json_path = osp.join(PROJ_ROOT, "datasets/BOP_DATASETS/z_bracket/camera.json")
    models_dir = osp.join(PROJ_ROOT, "datasets/BOP_DATASETS/z_bracket/models")

    print(f"GDRN 체크포인트 로드: {ckpt_path}")
    gdrn_predictor = GdrnPredictor(
        config_file_path=config_path,
        ckpt_file_path=ckpt_path,
        camera_json_path=camera_json_path,
        path_to_obj_models=models_dir,
        objs={1: "z_bracket"},
    )

    data_dict = gdrn_predictor.preprocessing(outputs=yolo_outputs, image=image)
    out_dict = gdrn_predictor.inference(data_dict)
    poses = gdrn_predictor.postprocessing(data_dict, out_dict)

    print("\n추정된 포즈 (R: 3x3 회전, t: m 단위):")
    for name, pose in poses.items():
        print(f"  {name}:\n{pose}")

    # 시각화: 예측 3D 박스 + YOLO 2D 박스를 이미지에 그려서 파일로 저장 (GUI 창 없음)
    bbox3d_mm = load_model_bbox_corners_mm(osp.join(models_dir, "obj_000001.ply"))
    K = gdrn_predictor.cam
    vis = image.copy()
    for res in data_dict["cur_res"]:
        R, t_mm = res["R"], res["t"] * 1000.0  # t는 m 단위로 나옴 -> mm로 변환
        corners2d = project(bbox3d_mm, R, t_mm, K).astype(int)
        for a, b in EDGES:
            cv2.line(vis, tuple(corners2d[a]), tuple(corners2d[b]), (0, 0, 255), 3)
        bbox_est = res["bbox_est"].astype(int)
        cv2.rectangle(vis, (bbox_est[0], bbox_est[1]), (bbox_est[2], bbox_est[3]), (255, 0, 0), 2)
        c = corners2d.mean(0).astype(int)
        cv2.putText(vis, f"{res['score']:.2f}", tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(vis, "YOLO bbox=blue  pred pose=red", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    os.makedirs(args.out, exist_ok=True)
    out_path = osp.join(args.out, osp.splitext(osp.basename(args.image))[0] + "_pred.png")
    cv2.imwrite(out_path, vis)
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()

# vis_pred_bop.py — GDRNPP 추론 결과(csv)를 GT와 함께 이미지에 3D 박스로 시각화
# (BlenderProc/vis_bop.py와 같은 그리기 방식이지만, scene_gt.json 대신
#  --eval-only가 만든 예측 결과 csv를 읽는다. GPU/모델 로딩 불필요.)
#
# 사용법:
#   python vis_pred_bop.py            # test/000000 전체(최대 10장) GT(초록) vs 예측(빨강) 시각화
#   python vis_pred_bop.py 6          # im_id=6 한 장만
#   python vis_pred_bop.py "" 30      # 최대 30장
import os, sys, json, glob
import os.path as osp
import numpy as np
import cv2
from plyfile import PlyData

MODEL_PATH = "/home/smslab/gitHub/gdrnpp_bop2022/datasets/BOP_DATASETS/z_bracket/models/obj_000001.ply"
GT_SCENE_DIR = "/home/smslab/gitHub/gdrnpp_bop2022/datasets/BOP_DATASETS/z_bracket/test/000000"
RESULT_ROOT = "/home/smslab/gitHub/gdrnpp_bop2022/output/gdrn/z_bracket/8300img_15epoch"
OUT_DIR = "/home/smslab/gitHub/gdrnpp_bop2022/pred/bop_test"

# 3D 박스 12개 모서리 (vis_bop.py와 동일한 코너 생성 순서에 맞춤)
EDGES = [(0,1),(0,2),(1,3),(2,3),(4,5),(4,6),(5,7),(6,7),
         (0,4),(1,5),(2,6),(3,7)]


def load_model_bbox(path):
    """모델 ply의 3D 바운딩박스 8개 꼭짓점 (mm)"""
    ply = PlyData.read(path)
    v = ply["vertex"]
    pts = np.stack([v["x"], v["y"], v["z"]], axis=1)
    mn, mx = pts.min(0), pts.max(0)
    corners = np.array([[x, y, z] for x in (mn[0], mx[0])
                                   for y in (mn[1], mx[1])
                                   for z in (mn[2], mx[2])], dtype=np.float64)
    return corners


def project(pts3d, R, t, K):
    cam = R @ pts3d.T + t.reshape(3, 1)
    proj = K @ cam
    proj = proj[:2] / proj[2]
    return proj.T


def find_latest_result_csv():
    """가장 최근 --eval-only 실행이 만든 결과 csv를 자동으로 찾는다
    (output/gdrn/z_bracket/.../inference_epoch_X_iter_Y/z_bracket_test/*.csv)"""
    csvs = glob.glob(osp.join(RESULT_ROOT, "inference_epoch_*_iter_*", "z_bracket_test", "*.csv"))
    if not csvs:
        raise FileNotFoundError(
            f"{RESULT_ROOT}/inference_epoch_*/z_bracket_test/*.csv 없음 -- 먼저 --eval-only로 추론을 돌리세요"
        )
    csvs.sort(key=osp.getmtime)
    return csvs[-1]


def load_predictions(csv_path):
    """GDRNPP save_and_eval_results가 쓰는 형식:
    header: scene_id,im_id,obj_id,score,R,t,time  (R/t는 내부가 공백으로 구분됨)"""
    preds = {}  # (scene_id, im_id) -> [ {obj_id, R, t, score}, ... ]
    with open(csv_path) as f:
        header = f.readline().strip().split(",")
        for line in f:
            parts = line.strip().split(",")
            row = dict(zip(header, parts))
            scene_id = int(row["scene_id"])
            im_id = int(row["im_id"])
            R = np.array([float(x) for x in row["R"].split(" ")]).reshape(3, 3)
            t = np.array([float(x) for x in row["t"].split(" ")])
            preds.setdefault((scene_id, im_id), []).append(
                {"obj_id": int(row["obj_id"]), "R": R, "t": t, "score": float(row["score"])}
            )
    return preds


def main():
    filt_im = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] != "" else None
    max_n = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    csv_path = find_latest_result_csv()
    print(f"예측 결과 csv: {csv_path}")
    preds = load_predictions(csv_path)
    print(f"예측된 이미지 수: {len(preds)}")

    os.makedirs(OUT_DIR, exist_ok=True)
    scene_gt = json.load(open(osp.join(GT_SCENE_DIR, "scene_gt.json")))
    scene_cam = json.load(open(osp.join(GT_SCENE_DIR, "scene_camera.json")))
    bbox3d = load_model_bbox(MODEL_PATH)

    rgb_dir = osp.join(GT_SCENE_DIR, "rgb")
    scene_id = 0  # held-out test는 scene 000000 하나뿐

    count = 0
    for im_id_str in sorted(scene_gt.keys(), key=lambda x: int(x)):
        im_id = int(im_id_str)
        if filt_im is not None and im_id != filt_im:
            continue
        if (scene_id, im_id) not in preds:
            continue

        img_path = osp.join(rgb_dir, f"{im_id:06d}.jpg")
        if not osp.exists(img_path):
            img_path = osp.join(rgb_dir, f"{im_id:06d}.png")
        if not osp.exists(img_path):
            print(f"이미지 없음: {img_path}")
            continue
        img = cv2.imread(img_path)

        K = np.array(scene_cam[im_id_str]["cam_K"]).reshape(3, 3)

        # GT: 초록
        for inst in scene_gt[im_id_str]:
            R = np.array(inst["cam_R_m2c"]).reshape(3, 3)
            t = np.array(inst["cam_t_m2c"])
            corners2d = project(bbox3d, R, t, K).astype(int)
            for i, j in EDGES:
                cv2.line(img, tuple(corners2d[i]), tuple(corners2d[j]), (0, 255, 0), 2)

        # 예측: 빨강
        for inst in preds[(scene_id, im_id)]:
            corners2d = project(bbox3d, inst["R"], inst["t"], K).astype(int)
            for i, j in EDGES:
                cv2.line(img, tuple(corners2d[i]), tuple(corners2d[j]), (0, 0, 255), 2)
            c = corners2d.mean(0).astype(int)
            cv2.putText(
                img, f"{inst['score']:.2f}", tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
            )

        cv2.putText(img, "GT=green  Pred=red", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        out_path = osp.join(OUT_DIR, f"im{im_id:06d}.png")
        cv2.imwrite(out_path, img)
        print(f"저장: {out_path}")
        count += 1
        if count >= max_n:
            break

    print(f"\n총 {count}장 저장됨 -> {OUT_DIR}/")


if __name__ == "__main__":
    main()

# vis_pose.py — CSV의 6D pose를 이미지에 3D 바운딩박스로 시각화
# 사용법: python vis_pose.py <결과.csv> [scene_id] [im_id]
import os, sys, json
import os.path as osp
import numpy as np
import cv2
from plyfile import PlyData

DATA_ROOT = "datasets/BOP_DATASETS/itodd"
MODELS_DIR = osp.join(DATA_ROOT, "models")
OUT_DIR = "vis_output"
SCORE_THR = 0.9   # 이 점수 이상만 그림

os.makedirs(OUT_DIR, exist_ok=True)

def load_ply_bbox(obj_id):
    """모델 ply를 읽어 3D 바운딩박스 8개 꼭짓점 반환 (단위: mm)"""
    path = osp.join(MODELS_DIR, f"obj_{int(obj_id):06d}.ply")
    ply = PlyData.read(path)
    v = ply["vertex"]
    pts = np.stack([v["x"], v["y"], v["z"]], axis=1)  # (N,3)
    mn, mx = pts.min(0), pts.max(0)
    # 8 꼭짓점
    corners = np.array([[x, y, z] for x in (mn[0], mx[0])
                                   for y in (mn[1], mx[1])
                                   for z in (mn[2], mx[2])], dtype=np.float64)
    return corners

# 3D 박스 12개 모서리 연결 (꼭짓점 인덱스 쌍)
EDGES = [(0,1),(0,2),(1,3),(2,3),(4,5),(4,6),(5,7),(6,7),
         (0,4),(1,5),(2,6),(3,7)]

def project(pts3d, R, t, K):
    """3D 점을 2D로 투영. pts3d:(N,3) mm, R:(3,3), t:(3,) mm, K:(3,3)"""
    cam = (R @ pts3d.T + t.reshape(3,1))   # (3,N)
    proj = K @ cam                          # (3,N)
    proj = proj[:2] / proj[2]               # (2,N)
    return proj.T                           # (N,2)

def main():
    csv_path = sys.argv[1]
    filt_scene = int(sys.argv[2]) if len(sys.argv) > 2 else None
    filt_im = int(sys.argv[3]) if len(sys.argv) > 3 else None

    # scene_camera.json 캐시
    cam_cache = {}
    def get_K(scene_id, im_id):
        if scene_id not in cam_cache:
            p = osp.join(DATA_ROOT, "test", f"{scene_id:06d}", "scene_camera.json")
            cam_cache[scene_id] = json.load(open(p))
        K = np.array(cam_cache[scene_id][str(im_id)]["cam_K"]).reshape(3,3)
        return K

    # 이미지별로 결과 모으기
    from collections import defaultdict
    by_img = defaultdict(list)
    with open(csv_path) as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6: continue
            scene_id, im_id, obj_id, score = int(parts[0]), int(parts[1]), int(parts[2]), float(parts[3])
            R = np.array([float(x) for x in parts[4].split()]).reshape(3,3)
            t = np.array([float(x) for x in parts[5].split()])
            by_img[(scene_id, im_id)].append((obj_id, score, R, t))

    bbox_cache = {}
    count = 0
    for (scene_id, im_id), dets in sorted(by_img.items()):
        if filt_scene is not None and scene_id != filt_scene: continue
        if filt_im is not None and im_id != filt_im: continue

        img_path = osp.join(DATA_ROOT, "test", f"{scene_id:06d}", "gray", f"{im_id:06d}.tif")
        if not osp.exists(img_path):
            print(f"이미지 없음: {img_path}"); continue
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # 컬러 박스 그리려고 3채널로
        K = get_K(scene_id, im_id)

        for obj_id, score, R, t in dets:
            if score < SCORE_THR: continue
            if obj_id not in bbox_cache:
                bbox_cache[obj_id] = load_ply_bbox(obj_id)
            corners2d = project(bbox_cache[obj_id], R, t, K).astype(int)
            color = (0, 255, 0)
            for i, j in EDGES:
                cv2.line(img, tuple(corners2d[i]), tuple(corners2d[j]), color, 2)
            # 라벨
            c = corners2d.mean(0).astype(int)
            cv2.putText(img, f"obj{obj_id}:{score:.2f}", tuple(c),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        out_path = osp.join(OUT_DIR, f"scene{scene_id:06d}_im{im_id:06d}.png")
        cv2.imwrite(out_path, img)
        print(f"저장: {out_path}  (객체 {len([d for d in dets if d[1]>=SCORE_THR])}개)")
        count += 1
        if count >= 20: break  # 최대 10장

    print(f"\n총 {count}장 저장됨 → {OUT_DIR}/")

if __name__ == "__main__":
    main()
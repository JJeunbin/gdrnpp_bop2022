_base_ = ["../../_base_/gdrn_base.py"]

import glob
import os.path as osp

# NOTE: mmcv.Config.fromfile()은 이 파일을 임시 경로로 복사한 뒤 실행하기 때문에
# 여기서 __file__ 기준 상대경로를 쓰면 엉뚱한 곳을 가리킨다 (예: 0img_...가 되어버림).
# 대신 ref/z_bracket.py가 이미 갖고 있는, sys.path 기준으로 안전하게 계산된 경로를 재사용한다.
from ref import z_bracket as _ref_z_bracket

# OUTPUT_DIR을 "{학습 이미지 수}img_{epoch 수}epoch" 형태로 자동 생성.
# -> train_pbr 씬이 늘거나 TOTAL_EPOCHS를 바꿔도 폴더명이 항상 실제 값과 일치하고,
#    실험마다 손으로 지어주던 이름(예: convnext_a1_AugCosyAAE_..._zBracket)이나
#    수동 rename이 더 이상 필요 없음.
_TRAIN_PBR_ROOT = osp.join(_ref_z_bracket.dataset_root, "train_pbr")
_NUM_TRAIN_IMAGES = sum(
    len(glob.glob(osp.join(_scene, "rgb", "*")))
    for _scene in glob.glob(osp.join(_TRAIN_PBR_ROOT, "*"))
    if osp.isdir(_scene) and osp.basename(_scene) != "xyz_crop"
)
TOTAL_EPOCHS = 15  # 8300img_3epoch에서 --opts로 15까지 이어 학습 완료 -> 파일 값도 실제와 맞춤
OUTPUT_DIR = f"output/gdrn/z_bracket/{_NUM_TRAIN_IMAGES}img_{TOTAL_EPOCHS}epoch"

INPUT = dict(
    DZI_PAD_SCALE=1.5,
    TRUNCATE_FG=False,
    # VOC2012 기반 배경 합성은 사용하지 않음: BlenderProc 렌더링 단계에서 이미
    # CCTextures로 배경(바닥/벽)을 씬마다 랜덤화했기 때문에 동일한 목적이 충족됨.
    CHANGE_BG_PROB=0.0,
    COLOR_AUG_PROB=0.8,
    MIN_SIZE_TRAIN=1080,
    MAX_SIZE_TRAIN=1920,
    MIN_SIZE_TEST=1080,
    MAX_SIZE_TEST=1920,
    COLOR_AUG_TYPE="code",
    COLOR_AUG_CODE=(
        "Sequential(["
        # Sometimes(0.5, PerspectiveTransform(0.05)),
        # Sometimes(0.5, CropAndPad(percent=(-0.05, 0.1))),
        # Sometimes(0.5, Affine(scale=(1.0, 1.2))),
        "Sometimes(0.5, CoarseDropout( p=0.2, size_percent=0.05) ),"
        "Sometimes(0.4, GaussianBlur((0., 3.))),"
        "Sometimes(0.3, pillike.EnhanceSharpness(factor=(0., 50.))),"
        "Sometimes(0.3, pillike.EnhanceContrast(factor=(0.2, 50.))),"
        "Sometimes(0.5, pillike.EnhanceBrightness(factor=(0.1, 6.))),"
        "Sometimes(0.3, pillike.EnhanceColor(factor=(0., 20.))),"
        "Sometimes(0.5, Add((-25, 25), per_channel=0.3)),"
        "Sometimes(0.3, Invert(0.2, per_channel=True)),"
        "Sometimes(0.5, Multiply((0.6, 1.4), per_channel=0.5)),"
        "Sometimes(0.5, Multiply((0.6, 1.4))),"
        "Sometimes(0.1, AdditiveGaussianNoise(scale=10, per_channel=True)),"
        "Sometimes(0.5, iaa.contrast.LinearContrast((0.5, 2.2), per_channel=0.3)),"
        "Sometimes(0.5, Grayscale(alpha=(0.0, 1.0))),"
        "], random_order=True)"
        # cosy+aae; reflective metal object -> keep the strong color/contrast jitter
    ),
)

SOLVER = dict(
    # bs=24 measured ~11.7GB steady-state VRAM on the 5090 (32GB) -> bs=48 has
    # comfortable headroom (~22-24GB) and halves iters/epoch to cut wall-clock time.
    IMS_PER_BATCH=48,
    # 3 epochs (~2-2.5h at bs=48) just to confirm the full pipeline (train loop +
    # eval loop) runs end-to-end without crashing on the 8300-img pipeline-validation
    # dataset. Bump TOTAL_EPOCHS above (and re-run with --resume) for a real training
    # run later -- ANNEAL_POINT is a fraction of TOTAL_EPOCHS, so the cosine LR
    # schedule fully anneals to 0 within however many epochs this is set to.
    TOTAL_EPOCHS=TOTAL_EPOCHS,
    LR_SCHEDULER_NAME="flat_and_anneal",
    ANNEAL_METHOD="cosine",
    ANNEAL_POINT=0.72,
    OPTIMIZER_CFG=dict(_delete_=True, type="Ranger", lr=8e-4, weight_decay=0.01),  # restored to match bs=48
    WEIGHT_DECAY=0.0,
    WARMUP_FACTOR=0.001,
    WARMUP_ITERS=1000,
)

DATASETS = dict(
    TRAIN=("z_bracket_train_pbr",),
    # held-out synthetic scenes (never in train_pbr) with full GT -> real ADD scoring
    TEST=("z_bracket_test",),
)

DATALOADER = dict(
    # GPU util was bouncing 5-87% at bs=24/workers=8 -> CPU-side aug/online-XYZ render
    # was the bottleneck, not the GPU. 24 cores available; give the loader more of them.
    NUM_WORKERS=16,
    FILTER_VISIB_THR=0.3,
)

MODEL = dict(
    LOAD_DETS_TEST=False,
    PIXEL_MEAN=[0.0, 0.0, 0.0],
    PIXEL_STD=[255.0, 255.0, 255.0],
    BBOX_TYPE="AMODAL_CLIP",  # VISIB or AMODAL
    POSE_NET=dict(
        NAME="GDRN_double_mask",
        XYZ_ONLINE=True,
        NUM_CLASSES=1,
        BACKBONE=dict(
            FREEZE=False,
            PRETRAINED="timm",
            INIT_CFG=dict(
                type="timm/convnext_base",
                pretrained=True,
                in_chans=3,
                features_only=True,
                out_indices=(3,),
            ),
        ),
        ## geo head: Mask, XYZ, Region
        GEO_HEAD=dict(
            FREEZE=False,
            INIT_CFG=dict(
                type="TopDownDoubleMaskXyzRegionHead",
                in_dim=1024,  # this is num out channels of backbone conv feature
            ),
            NUM_REGIONS=64,
            # single object -> no need for class-aware heads
            XYZ_CLASS_AWARE=False,
            MASK_CLASS_AWARE=False,
            REGION_CLASS_AWARE=False,
        ),
        PNP_NET=dict(
            INIT_CFG=dict(norm="GN", act="gelu"),
            REGION_ATTENTION=True,
            WITH_2D_COORD=True,
            ROT_TYPE="allo_rot6d",
            TRANS_TYPE="centroid_z",
        ),
        LOSS_CFG=dict(
            # xyz loss ----------------------------
            XYZ_LOSS_TYPE="L1",  # L1 | CE_coor
            XYZ_LOSS_MASK_GT="visib",  # trunc | visib | obj
            XYZ_LW=1.0,
            # mask loss ---------------------------
            MASK_LOSS_TYPE="L1",  # L1 | BCE | CE
            MASK_LOSS_GT="trunc",  # trunc | visib | gt
            MASK_LW=1.0,
            # full mask loss ---------------------------
            FULL_MASK_LOSS_TYPE="L1",  # L1 | BCE | CE
            FULL_MASK_LW=1.0,
            # region loss -------------------------
            REGION_LOSS_TYPE="CE",  # CE
            REGION_LOSS_MASK_GT="visib",  # trunc | visib | obj
            REGION_LW=1.0,
            # pm loss --------------
            PM_LOSS_SYM=False,  # z_bracket is asymmetric
            PM_R_ONLY=True,  # only do R loss in PM
            PM_LW=1.0,
            # centroid loss -------
            CENTROID_LOSS_TYPE="L1",
            CENTROID_LW=1.0,
            # z loss -----------
            Z_LOSS_TYPE="L1",
            Z_LW=1.0,
        ),
    ),
)

VAL = dict(
    DATASET_NAME="z_bracket",
    SCRIPT_PATH="lib/pysixd/scripts/eval_pose_results_more.py",
    TARGETS_FILENAME="test_targets_bop19.json",
    # skip vsd/mssd/mspd: those need a full mesh renderer + per-dataset vsd_delta
    # tuning; ad/re/te/rete/proj are purely geometric (pose vs GT) and enough to
    # answer "is the pose regression head learning anything real".
    ERROR_TYPES="ad,rete,re,te,proj",
    RENDERER_TYPE="cpp",
    SPLIT="test",
    SPLIT_TYPE="",
    N_TOP=-1,  # VIVO: evaluate all instances per image (multiple z_brackets per bin)
    EVAL_CACHED=False,
    SCORE_ONLY=False,
    EVAL_PRINT_ONLY=False,
    EVAL_PRECISION=False,
    USE_BOP=True,
    SAVE_BOP_CSV_ONLY=False,  # False -> actually run the scoring, not just dump the csv
)

# no detector for z_bracket yet -> evaluate pose regression alone using GT boxes;
# eval every epoch (only 3 total) so the eval pipeline is exercised at least once
TEST = dict(EVAL_PERIOD=1, VIS=False, TEST_BBOX_TYPE="gt")

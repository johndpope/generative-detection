POSE_DIM = 4
FILL_FACTOR_DIM=1
LHW_DIM = 3
BBOX_3D_DIM = POSE_DIM + LHW_DIM
FINAL_PERTURB_RAD = 0.5

LABEL_NAME2ID = {
    'car': 0, 
    'truck': 1,
    'trailer': 2,
    'bus': 3,
    'construction_vehicle': 4,
    'bicycle': 5,
    'motorcycle': 6,
    'pedestrian': 7,
    'traffic_cone': 8,
    'barrier': 9,
}
BACKGROUND_CLASS_IDX = len(LABEL_NAME2ID)
LABEL_NAME2ID.update({'background': BACKGROUND_CLASS_IDX})

LABEL_ID2NAME = {v: k for k, v in LABEL_NAME2ID.items()}
CAM_NAMESPACE = 'CAM'

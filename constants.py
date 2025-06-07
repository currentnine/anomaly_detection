import os
current_dir = os.path.dirname(os.path.abspath(__file__))  


CHECKPOINT_DIR = "checkpoint_dir"

MVTEC_CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
    "idcard",
    "window",
    "swir"
]

BACKBONE_DEIT = "deit_base_distilled_patch16_384"
BACKBONE_CAIT = "cait_m48_448"
BACKBONE_RESNET18 = "resnet18"
BACKBONE_WIDE_RESNET50 = "wide_resnet50_2"

SUPPORTED_BACKBONES = [
    BACKBONE_DEIT,
    BACKBONE_CAIT,
    BACKBONE_RESNET18,
    BACKBONE_WIDE_RESNET50,
]
DATASET_PATH = "E:/fastflow_dataset"
CHECKPOINT_PATH = "C:/Users/wjdgu/OneDrive/Desktop/project/2024-1/capstone_design/checkpoint_dir/exp5"
BATCH_SIZE = 16
NUM_EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-5
INPUT_SIZE = 256
CHECKPOINT_INTERVAL = 2
TRAIN_STATUS = False
OUTPUT_FILE_PATH = "C:/Users/wjdgu/OneDrive/Desktop/project/2024-1/capstone_design/checkpoint_dir/exp5/output.txt"
WEIGHT_PATH = "_"
SCORE_ALL = '_'
SCORE_TEST = "_"
THRESHOLD = "_"
IMAGE_TEST_PATH = "C:/Users/wjdgu/OneDrive/Desktop/project/2024-1/capstone_design/inputE.png"
MULTIPLE_TEST_PATH = "_"

# IMAGE_TEST_PATH = "C:/jaehyuk/FastFlow/window_v3/test/bad/20221212_161421.png"
# IMAGE_prediction_PATH = "C:/Users/wjdgu/OneDrive/Desktop/project/2024-1/fastflow_code_bilel/waiting.jpg"
IMAGE_prediction_PATH = "C:/Users/wjdgu/OneDrive/Desktop/project/2024-1/capstone_design/waiting_1.png"
PREDECTION_RESUILT = 'No result...'

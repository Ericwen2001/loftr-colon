from configs.data.base import cfg


TRAIN_BASE_PATH = "assets/homo_finetune"
cfg.DATASET.TRAINVAL_DATA_SOURCE = "Homo"
cfg.DATASET.TRAIN_DATA_ROOT = "data/homo/test"
cfg.DATASET.TRAIN_NPZ_ROOT = f"{TRAIN_BASE_PATH}"
cfg.DATASET.TRAIN_LIST_PATH = f"{TRAIN_BASE_PATH}/homo_list_path.txt"
cfg.DATASET.TRAIN_INTRINSIC_PATH = f"{TRAIN_BASE_PATH}/intrinsics.npz"

TEST_BASE_PATH = "assets/homo_val"
cfg.DATASET.TEST_DATA_SOURCE = "Homo"
cfg.DATASET.VAL_DATA_ROOT = cfg.DATASET.TEST_DATA_ROOT = "data/homo_val/test"
cfg.DATASET.VAL_NPZ_ROOT = cfg.DATASET.TEST_NPZ_ROOT = TEST_BASE_PATH
cfg.DATASET.VAL_LIST_PATH = cfg.DATASET.TEST_LIST_PATH = f"{TEST_BASE_PATH}/val_list_path.txt"
cfg.DATASET.VAL_INTRINSIC_PATH = cfg.DATASET.TEST_INTRINSIC_PATH = f"{TEST_BASE_PATH}/intrinsics.npz"
cfg.DATASET.MIN_OVERLAP_SCORE_TEST = 0.0   # for both test and val

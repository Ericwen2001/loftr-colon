from configs.data.base import cfg


TRAIN_BASE_PATH = "assets/homo_finetune"
cfg.DATASET.TRAINVAL_DATA_SOURCE = "Homo"
cfg.DATASET.TRAIN_DATA_ROOT = "data/homo/test"
cfg.DATASET.TRAIN_NPZ_ROOT = f"{TRAIN_BASE_PATH}"
cfg.DATASET.TRAIN_LIST_PATH = f"{TRAIN_BASE_PATH}/homo_list_path.txt"
cfg.DATASET.TRAIN_INTRINSIC_PATH = f"{TRAIN_BASE_PATH}/intrinsics.npz"


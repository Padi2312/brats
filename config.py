import torch
from datetime import datetime
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False
DATASET_PATH = "C:\\bratsdata"  # '../data/BraTS2021_Training_Data'
MODEL_FOLDER_PATH = "./output_models"
CHECKPOINT_PATH = "checkpoint.pt"

INPUT_CHANNELS = 4
OUTPUT_CHANNELS = 4

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64

BATCH_SIZE = 2
LEARNING_RATE = 0.0001  # 0.0001
NUM_EPOCHS = 20

VAL_SPLIT = 0.3
SEED = 126728

MODEL_PATH = (
    MODEL_FOLDER_PATH
    + "/"
    + "model_"
    + f"{str(datetime.now().day)}-{str(datetime.now().month)}-{str(datetime.now().year)}-{str(datetime.now().minute)}_"
    + f"B{BATCH_SIZE}_E{NUM_EPOCHS}_LR{LEARNING_RATE}_WxH_{IMAGE_WIDTH}x{IMAGE_HEIGHT}"
    + ".pt"
)


if not os.path.exists(MODEL_FOLDER_PATH):
    os.mkdir(MODEL_FOLDER_PATH)

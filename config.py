import torch
from datetime import datetime
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# PIN_MEMORY = True if DEVICE == "cuda" else False
PIN_MEMORY = False
DATASET_PATH = "C:\\bratsdata"  # '../data/BraTS2021_Training_Data'
MODEL_FOLDER_PATH = "./output_models"
MODEL_PATH = (
    MODEL_FOLDER_PATH + "/" + "unet_model" + str(datetime.now().microsecond) + ".pt"
)

CHECKPOINT_PATH = "checkpoint.pt"

INPUT_CHANNELS = 4
OUTPUT_CHANNELS = 4

IMAGE_WIDTH = 192
IMAGE_HEIGHT = 192

BATCH_SIZE = 16 
LEARNING_RATE = 0.0001  # 0.0001
NUM_EPOCHS = 20

TEST_VALIDATION_SPLIT = 0.2
SEED = 126728


if not os.path.exists(MODEL_FOLDER_PATH):
    os.mkdir(MODEL_FOLDER_PATH)

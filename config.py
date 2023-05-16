import torch
from datetime import datetime
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False
DATASET_PATH = "C:\\bratsdata"  # '../data/BraTS2021_Training_Data'
DATASET_PROCESSED_PATH_128 = "D:\\bratsdata_processed_128"
DATASET_PROCESSED_PATH_128_CC = "D:\\bratsdata_processed_128_cc"
DATASET_PROCESSED_PATH_128_CC_LINUX = "/mnt/d/bratsdata_processed"
DATASET_PROCESSED_PATH_128_NORMED = "D:\\bratsdata_processed_128_z_norm"
DATASET_PROCESSED_PATH_64 = "D:\\bratsdata_processed_64"
DATASET_PROCESSED_PATH = DATASET_PROCESSED_PATH_128_CC

MODEL_FOLDER_PATH = "./output_models"

INPUT_CHANNELS = 4
OUTPUT_CHANNELS = 4

IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128

if not os.path.exists(MODEL_FOLDER_PATH):
    os.mkdir(MODEL_FOLDER_PATH)

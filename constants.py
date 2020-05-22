# Network Constants

USE_SMALL_NET = False

CROP_SIZE = 227
CROP_PAD = 2
MAX_TRACK_LENGTH = 32

import os.path

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs/")
DATA_DIR = os.path.join(
    os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir, "Datasets"
)

GPU_ID = "0"

# Drawing constants
OUTPUT_WIDTH = 640
OUTPUT_HEIGHT = 480
PADDING = 2

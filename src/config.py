import os

MODEL_PATH = "outputs/vehicle_detection_v1/weights/best.pt"
VIDEO_INPUT = "data/raw/video-data-vehicles/"
VIDEO_OUTPUT = "results/video/"
IMG_INPUT = "data/raw/img-data-vehicles/"
IMG_OUTPUT = "results/img/"
REPORT_OUTPUT = "results/traffic_frequency_report.csv"

CONF_THRESHOLD = 0.45
IMG_SIZE = 640

MAX_AGE = 45
N_INIT = 3
MAX_COSINE_DIST = 0.2

LINE_START = (100, 500)
LINE_END = (1180, 500)

TIME_WINDOW_SECONDS = 60

CLASS_NAMES = {
    0: 'bike',
    1: 'bus',
    2: 'car',
    3: 'motorbike',
    4: 'plane',
    5: 'ship',
    6: 'train',
    7: 'truck'
}
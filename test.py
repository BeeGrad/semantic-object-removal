import cv2
import numpy as np
from utils.utils import calculate_psnr
from utils.testutils import freely_select_from_image, select_by_edge
from scripts.config import Config

cfg = Config()
original_image = cv2.imread(cfg.test_im_path)

if (cfg.test_mask_method == "freely_select_from_image"):
    input_image = freely_select_from_image(original_image)

if (cfg.test_mask_method == "select_by_edge"):
    input_image = select_by_edge(original_image)

print(calculate_psnr(input_image, original_image))
# Take output with pre-trained network

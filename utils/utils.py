import numpy as np
import math
from scripts.config import Config

# Math utilities for model creating, training, and testing.

cfg = Config()

def calculate_psnr(img1, img2):
    mse = np.mean( (img2 - img1) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = cfg.max_pixel_value
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

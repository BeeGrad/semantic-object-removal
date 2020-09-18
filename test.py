import cv2
import numpy as np
from utils.utils import calculate_psnr
from utils.testutils import freely_select_from_image, select_by_edge
from scripts.config import Config
from models.mathematicalmodels.model import InpaintMathematical
from matplotlib import pyplot as plt

cfg = Config()
original_image = cv2.imread(cfg.test_im_path)
cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

if (cfg.test_mask_method == "freely_select_from_image"):
    input_image, mask = freely_select_from_image(original_image)

if (cfg.test_mask_method == "select_by_edge"):
    input_image = select_by_edge(original_image)

if (cfg.test_inpaint_method == "mathematical"):
    inpaint = InpaintMathematical(input_image, mask, cfg.freely_select_mask_size)
    output = inpaint.run()

print(calculate_psnr(input_image, original_image))

fig=plt.figure(figsize=(1, 3))
fig.add_subplot(1, 3, 1)
plt.imshow(original_image)
fig.add_subplot(1, 3, 2)
plt.imshow(input_image)
fig.add_subplot(1, 3, 3)
plt.imshow(output)
plt.show()
# Take output with pre-trained network

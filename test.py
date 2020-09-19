import cv2
import numpy as np
from utils.utils import calculate_psnr
from utils.testutils import freely_select_from_image, select_by_edge
from scripts.config import Config
from models.mathematicalmodels.model import InpaintMathematical
from models.edgeconnect.model import EdgeConnect
from matplotlib import pyplot as plt

cfg = Config()
original_image = cv2.imread(cfg.test_im_path)

if (cfg.test_mask_method == "freely_select_from_image"):
    input_image, mask = freely_select_from_image(original_image)

if (cfg.test_mask_method == "select_by_edge"):
    input_image = select_by_edge(original_image)

original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

if (cfg.test_inpaint_method == "Mathematical"):
    inpaint = InpaintMathematical(input_image, mask, cfg.freely_select_mask_size)
    output = inpaint.run()

if (cfg.test_inpaint_method == "EdgeConnect"):
    inpaint = EdgeConnect()
    output = inpaint.single_test(input_image, mask)
    print(output.shape)

print(calculate_psnr(input_image, original_image))
print(calculate_psnr(output, original_image))
print(output.max())

fig=plt.figure(figsize=(2, 2))
fig.add_subplot(2, 2, 1)
plt.imshow(original_image)
fig.add_subplot(2, 2, 2)
plt.imshow(input_image)
fig.add_subplot(2, 2, 3)
plt.imshow(mask)
fig.add_subplot(2, 2, 4)
plt.imshow(output)
plt.show()
# Take output with pre-trained network

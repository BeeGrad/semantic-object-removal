import cv2
import numpy as np
from utils.utils import calculate_psnr
from utils.testutils import freely_select_from_image, select_by_edge
from scripts.config import Config
from models.mathematicalmodels.model import InpaintMathematical
from models.edgeconnect.model import EdgeConnect
from models.contextual.model import GenerativeContextual
from matplotlib import pyplot as plt

cfg = Config()
original_image = cv2.imread(cfg.test_im_path)
print("Original Image is loaded")

if (cfg.test_mask_method == "freely_select_from_image"):
    input_image, mask, img_gray, edge_org = freely_select_from_image(original_image)

if (cfg.test_mask_method == "select_by_edge"):
    input_image = select_by_edge(original_image)

# Opencv saves image channels as BGR, from now on to show those images correctly
# We convert images BGR2RGB
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

if (cfg.test_inpaint_method == "Mathematical"):
    inpaint = InpaintMathematical(input_image, mask, cfg.freely_select_mask_size)
    output = inpaint.run()

if (cfg.test_inpaint_method == "EdgeConnect"):
    inpaint = EdgeConnect()
    output, edge_generated = inpaint.single_test(input_image, mask, img_gray, edge_org)

if (cfg.test_inpaint_method == "Contextual"):
    inpaint = GenerativeContextual()
    output_contextual = inpaint.single_test(input_image, mask)

original_image = original_image/255
input_image = input_image/255
# print(f"PSNR value of masked image: {calculate_psnr(input_image, original_image, mask)}")
print(f"PSNR value of masked image: {calculate_psnr(input_image, original_image)}]")
print(f"PSNR value of inpainted image: {calculate_psnr(output, original_image)}]")
print(original_image)
# Context Section
if (cfg.test_inpaint_method == "Contextual"):
    fig=plt.figure(figsize=(2, 2))
    fig.add_subplot(2, 2, 1)
    plt.imshow(original_image)
    fig.add_subplot(2, 2, 2)
    plt.imshow(input_image)
    fig.add_subplot(2, 2, 3)
    plt.imshow(output_contextual)
    # fig.add_subplot(2, 2, 4)
    # plt.imshow(edge_org)
    plt.show()

# EdgeConnect Section
if (cfg.test_inpaint_method == "EdgeConnect"):
    fig=plt.figure(figsize=(3, 2))
    fig.add_subplot(3, 2, 1)
    plt.imshow(original_image)
    fig.add_subplot(3, 2, 2)
    plt.imshow(input_image)
    fig.add_subplot(3, 2, 3)
    plt.imshow(img_gray)
    fig.add_subplot(3, 2, 4)
    plt.imshow(edge_org)
    fig.add_subplot(3, 2, 5)
    plt.imshow(edge_generated)
    fig.add_subplot(3, 2, 6)
    plt.imshow(output)
    plt.show()

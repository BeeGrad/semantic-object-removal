import os
import subprocess
import glob
import cv2
import numpy as np
from utils.utils import calculate_psnr
from utils.testutils import freely_select_from_image, select_by_edge, select_by_train_mask
from scripts.config import Config
from models.mathematicalmodels.model import InpaintMathematical
from models.edgeconnect.model import EdgeConnect
from models.contextual.model import GenerativeContextual
from models.beemodels.unifiedModel import EdgeContextUnifiedModel
from matplotlib import pyplot as plt
from skimage.measure import compare_ssim

cfg = Config()
original_image = cv2.imread(cfg.test_im_path)
print("Original Image is loaded")
print(original_image)

if cfg.test_mask_method == "freely_select_from_image":
    input_image, mask, img_gray, edge_org = freely_select_from_image(original_image)

if cfg.test_mask_method == "select_by_edge":
    input_image = select_by_edge(original_image)

if (cfg.test_mask_method == "select_by_train_mask"):
    input_image, img_gray, edge_org, mask = select_by_train_mask(original_image/255.0)

# Opencv saves image channels as BGR, from now on to show those images correctly
# We convert images BGR2RGB
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

fig=plt.figure(figsize=(2, 2))
fig.add_subplot(2, 2, 1)
plt.imshow(input_image)
fig.add_subplot(2, 2, 2)
plt.imshow(img_gray)
fig.add_subplot(2, 2, 3)
plt.imshow(edge_org)
fig.add_subplot(2, 2, 4)
plt.imshow(mask)
plt.show()

# Inpaint Models
print("Mathematical Model Section Started! ...")
inpaintMath = InpaintMathematical(input_image, mask, cfg.freely_select_mask_size)
outputMath = inpaintMath.run()
print("Mathematical Model Section Ended!")

print("EdgeConnect Model Section Started! ...")
inpaintEdge = EdgeConnect()
outputEdge, edge_generated = inpaintEdge.single_test(input_image, mask, img_gray, edge_org)
print("EdgeConnect Model Section Ended!")

print("Contextual Model Section Started! ...")
inpaintContext = GenerativeContextual()
outputContextual = inpaintContext.single_test(input_image, mask)
print("Contextual Model Section Ended!")

print("Unified Model Section Started! ...")
inpaintUnified = EdgeContextUnifiedModel()
outputUnified, halfOutputUnified = inpaintUnified.single_test(input_image, mask, img_gray, edge_org)
print("Unified Model Section Ended!")
#######

original_image = original_image/255
input_image = input_image/255

# print(f"PSNR value of masked image: {calculate_psnr(input_image, original_image, mask)}")
print(f"PSNR value of masked image: {calculate_psnr(input_image, original_image)}]")
print(f"PSNR value of Math inpainted image: {calculate_psnr(outputMath/255, original_image)}]")
print(f"PSNR value of Contextual inpainted image: {calculate_psnr(outputContextual, original_image)}]")
print(f"PSNR value of Edge inpainted image: {calculate_psnr(outputEdge, original_image)}]")
print(f"PSNR value of Unified inpainted image: {calculate_psnr(outputUnified, original_image)}]")

print(f"SSIM value of masked image: {compare_ssim(original_image, input_image, data_range=1, win_size=11, multichannel=True)}]")
print(f"SSIM value of Math inpainted image: {compare_ssim(original_image, outputMath/255, data_range=1, win_size=11, multichannel=True)}]")
print(f"SSIM value of Contextual inpainted image: {compare_ssim(original_image, outputContextual, data_range=1, win_size=11, multichannel=True)}]")
print(f"SSIM value of Edge inpainted image: {compare_ssim(original_image, outputEdge, data_range=1, win_size=11, multichannel=True)}]")
print(f"SSIM value of Unified inpainted image: {compare_ssim(original_image, outputUnified, data_range=1, win_size=11, multichannel=True)}]")

# Math Section
fig=plt.figure(figsize=(2, 2))
fig.add_subplot(2, 2, 1)
plt.imshow(original_image)
fig.add_subplot(2, 2, 2)
plt.imshow(input_image)
fig.add_subplot(2, 2, 3)
plt.imshow(outputMath)
plt.show()

# Context Section
fig=plt.figure(figsize=(2, 2))
fig.add_subplot(2, 2, 1)
plt.imshow(original_image)
fig.add_subplot(2, 2, 2)
plt.imshow(input_image)
fig.add_subplot(2, 2, 3)
plt.imshow(outputContextual)
plt.show()

# EdgeConnect Section
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
plt.imshow(outputEdge)
plt.show()

# Unified Section
fig=plt.figure(figsize=(2, 2))
fig.add_subplot(2, 2, 1)
plt.imshow(original_image)
fig.add_subplot(2, 2, 2)
plt.imshow(input_image)
fig.add_subplot(2, 2, 3)
plt.imshow(outputUnified)
fig.add_subplot(2, 2, 4)
plt.imshow(halfOutputUnified)
plt.show()

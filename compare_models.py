import cv2
import numpy as np
from utils.utils import calculate_psnr
from utils.testutils import select_by_train_mask
from scripts.config import Config
from models.beemodels.vanillaGAN import VanillaGAN
from models.beemodels.fpnModel import fpnGan
from matplotlib import pyplot as plt
from skimage.measure import compare_ssim
from models.beemodels.fpnNetwork import FPN
from models.beemodels.fpnNetwork import InpaintingModel as InpaintingModelFPN
from models.beemodels.vanillaNetwork import InpaintingModel
from skimage.measure import compare_ssim
import torchvision
import torch
from scripts.dataOperations import DataRead

cfg = Config()
data = DataRead()
data.create_data_loaders()

for i in range(1):
    inpaint_model = InpaintingModel().to(cfg.DEVICE)
    psnrVanilla = []
    ssimVanilla = []

    for i, images in enumerate(data.test_loader):
        images , masked_images, masks = data.return_inputs_fpn(images[0])

        images = images.to(cfg.DEVICE)
        masked_images = masked_images.to(cfg.DEVICE)
        masks = masks.to(cfg.DEVICE)

        out, gen_loss, dis_loss = inpaint_model.step(masked_images, masks, images)
        out = (out * masks) + (images * (1 - masks))

        compare_img = images.squeeze().permute(0,2,3,1).cpu().detach().numpy()
        compare_out = out.squeeze().permute(0,2,3,1).cpu().detach().numpy()

        for i in range(compare_img.shape[0]):
            psnr = calculate_psnr(compare_img[i], compare_out[i])
            psnrVanilla.append(psnr)
            ssim = compare_ssim(compare_img[i], compare_out[i], data_range=1, win_size=11, multichannel=True)
            ssimVanilla.append(ssim)

print(f"PSNR Average for Vanilla is {sum(psnrVanilla)/len(psnrVanilla)}!")
print(f"SSIM Average for Vanilla is {sum(ssimVanilla)/len(ssimVanilla)}!")

for i in range(1):
    fpn = FPN().to(cfg.DEVICE)
    inpaint_model = InpaintingModelFPN().to(cfg.DEVICE)
    psnrFPN = []
    ssimFPN = []

    for i, images in enumerate(data.test_loader):
        images , masked_images, masks = data.return_inputs_fpn(images[0])

        images = images.to(cfg.DEVICE)
        masked_images = masked_images.to(cfg.DEVICE)
        masks = masks.to(cfg.DEVICE)

        o1, o2, o3, o4 = fpn(masked_images)
        out, gen_loss, dis_loss = inpaint_model.step(o1, masks, images)
        out = (out * masks) + (images * (1 - masks))

        compare_img = images.squeeze().permute(0,2,3,1).cpu().detach().numpy()
        compare_out = out.squeeze().permute(0,2,3,1).cpu().detach().numpy()

        for i in range(compare_img.shape[0]):
            psnr = calculate_psnr(compare_img[i], compare_out[i])
            psnrFPN.append(psnr)
            ssim = compare_ssim(compare_img[i], compare_out[i], data_range=1, win_size=11, multichannel=True)
            ssimFPN.append(ssim)


print(f"PSNR Average for FPN is {sum(psnrFPN)/len(psnrFPN)}!")
print(f"SSIM Average for FPN is {sum(ssimFPN)/len(ssimFPN)}!")

for i in range(1):
    psnrMasked = []
    ssimMasked = []
    for i, images in enumerate(data.test_loader):
        images , masked_images, masks = data.return_inputs_fpn(images[0])
        images = images.permute(0,2,3,1).numpy()
        masked_images = masked_images.permute(0,2,3,1).numpy()
        for i in range(images.shape[0]):
            psnr = calculate_psnr(images[i], masked_images[i])
            psnrMasked.append(psnr)
            ssim = compare_ssim(images[i], masked_images[i], data_range=1, win_size=11, multichannel=True)
            ssimMasked.append(ssim)

print(f"PSNR Average for masked is {sum(psnrMasked)/len(psnrMasked)}!")
print(f"SSIM Average for masked is {sum(ssimMasked)/len(ssimMasked)}!")

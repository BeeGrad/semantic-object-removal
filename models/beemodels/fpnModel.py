import torch
import torch.nn as nn
import torch.optim as optim
from models.beemodels.fpnNetwork import FPN
from models.beemodels.fpnNetwork import InpaintingModel
from scripts.config import Config

cfg = Config()

class fpnGan():
    def __init__(self):
        super(fpnGan, self).__init__()

        self.iteration = 0

        self.fpn = FPN()
        self.inpaint_model = InpaintingModel()

        self.fpn_optimizer = optim.Adam(params=self.fpn.parameters(),
            lr=float(cfg.edge_LR), betas=(cfg.edge_BETA1, cfg.edge_BETA2))

        self.gen_optimizer = optim.Adam(params=self.inpaint_model.generator.parameters(),
            lr=float(cfg.edge_LR), betas=(cfg.edge_BETA1, cfg.edge_BETA2))

        self.dis_optimizer = optim.Adam(params=self.inpaint_model.discriminator.parameters(),
            lr=float(cfg.edge_LR) * float(cfg.edge_D2G_LR), betas=(cfg.edge_BETA1, cfg.edge_BETA2))

    def run(self, data):
        data.create_data_loaders()

        for i in range(self.iteration, cfg.epoch_num):
            self.iteration += 1
            psnr_values = []
            for i, images in enumerate(data.train_loader):
                self.gen_optimizer.zero_grad()
                self.dis_optimizer.zero_grad()
                self.fpn_optimizer.zero_grad()

                images , masked_images, masks = data.return_inputs_fpn(images[0]) # (3, 256, 256)

                o1, o2, o3, o4 = self.fpn(masked_images)
                out, gen_loss, dis_loss, logs = self.inpaint_model.step(o1, masks, images)
                print(f"Shape of Masked Images (Input of FPN):{masked_images.shape}")
                print(f"Shape of Features (Input of Inpaint GAN):{o1.shape}")
                print(f"Shape of Inpainted Image (Output of Inpaint GAN):{out.shape}")

                gen_loss.backward(retain_graph=True)
                self.gen_optimizer.step()
                self.fpn_optimizer.step()

                dis_loss.backward(retain_graph=True)
                self.dis_optimizer.step()

                break
            break

import torch
import torch.nn as nn
import torch.optim as optim
from scripts.config import Config
from models.beemodels.vanillaNetwork import InpaintingModel


cfg = Config()

class VanillaGAN():
    def __init__(self):
        super(VanillaGAN, self).__init__()

        self.iteration = 0
        self.inpaint_model = InpaintingModel()

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

                images , masked_images, masks = data.return_inputs_fpn(images[0]) # (3, 256, 256)

                out, gen_loss, dis_loss, logs = self.inpaint_model.step(images, masks)
                print(f"Shape of Inpainted Image (Output of Inpaint GAN):{out.shape}")

                gen_loss.backward()
                self.gen_optimizer.step()

                dis_loss.backward()
                self.dis_optimizer.step()

                if(i%1000==0):
                    print(f"{i}/{len(data.train_loader)}")

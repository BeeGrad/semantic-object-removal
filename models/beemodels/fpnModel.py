import torch
import torch.nn as nn
import torch.optim as optim
from models.beemodels.fpnNetwork import FPN
from models.beemodels.fpnNetwork import InpaintingModel
from scripts.config import Config
from utils.utils import calculate_psnr, show_sample_input_data_context

cfg = Config()

class fpnGan():
    def __init__(self):
        super(fpnGan, self).__init__()

        self.iteration = 0

        self.fpn = FPN().to(cfg.DEVICE)
        self.inpaint_model = InpaintingModel().to(cfg.DEVICE)

        self.fpn_optimizer = optim.Adam(params=self.fpn.parameters(),
            lr=float(cfg.edge_LR), betas=(cfg.edge_BETA1, cfg.edge_BETA2))

        self.gen_optimizer = optim.Adam(params=self.inpaint_model.generator.parameters(),
            lr=float(cfg.edge_LR), betas=(cfg.edge_BETA1, cfg.edge_BETA2))

        self.dis_optimizer = optim.Adam(params=self.inpaint_model.discriminator.parameters(),
            lr=float(cfg.edge_LR) * float(cfg.edge_D2G_LR), betas=(cfg.edge_BETA1, cfg.edge_BETA2))

    def run(self, data):
        data.create_data_loaders()

        if cfg.loadModel:
            self.load()

        for i in range(self.iteration, cfg.epoch_num):
            self.iteration += 1
            psnr_values = []
            for i, images in enumerate(data.train_loader):
                self.gen_optimizer.zero_grad()
                self.dis_optimizer.zero_grad()
                self.fpn_optimizer.zero_grad()

                images , masked_images, masks = data.return_inputs_fpn(images[0]) # (3, 256, 256)

                if cfg.show_sample_data:
                    show_sample_input_data_context(images, masked_images, masks)
                    cfg.show_sample_data = False

                images = images.to(cfg.DEVICE)
                masked_images = masked_images.to(cfg.DEVICE)
                masks = masks.to(cfg.DEVICE)

                o1, o2, o3, o4 = self.fpn(masked_images)
                out, gen_loss, dis_loss, logs = self.inpaint_model.step(o1, masks, images)
                out = (out * masks) + (images * (1 - masks))

                show_sample_input_data_context(masked_images, out, masks)

                print(f"Shape of Masked Images (Input of FPN):{masked_images.shape}")
                print(f"Shape of Features (Input of Inpaint GAN):{o1.shape}")
                print(f"Shape of Inpainted Image (Output of Inpaint GAN):{out.shape}")

                gen_loss.backward()
                self.gen_optimizer.step()
                self.fpn_optimizer.step()

                dis_loss.backward()
                self.dis_optimizer.step()

                psnr = calculate_psnr(images.squeeze().cpu().detach().numpy(), out.squeeze().cpu().detach().numpy())
                print(psnr)
                psnr_values.append(psnr)

                if(i%1000==0):
                    print(f"{i}/{len(data.train_loader)}")
                break

            print(f"Epoch {self.iteration} is done!")
            print(f"PSNR Average for Epoch {self.iteration} is {sum(psnr_values)/len(psnr_values)}!")
            self.save()

    def save(self):
        """
        Input:
            none
        Output:
            none
        Description:
            Saves 3 pytroch model for fpn and inpaint models
        """
        torch.save({
            'iteration': self.iteration,
            'generator': self.fpn.state_dict()
        }, cfg.fpn_fpnNetwork_path)

        torch.save({
            'discriminator': self.inpaint_model.discriminator.state_dict()
        }, cfg.fpn_inpaint_disc_path)

        torch.save({
            'generator': self.inpaint_model.generator.state_dict()
        }, cfg.fpn_inpaint_gen_path)
        print("Models are saved!")

    def load(self):
        """
        Input:
            none
        Output:
            none
        Description:
            Load 3 pytorch model for fpn an inpaint models for training
        """
        fpnModel = torch.load(cfg.fpn_fpnNetwork_path, map_location=lambda storage, loc: storage)
        inpaintDisc = torch.load(cfg.fpn_inpaint_disc_path, map_location=lambda storage, loc: storage)
        inpaintGen = torch.load(cfg.fpn_inpaint_gen_path, map_location=lambda storage, loc: storage)

        self.iteration = fpnModel['iteration']
        self.fpn.load_state_dict(fpnModel['generator'])
        self.inpaint_model.generator.load_state_dict(inpaintGen['generator'])
        self.inpaint_model.discriminator.load_state_dict(inpaintDisc['discriminator'])
        print("Models are loaded!")

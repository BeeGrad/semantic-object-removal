import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
from models.edgeconnect.network import InpaintGenerator, EdgeGenerator, Discriminator
from scripts.loss import AdversarialLoss, PerceptualLoss, StyleLoss
from scripts.config import Config
from skimage.feature import canny
from skimage.color import rgb2gray

cfg = Config()

class BaseModel(nn.Module):
    """
    Input:
        none
    Output:
        none
    Description:
        Base Model for Edge and Inpainting Models.
    """
    def __init__(self):
        super(BaseModel, self).__init__()
        self.iteration = 0

    def save(self):
        pass
    def load(self):
        pass

class EdgeModel(BaseModel):
    """
    Input:
        none
    Output:
        none
    Description:
        Class of Edge GAN model. All the necessary adjustment are don in this class
    """
    def __init__(self):
        super(EdgeModel, self).__init__()
        self.generator = EdgeGenerator()
        self.discriminator = Discriminator()

        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss()
        self.style_loss = StyleLoss()
        self.adversarial_loss = AdversarialLoss(type=cfg.GAN_LOSS)

        self.gen_optimizer = optim.Adam(params=self.generator.parameters(),
            lr=float(cfg.LR), betas=(cfg.BETA1, cfg.BETA2))

        self.dis_optimizer = optim.Adam(params=self.discriminator.parameters(),
            lr=float(cfg.LR) * float(cfg.D2G_LR), betas=(cfg.BETA1, cfg.BETA2))

    def step(self, images, edges, masks):
        """
        Input:
            images: original images
            edges: images with only edge information
            masks: images with mask information
        Output:
            outputs: output of the forward function
            gen_loss: Adverserial Loss of the concat of images and fake outputs
            dis_loss: Sum of loss for real and fake inputs
            logs: Dict of losses
        Description:
            1- Find fake output
            2- Calculate loss for discriminator
            3- Calculate loss for generator
        """
        self.iteration += 1

        # zero optimizers
        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()


        # process outputs
        outputs = self(images, edges, masks)
        gen_loss = 0
        dis_loss = 0


        # discriminator loss
        dis_input_real = torch.cat((images, edges), dim=1)
        dis_input_fake = torch.cat((images, outputs.detach()), dim=1)
        dis_real, dis_real_feat = self.discriminator(dis_input_real)        # in: (grayscale(1) + edge(1))
        dis_fake, dis_fake_feat = self.discriminator(dis_input_fake)        # in: (grayscale(1) + edge(1))
        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2


        # generator adversarial loss
        gen_input_fake = torch.cat((images, outputs), dim=1)
        gen_fake, gen_fake_feat = self.discriminator(gen_input_fake)        # in: (grayscale(1) + edge(1))
        gen_gan_loss = self.adversarial_loss(gen_fake, True, False)
        gen_loss += gen_gan_loss


        # generator feature matching loss
        gen_fm_loss = 0
        for i in range(len(dis_real_feat)):
            gen_fm_loss += self.l1_loss(gen_fake_feat[i], dis_real_feat[i].detach())
        gen_fm_loss = gen_fm_loss * cfg.FM_LOSS_WEIGHT
        gen_loss += gen_fm_loss


        # create logs
        logs = [
            ("l_d1", dis_loss.item()),
            ("l_g1", gen_gan_loss.item()),
            ("l_fm", gen_fm_loss.item()),
        ]

        return outputs, gen_loss, dis_loss, logs

    def forward(self, images, edges, masks):
        """
        Input:
            images: original images
            edges: images with only edge information
            masks: images with mask information
        Output:
            outputs: output of the generator
        Description:
            Masks edges and images. Then concat them and give it to generator as input
        """
        edges_masked = (edges * (1 - masks))
        images_masked = (images * (1 - masks)) + masks
        inputs = torch.cat((images_masked, edges_masked, masks), dim=1)
        outputs = self.generator(inputs)                                    # in: [grayscale(1) + edge(1) + mask(1)]
        return outputs

    def backward(self, gen_loss=None, dis_loss=None):
        """
        Input:
            none
        Output:
            none
        Description:
            step both optimizers
        """

        if dis_loss is not None:
            dis_loss.backward()
        self.dis_optimizer.step()

        if gen_loss is not None:
            gen_loss.backward()
        self.gen_optimizer.step()

class InpaintingModel(BaseModel):
    """
    Input:
        none
    Output:
        none
    Description:
        Class of Inpainting GAN model. All the necessary adjustment are don in this class
    """
    def __init__(self):
        super(InpaintingModel, self).__init__()
        self.generator = InpaintGenerator()
        self.discriminator = Discriminator(in_channels=3)

        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss()
        self.style_loss = StyleLoss()
        self.adversarial_loss = AdversarialLoss(type=cfg.GAN_LOSS)

        self.gen_optimizer = optim.Adam(params=self.generator.parameters(),
            lr=float(cfg.LR), betas=(cfg.BETA1, cfg.BETA2))

        self.dis_optimizer = optim.Adam(params=self.discriminator.parameters(),
            lr=float(cfg.LR) * float(cfg.D2G_LR), betas=(cfg.BETA1, cfg.BETA2))

    def step(self, images, edges, masks):
        """
        Input:
            images: original images
            edges: images with only edge information
            masks: images with mask information
        Output:
            outputs: output of the forward function
            gen_loss: Adverserial Loss of the concat of images and fake outputs
            dis_loss: Sum of loss for real and fake inputs
            logs: Dict of losses
        Description:
            1- Find fake output
            2- Calculate loss for discriminator
            3- Calculate loss for generator
        """
        self.iteration += 1

        # zero optimizers
        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()


        # process outputs
        outputs = self(images, edges, masks)
        gen_loss = 0
        dis_loss = 0


        # discriminator loss
        dis_input_real = images
        dis_input_fake = outputs.detach()
        dis_real, _ = self.discriminator(dis_input_real)                    # in: [rgb(3)]
        dis_fake, _ = self.discriminator(dis_input_fake)                    # in: [rgb(3)]
        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2


        # generator adversarial loss
        gen_input_fake = outputs
        gen_fake, _ = self.discriminator(gen_input_fake)        # in: (grayscale(1) + edge(1))
        gen_gan_loss = self.adversarial_loss(gen_fake, True, False)
        gen_loss += gen_gan_loss


        # generator adversarial loss
        gen_input_fake = outputs
        gen_fake, _ = self.discriminator(gen_input_fake)                    # in: [rgb(3)]
        gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * cfg.INPAINT_ADV_LOSS_WEIGHT
        gen_loss += gen_gan_loss


        # generator l1 loss
        gen_l1_loss = self.l1_loss(outputs, images) * cfg.L1_LOSS_WEIGHT / torch.mean(masks)
        gen_loss += gen_l1_loss


        # generator perceptual loss
        gen_content_loss = self.perceptual_loss(outputs, images)
        gen_content_loss = gen_content_loss * cfg.CONTENT_LOSS_WEIGHT
        gen_loss += gen_content_loss


        # generator style loss
        gen_style_loss = self.style_loss(outputs * masks, images * masks)
        gen_style_loss = gen_style_loss * cfg.STYLE_LOSS_WEIGHT
        gen_loss += gen_style_loss


        # create logs
        logs = [
            ("l_d2", dis_loss.item()),
            ("l_g2", gen_gan_loss.item()),
            ("l_l1", gen_l1_loss.item()),
            ("l_per", gen_content_loss.item()),
            ("l_sty", gen_style_loss.item()),
        ]

        return outputs, gen_loss, dis_loss, logs

    def forward(self, images, edges, masks):
        """
        Input:
            images: original images
            edges: images with only edge information
            masks: images with mask information
        Output:
            outputs: output of the generator
        Description:
            Masks only images. Then concat them and give it to generator as input
        """
        images_masked = (images * (1 - masks).float()) + masks
        inputs = torch.cat((images_masked, edges), dim=1)
        outputs = self.generator(inputs)                                    # in: [rgb(3) + edge(1)]
        return outputs

    def backward(self, gen_loss=None, dis_loss=None):
        """
        Input:
            none
        Output:
            none
        Description:
            step both optimizers
        """
        dis_loss.backward()
        self.dis_optimizer.step()

        gen_loss.backward()
        self.gen_optimizer.step()

class EdgeConnect():
    def __init__(self, train_data_loader = None, test_data_loader = None):
        self.train_loader = train_data_loader
        self.test_data_loader = test_data_loader

        self.edge_model = EdgeModel().to(cfg.DEVICE)
        self.inpaint_model = InpaintingModel().to(cfg.DEVICE)

        self.iteration = 0

    def single_test(self, test_image, mask):
        edgeDisc = torch.load(cfg.test_edge_disc_path, map_location=lambda storage, loc: storage)
        edgeGen = torch.load(cfg.test_edge_gen_path, map_location=lambda storage, loc: storage)
        inpaintDisc = torch.load(cfg.test_inpaint_disc_path, map_location=lambda storage, loc: storage)
        inpaintGen = torch.load(cfg.test_inpaint_gen_path, map_location=lambda storage, loc: storage)
        print("Models are loaded")

        self.iteration = edgeGen['iteration']
        self.edge_model.generator.load_state_dict(edgeGen['generator'])
        self.edge_model.discriminator.load_state_dict(edgeDisc['discriminator'])
        self.inpaint_model.generator.load_state_dict(inpaintGen['generator'])
        self.inpaint_model.discriminator.load_state_dict(inpaintDisc['discriminator'])
        print("Weights are updated")

        # image_gray = rgb2gray(test_image)
        image_gray = cv2.cvtColor(test_image, cv2.COLOR_RGB2GRAY)
        edge = canny(image_gray, sigma=cfg.SIGMA, mask=mask)

        test_image = torch.FloatTensor(test_image) / 255
        mask = torch.FloatTensor(mask) / 255
        image_gray = torch.FloatTensor(image_gray)
        edge = torch.FloatTensor(edge)

        test_image = test_image.permute(2,0,1)
        test_image = test_image.unsqueeze(0)
        mask = mask.unsqueeze(0)
        edge = edge.unsqueeze(0)
        image_gray = image_gray.unsqueeze(0)
        mask = mask.unsqueeze(0)
        edge = edge.unsqueeze(0)
        image_gray = image_gray.unsqueeze(0)

        print(f"Mask shape: {mask.shape}")
        print(f"Edge shape: {edge.shape}")
        print(f"Gray_data shape: {image_gray.shape}")
        print(f"Test Image shape: {test_image.shape}")

        print("Inputs are ready!")
        e_outputs, e_gen_loss, e_dis_loss, e_logs = self.edge_model.step(image_gray, edge, mask)
        print("Edge model completed!")
        e_outputs = e_outputs * mask + edge * (1 - mask)
        print("Inpaint inputs are ready!")
        i_outputs, i_gen_loss, i_dis_loss, i_logs = self.inpaint_model.step(test_image, e_outputs, mask)
        i_outputs = i_outputs
        outputs_merged = (i_outputs * mask) + (test_image * (1 - mask))
        print("Inpaint is completed!")
        output_image = outputs_merged.squeeze().permute(1,2,0)
        return output_image.detach().numpy()

    def train(self):
        if cfg.loadModel:
            self.load()
        for i in range(self.iteration, cfg.epoch_num):
            self.iteration += 1
            for images, masked_images, images_gray, masks, edges in self.train_loader:
                e_outputs, e_gen_loss, e_dis_loss, e_logs = self.edge_model.step(images_gray, edges, masks)
                e_outputs = e_outputs * masks + edges * (1 - masks)
                i_outputs, i_gen_loss, i_dis_loss, i_logs = self.inpaint_model.step(images, e_outputs, masks)
                outputs_merged = (i_outputs * masks) + (images * (1 - masks))
                print(i_dis_loss)
                self.inpaint_model.backward(i_gen_loss, i_dis_loss)
                self.edge_model.backward(e_gen_loss, e_dis_loss)
            break

            self.save()

    def save(self):
        torch.save({
            'iteration': self.iteration,
            'generator': self.edge_model.generator.state_dict()
        }, cfg.edge_gen_path)

        torch.save({
            'discriminator': self.edge_model.discriminator.state_dict()
        }, cfg.edge_disc_path)

        torch.save({
            'iteration': self.iteration,
            'generator': self.inpaint_model.generator.state_dict()
        }, cfg.inpaint_gen_path)

        torch.save({
            'discriminator': self.inpaint_model.discriminator.state_dict()
        }, cfg.inpaint_disc_path)

    def load(self):
        edgeDisc = torch.load(cfg.edge_disc_path, map_location=lambda storage, loc: storage)
        edgeGen = torch.load(cfg.edge_gen_path, map_location=lambda storage, loc: storage)
        inpaintDisc = torch.load(cfg.inpaint_disc_path, map_location=lambda storage, loc: storage)
        inpaintGen = torch.load(cfg.inpaint_gen_path, map_location=lambda storage, loc: storage)

        self.iteration = edgeGen['iteration']
        self.edge_model.generator.load_state_dict(edgeGen['generator'])
        self.edge_model.discriminator.load_state_dict(edgeDisc['discriminator'])
        self.inpaint_model.generator.load_state_dict(inpaintGen['generator'])
        self.inpaint_model.discriminator.load_state_dict(inpaintDisc['discriminator'])

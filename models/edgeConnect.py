import torch
import torch.nn as nn
import torch.optim as optim
from networks.EdgeConnectNetworks import InpaintGenerator, EdgeGenerator, Discriminator
from loss import AdversarialLoss, PerceptualLoss, StyleLoss
from config import Config

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
        Class of Edge connect model. All the necessary adjustment are don in this class
    """
    def __init__(self):
        super(EdgeConnect, self).__init__()
        self.generator = EdgeGenerator()
        self.discriminator = Discriminator()

        l1_loss = nn.L1Loss()
        perceptual_loss = PerceptualLoss()
        style_loss = StyleLoss()
        adversarial_loss = AdversarialLoss(type=cfg.GAN_LOSS)

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
        gen_fm_loss = gen_fm_loss * self.config.FM_LOSS_WEIGHT
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

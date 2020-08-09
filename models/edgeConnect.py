import torch
import torch.nn as nn
import torch.optim as optim
from networks.EdgeConnectNetworks import InpaintGenerator, EdgeGenerator, Discriminator
from config import Config

cfg = Config()

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
    def save(self):
        pass
    def load(self):
        pass

class EdgeConnect(BaseModel):
    def __init__(self):
        super(EdgeConnect, self).__init__()
        self.generator = EdgeGenerator()
        self.discriminator = Discriminator()

        l1_loss = nn.L1Loss()

        self.gen_optimizer = optim.Adam(params=self.generator.parameters(),
            lr=float(cfg.LR), betas=(cfg.BETA1, cfg.BETA2))

        self.dis_optimizer = optim.Adam(params=self.discriminator.parameters(),
            lr=float(cfg.LR) * float(cfg.D2G_LR), betas=(cfg.BETA1, cfg.BETA2))

    def run(self):
        pass

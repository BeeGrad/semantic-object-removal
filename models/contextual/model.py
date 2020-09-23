import torch
import torch.nn as nn
from models.contextual.network import GlobalDis, DisConvModule, Generator, ContextualAttention
from scripts.config import Config

cfg = Config()

class GenerativeContextual(nn.Module):
    def __init__(self, train_dataloader, test_dataloader):
        super(GenerativeContextual, self).__init__()
        self.GlobalDis = GlobalDis().to(cfg.DEVICE)
        self.DisConvModule = DisConvModule().to(cfg.DEVICE)
        self.Generator = Generator().to(cfg.DEVICE)

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

    def run(self):
        for images, masked_images, images_gray, masks, edges in self.train_dataloader:
            self.GlobalDis(images)
            self.DisConvModule(images)
            self.Generator(images, masks)
            break

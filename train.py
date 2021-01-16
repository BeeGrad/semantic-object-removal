# Training

from scripts.dataOperations import DataRead
from scripts.config import Config
from models.edgeconnect.model import EdgeConnect
from models.contextual.model import GenerativeContextual
from models.beemodels.fpnModel import fpnGan
from models.beemodels.vanillaGAN import VanillaGAN

cfg = Config()
data = DataRead()

if cfg.model == "EdgeConnect":
    edgeConnectModel = EdgeConnect()
    edgeConnectModel.run(data)

elif cfg.model == "Contextual":
    contextualModel = GenerativeContextual()
    contextualModel.run(data)

elif cfg.model == "FPNGan":
    fpnGanModel = fpnGan()
    fpnGanModel.run(data)

elif cfg.model == "VanillaGAN":
    vanillaGanModel = VanillaGAN()
    vanillaGanModel.run(data)

print(f"Training for {cfg.model} model is completed!")

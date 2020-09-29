# Training

from scripts.dataOperations import DataRead
from scripts.config import Config
from models.edgeconnect.model import EdgeConnect
from models.contextual.model import GenerativeContextual


cfg = Config()
data = DataRead(cfg.dataset, cfg.masking_type, cfg.batch_size)

if cfg.model == "EdgeConnect":
    edgeConnectModel = EdgeConnect()
    edgeConnectModel.train(data)

elif cfg.model == "Contextual":
    contextualModel = GenerativeContextual()
    contextualModel.run(data)

print(f"Training for {cfg.model} model is completed!")

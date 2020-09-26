# Training

from scripts.dataOperations import DataRead
from scripts.config import Config
from models.edgeconnect.model import EdgeConnect
from models.contextual.model import GenerativeContextual


cfg = Config()
data = DataRead(cfg.dataset, cfg.masking_type, cfg.batch_size)

if cfg.show_sample_data:
    data.show_sample_data()

if cfg.show_masked_data:
    data.show_masked_and_original()

if cfg.model == "EdgeConnect":
    edgeConnectModel = EdgeConnect()
    edgeConnectModel.train(data)

elif cfg.model == "Contextual":
    contextualModel = GenerativeContextual(data.train_data_loader, data.test_data_loader)
    contextualModel.run()

print(f"Training for {cfg.model} model is completed!")

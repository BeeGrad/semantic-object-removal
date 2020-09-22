# Training

from scripts.dataOperations import DataRead
from scripts.config import Config
from models.edgeconnect.model import EdgeConnect
from models.contextual.network import GlobalDis, DisConvModule, Generator, ContextualAttention


cfg = Config()
data = DataRead(cfg.dataset, cfg.masking_type, cfg.batch_size)
data.create_data_loaders()

if cfg.show_sample_data:
    data.show_sample_data()

elif cfg.show_masked_data:
    data.show_masked_and_original()

if cfg.model == "EdgeConnect":
    edgeConnectModel = EdgeConnect(data.train_data_loader, data.test_data_loader)
    edgeConnectModel.train()

elif cfg.model == "Contextual":
    contextualModel = DisConvModule()
    s = GlobalDis()
    g = Generator()
    c = ContextualAttention()

print(f"Training for {cfg.model} model is completed!")

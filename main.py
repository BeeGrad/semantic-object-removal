from dataOperations import DataRead
from config import Config
from models.edgeConnect import EdgeConnect

cfg = Config()
data = DataRead(cfg.dataset, cfg.masking_type, cfg.batch_size)
data.create_data_loaders()

if(cfg.show_sample_data):
    data.show_sample_data()

if(cfg.show_masked_data):
    data.show_masked_and_original()

if(cfg.model == "EdgeConnect"):
    model = EdgeConnect()

print(f"Training for {cfg.model} model is completed!")

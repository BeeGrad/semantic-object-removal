import torch
import torch.nn as nn
from scripts.config import Config
from models.edgeconnect.model import EdgeModel, InpaintingModel
from models.contextual.network import GlobalDis, LocalDis, Generator

cfg = Config()


class EdgeContextUnifiedModel():
    def __init__(self):
        super(EdgeContextUnifiedModel, self).__init__()
        self.edge_model = EdgeModel().to(cfg.DEVICE)
        self.inpaint_model = InpaintingModel().to(cfg.DEVICE)

        self.GlobalDis = GlobalDis().to(cfg.DEVICE)
        self.LocalDis = LocalDis().to(cfg.DEVICE)
        self.Generator = Generator().to(cfg.DEVICE)

    def single_test(self, test_image, mask, image_gray, edge):
        edgeDisc = torch.load(cfg.test_edge_disc_path, map_location=lambda storage, loc: storage)
        edgeGen = torch.load(cfg.test_edge_gen_path, map_location=lambda storage, loc: storage)
        inpaintDisc = torch.load(cfg.test_inpaint_disc_path, map_location=lambda storage, loc: storage)
        inpaintGen = torch.load(cfg.test_inpaint_gen_path, map_location=lambda storage, loc: storage)
        print("Edge Models are loaded")


        self.iteration = edgeGen['iteration']
        self.edge_model.generator.load_state_dict(edgeGen['generator'])
        self.edge_model.discriminator.load_state_dict(edgeDisc['discriminator'])
        self.inpaint_model.generator.load_state_dict(inpaintGen['generator'])
        self.inpaint_model.discriminator.load_state_dict(inpaintDisc['discriminator'])
        print("Edge Weights are updated")

        test_image = torch.FloatTensor(test_image) / 255
        mask = torch.FloatTensor(mask) / 255
        image_gray = torch.FloatTensor(image_gray) / 255
        edge = torch.FloatTensor(edge) / 255

        test_image = test_image.permute(2,0,1)
        test_image = test_image.unsqueeze(0)
        mask = mask.unsqueeze(0)
        edge = edge.unsqueeze(0)
        image_gray = image_gray.unsqueeze(0)
        mask = mask.unsqueeze(0)
        edge = edge.unsqueeze(0)
        image_gray = image_gray.unsqueeze(0)

        e_outputs, e_gen_loss, e_dis_loss, e_logs = self.edge_model.step(image_gray, edge, mask)
        e_outputs = e_outputs * mask + edge * (1 - mask)
        i_outputs, i_gen_loss, i_dis_loss, i_logs = self.inpaint_model.step(test_image, e_outputs, mask)
        i_outputs = i_outputs
        outputs_merged = (i_outputs * mask) + (test_image * (1 - mask))

        output_image = outputs_merged.squeeze().permute(1,2,0)

        # Contextual Part
        gen = torch.load(cfg.test_context_gen_path, map_location=lambda storage, loc: storage)
        discs = torch.load(cfg.test_context_discs_path, map_location=lambda storage, loc: storage)
        print("Contextual Models are loaded")

        self.Generator.load_state_dict(gen['generator'])
        self.LocalDis.load_state_dict(discs['localDiscriminator'])
        self.GlobalDis.load_state_dict(discs['globalDiscriminator'])
        print("Contextual Weights are loaded")

        img = output_image.permute(2,0,1)
        img = img.unsqueeze(0)

        x2, offset_flow = self.Generator.fine_generator(test_image, img, mask)
        x2_inpaint = x2 * mask + img * (1. - mask)

        full_output = x2_inpaint.detach().permute(0,2,3,1).cpu().numpy().squeeze()
        edge_connect_output = output_image.detach().numpy()

        return full_output, edge_connect_output

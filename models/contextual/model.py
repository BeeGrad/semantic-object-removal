import torch
import torch.nn as nn
from torch import autograd
from models.contextual.network import GlobalDis, LocalDis, Generator
from scripts.config import Config
from utils.utils import random_bbox, local_patch, spatial_discounting_mask, show_sample_input_data_context, \
                        calculate_psnr

cfg = Config()

class GenerativeContextual(nn.Module):
    def __init__(self):
        super(GenerativeContextual, self).__init__()
        self.GlobalDis = GlobalDis().to(cfg.DEVICE)
        self.LocalDis = LocalDis().to(cfg.DEVICE)
        self.Generator = Generator().to(cfg.DEVICE)

        self.optimizer_g = torch.optim.Adam(self.Generator.parameters(), lr=cfg.context_LR,
                                            betas=(cfg.context_BETA1, cfg.context_BETA2))
        d_params = list(self.LocalDis.parameters()) + list(self.GlobalDis.parameters())
        self.optimizer_d = torch.optim.Adam(d_params, lr=cfg.context_LR,
                                            betas=(cfg.context_BETA1, cfg.context_BETA2))

        self.iteration = 0

    def single_test(self, img, mask):
        gen = torch.load(cfg.test_context_gen_path, map_location=lambda storage, loc: storage)
        discs = torch.load(cfg.test_context_discs_path, map_location=lambda storage, loc: storage)

        print('#############################')
        # Our trained Load
        self.Generator.load_state_dict(gen['generator'])
        self.LocalDis.load_state_dict(discs['localDiscriminator'])
        self.GlobalDis.load_state_dict(discs['globalDiscriminator'])

        # Pre-trained Load
        # self.LocalDis.load_state_dict(discs['localD'])
        # self.GlobalDis.load_state_dict(discs['globalD'])
        # self.Generator.load_state_dict(gen)

        img = torch.FloatTensor(img) / 255
        mask = torch.FloatTensor(mask) / 255
        img = img.permute(2,0,1)
        img = img.unsqueeze(0)
        mask = mask.unsqueeze(0)
        mask = mask.unsqueeze(0)

        print(mask.shape)
        print(img.shape)

        img = img.to(cfg.DEVICE)
        mask = mask.to(cfg.DEVICE)

        x1, x2, offset_flow = self.Generator(img, mask)

        x1_inpaint = x1 * mask + img * (1. - mask)
        x2_inpaint = x2 * mask + img * (1. - mask)

        return x2_inpaint.detach().permute(0,2,3,1).cpu().numpy().squeeze()

    def run(self, data):
        """
        Input:
            data
        Output:
            none
        Description:
            Trains network
        """
        self.train()
        l1_loss = nn.L1Loss()
        losses = {}
        psnr_values = []
        self.epoch = 0

        bboxes = random_bbox()
        data.context_create_data_loaders()

        for epoch in range(self.epoch, cfg.epoch_num):
            for i, images in enumerate(data.train_loader):
                if i>20000:
                    break
                images, masks, masked_images = data.return_inputs_contextual(images[0], bboxes)

                if cfg.show_sample_data:
                    show_sample_input_data_context(images, masked_images, masks)
                    cfg.show_sample_data = False

                compute_loss_g = self.iteration % cfg.n_critic == 0

                masked_images = masked_images.to(cfg.DEVICE)
                masks = masks.to(cfg.DEVICE)
                images = images.to(cfg.DEVICE)

                x1, x2, offset_flow = self.Generator(masked_images, masks)
                local_patch_gt = local_patch(images, bboxes)
                x1_inpaint = x1 * masks + masked_images * (1. - masks)
                x2_inpaint = x2 * masks + masked_images * (1. - masks)
                local_patch_x1_inpaint = local_patch(x1_inpaint, bboxes)
                local_patch_x2_inpaint = local_patch(x2_inpaint, bboxes)

                # D part
                # wgan d loss
                local_patch_real_pred, local_patch_fake_pred = self.dis_forward(self.LocalDis, local_patch_gt, local_patch_x2_inpaint.detach())
                global_real_pred, global_fake_pred = self.dis_forward(self.GlobalDis, images, x2_inpaint.detach())
                losses['wgan_d'] = torch.mean(local_patch_fake_pred - local_patch_real_pred) + \
                    torch.mean(global_fake_pred - global_real_pred) * cfg.context_global_wgan_loss_alpha

                # gradients penalty loss
                local_penalty = self.calc_gradient_penalty(self.LocalDis, local_patch_gt, local_patch_x2_inpaint.detach())
                global_penalty = self.calc_gradient_penalty(self.GlobalDis, images, x2_inpaint.detach())
                losses['wgan_gp'] = local_penalty + global_penalty

                # G Part
                if compute_loss_g:
                    sd_mask = spatial_discounting_mask()
                    losses['l1'] = l1_loss(local_patch_x1_inpaint * sd_mask, local_patch_gt * sd_mask) * \
                        cfg.spatial_discounting_mask + l1_loss(local_patch_x2_inpaint * sd_mask, local_patch_gt * sd_mask)
                    losses['ae'] = l1_loss(x1 * (1. - masks), images * (1. - masks)) * \
                        cfg.spatial_discounting_mask + l1_loss(x2 * (1. - masks), images * (1. - masks))

                    # wgan g loss
                    local_patch_real_pred, local_patch_fake_pred = self.dis_forward(
                        self.LocalDis, local_patch_gt, local_patch_x2_inpaint)
                    global_real_pred, global_fake_pred = self.dis_forward(
                        self.GlobalDis, images, x2_inpaint)
                    losses['wgan_g'] = -(torch.mean(local_patch_fake_pred)) - \
                        torch.mean(global_fake_pred) * cfg.context_global_wgan_loss_alpha

                inpainted_result = x2_inpaint

                for k in losses.keys():
                    if not losses[k].dim() == 0:
                        losses[k] = torch.mean(losses[k])

                ###### Backward pass ###### Inplace hatasi veriyordu G ve D yerini degistirdim!!!
                # Update G
                if compute_loss_g:
                    self.optimizer_g.zero_grad()
                    losses['g'] = losses['l1'] * cfg.context_l1_loss_alpha + \
                                    losses['ae'] * cfg.context_ae_loss_alpha + \
                                    losses['wgan_g'] * cfg.context_gan_loss_alpha
                    losses['g'].backward()
                    self.optimizer_g.step()

                # Update D
                self.optimizer_d.zero_grad()
                losses['d'] = losses['wgan_d'] + losses['wgan_gp'] * cfg.context_wgan_gp_lambda
                losses['d'].backward()
                self.optimizer_d.step()

                self.iteration += 1
                psnr = calculate_psnr(images.squeeze().cpu().detach().numpy(), inpainted_result.squeeze().cpu().detach().numpy())
                psnr_values.append(psnr)
                if i % 10000 == 0:
                    print(f"Epoch: {epoch}, Iteration: {i}/{20000}")

            print(f"Epoch {self.iteration} is done!")
            print(f"PSNR Average for Epoch {self.iteration} is {sum(psnr_values)/len(psnr_values)}!")
            self.save()

    def save(self):
        """
        Input:
            none
        Output:
            none
        Description:
            Saves 3 pytroch model for localDis, globalDis and Gen
        """

        torch.save({
            'epoch': self.epoch,
            'generator': self.Generator.state_dict()
        }, cfg.context_generator_path)

        torch.save({
            'localDiscriminator': self.LocalDis.state_dict(),
            'globalDiscriminator': self.GlobalDis.state_dict()
        }, cfg.context_discs_path)

    def dis_forward(self, netD, ground_truth, x_inpaint):
        """
        Input:

        Output:

        Description:

        """
        assert ground_truth.size() == x_inpaint.size()
        batch_size = ground_truth.size(0)
        batch_data = torch.cat([ground_truth, x_inpaint], dim=0)
        batch_output = netD(batch_data)
        real_pred, fake_pred = torch.split(batch_output, batch_size, dim=0)

        return real_pred, fake_pred

    def calc_gradient_penalty(self, netD, real_data, fake_data):
        """
        Input:

        Output:

        Description:

        """
        batch_size = real_data.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = alpha.expand_as(real_data)
        if cfg.use_cuda:
            alpha = alpha.cuda()

        interpolates = alpha * real_data + (1 - alpha) * fake_data
        interpolates = interpolates.requires_grad_().clone()

        disc_interpolates = netD(interpolates)
        grad_outputs = torch.ones(disc_interpolates.size())

        if cfg.use_cuda:
            grad_outputs = grad_outputs.cuda()

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=grad_outputs, create_graph=True,
                                  retain_graph=True, only_inputs=True)[0]

        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty

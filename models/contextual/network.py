import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm as spectral_norm_fn
from scripts.config import Config

cfg = Config()

class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding=0,
                 conv_padding=0, dilation=1, weight_norm='none', norm='none',
                 activation='relu', pad_type='zero', transpose=False):
        super(Conv2dBlock, self).__init__()
        # initialize padding
        self.pad = nn.ReflectionPad2d(padding)

        # initialize normalization
        norm_dim = output_dim
        self.norm = nn.BatchNorm2d(norm_dim)
        self.weight_norm = spectral_norm_fn

        # initialize activation
        if cfg.context_activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif cfg.context_activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif cfg.context_activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)

        # initialize convolution
        if cfg.context_conv_type == 'transpose':
            self.conv = nn.ConvTranspose2d(input_dim, output_dim,
                                           kernel_size, stride,
                                           padding=conv_padding,
                                           output_padding=conv_padding,
                                           dilation=dilation,
                                           bias=True)
        elif cfg.context_conv_type == 'normal':
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride,
                                  padding=conv_padding, dilation=dilation,
                                  bias=True)

        if self.weight_norm:
            self.conv = self.weight_norm(self.conv)

    def forward(self, x):
        x = self.conv(self.pad(x))
        x = self.norm(x)
        x = self.activation(x)

        return x

def gen_conv(input_dim = cfg.context_input_dim, output_dim = cfg.context_gen_feat_dim,
            kernel_size=3, stride=1, padding=0, rate=1, activation='elu'):
    return Conv2dBlock(input_dim, output_dim, kernel_size, stride,
                       conv_padding=padding, dilation=rate,
                       activation=activation)


def dis_conv(input_dim = cfg.context_input_dim, output_dim = cfg.context_gen_feat_dim,
            kernel_size=5, stride=2, padding=0, rate=1, activation='lrelu'):
    return Conv2dBlock(input_dim, output_dim, kernel_size, stride,
                       conv_padding=padding, dilation=rate,
                       activation=activation)

class CoarseGenerator(nn.Module):
    def __init__(self):
        super(CoarseGenerator, self).__init__()
        self.use_cuda = True
        self.device_ids = None

        self.conv1 = gen_conv(cfg.context_input_dim + 2, cfg.context_gen_feat_dim, 5, 1, 2)
        self.conv2_downsample = gen_conv(cfg.context_gen_feat_dim, cfg.context_gen_feat_dim*2, 3, 2, 1)
        self.conv3 = gen_conv(cfg.context_gen_feat_dim*2, cfg.context_gen_feat_dim*2, 3, 1, 1)
        self.conv4_downsample = gen_conv(cfg.context_gen_feat_dim*2, cfg.context_gen_feat_dim*4, 3, 2, 1)
        self.conv5 = gen_conv(cfg.context_gen_feat_dim*4, cfg.context_gen_feat_dim*4, 3, 1, 1)
        self.conv6 = gen_conv(cfg.context_gen_feat_dim*4, cfg.context_gen_feat_dim*4, 3, 1, 1)

        self.conv7_atrous = gen_conv(cfg.context_gen_feat_dim*4, cfg.context_gen_feat_dim*4, 3, 1, 2, rate=2)
        self.conv8_atrous = gen_conv(cfg.context_gen_feat_dim*4, cfg.context_gen_feat_dim*4, 3, 1, 4, rate=4)
        self.conv9_atrous = gen_conv(cfg.context_gen_feat_dim*4, cfg.context_gen_feat_dim*4, 3, 1, 8, rate=8)
        self.conv10_atrous = gen_conv(cfg.context_gen_feat_dim*4, cfg.context_gen_feat_dim*4, 3, 1, 16, rate=16)

        self.conv11 = gen_conv(cfg.context_gen_feat_dim*4, cfg.context_gen_feat_dim*4, 3, 1, 1)
        self.conv12 = gen_conv(cfg.context_gen_feat_dim*4, cfg.context_gen_feat_dim*4, 3, 1, 1)

        self.conv13 = gen_conv(cfg.context_gen_feat_dim*4, cfg.context_gen_feat_dim*2, 3, 1, 1)
        self.conv14 = gen_conv(cfg.context_gen_feat_dim*2, cfg.context_gen_feat_dim*2, 3, 1, 1)
        self.conv15 = gen_conv(cfg.context_gen_feat_dim*2, cfg.context_gen_feat_dim, 3, 1, 1)
        self.conv16 = gen_conv(cfg.context_gen_feat_dim, cfg.context_gen_feat_dim//2, 3, 1, 1)
        self.conv17 = gen_conv(cfg.context_gen_feat_dim//2, cfg.context_input_dim, 3, 1, 1, activation='none')

    def forward(self, x, mask):
        # For indicating the boundaries of images
        ones = torch.ones(x.size(0), 1, x.size(2), x.size(3))
        if self.use_cuda:
            ones = ones.cuda()
            mask = mask.cuda()
        # 5 x 256 x 256
        x = self.conv1(torch.cat([x, ones, mask], dim=1))
        x = self.conv2_downsample(x)
        # cfg.context_gen_feat_dim*2 x 128 x 128
        x = self.conv3(x)
        x = self.conv4_downsample(x)
        # cfg.context_gen_feat_dim*4 x 64 x 64
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7_atrous(x)
        x = self.conv8_atrous(x)
        x = self.conv9_atrous(x)
        x = self.conv10_atrous(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        # cfg.context_gen_feat_dim*2 x 128 x 128
        x = self.conv13(x)
        x = self.conv14(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        # cfg.context_gen_feat_dim x 256 x 256
        x = self.conv15(x)
        x = self.conv16(x)
        x = self.conv17(x)
        # 3 x 256 x 256
        x_stage1 = torch.clamp(x, -1., 1.)

        return x_stage1

class FineGenerator(nn.Module):
    def __init__(self):
        super(FineGenerator, self).__init__()
        self.use_cuda = True
        self.device_ids = None

        # 3 x 256 x 256
        self.conv1 = gen_conv(cfg.context_input_dim + 2, cfg.context_gen_feat_dim, 5, 1, 2)
        self.conv2_downsample = gen_conv(cfg.context_gen_feat_dim, cfg.context_gen_feat_dim, 3, 2, 1)
        # cfg.context_gen_feat_dim*2 x 128 x 128
        self.conv3 = gen_conv(cfg.context_gen_feat_dim, cfg.context_gen_feat_dim*2, 3, 1, 1)
        self.conv4_downsample = gen_conv(cfg.context_gen_feat_dim*2, cfg.context_gen_feat_dim*2, 3, 2, 1)
        # cfg.context_gen_feat_dim*4 x 64 x 64
        self.conv5 = gen_conv(cfg.context_gen_feat_dim*2, cfg.context_gen_feat_dim*4, 3, 1, 1)
        self.conv6 = gen_conv(cfg.context_gen_feat_dim*4, cfg.context_gen_feat_dim*4, 3, 1, 1)

        self.conv7_atrous = gen_conv(cfg.context_gen_feat_dim*4, cfg.context_gen_feat_dim*4, 3, 1, 2, rate=2)
        self.conv8_atrous = gen_conv(cfg.context_gen_feat_dim*4, cfg.context_gen_feat_dim*4, 3, 1, 4, rate=4)
        self.conv9_atrous = gen_conv(cfg.context_gen_feat_dim*4, cfg.context_gen_feat_dim*4, 3, 1, 8, rate=8)
        self.conv10_atrous = gen_conv(cfg.context_gen_feat_dim*4, cfg.context_gen_feat_dim*4, 3, 1, 16, rate=16)

        # attention branch
        # 3 x 256 x 256
        self.pmconv1 = gen_conv(input_dim + 2, cfg.context_gen_feat_dim, 5, 1, 2)
        self.pmconv2_downsample = gen_conv(cfg.context_gen_feat_dim, cfg.context_gen_feat_dim, 3, 2, 1)
        # cfg.context_gen_feat_dim*2 x 128 x 128
        self.pmconv3 = gen_conv(cfg.context_gen_feat_dim, cfg.context_gen_feat_dim*2, 3, 1, 1)
        self.pmconv4_downsample = gen_conv(cfg.context_gen_feat_dim*2, cfg.context_gen_feat_dim*4, 3, 2, 1)
        # cfg.context_gen_feat_dim*4 x 64 x 64
        self.pmconv5 = gen_conv(cfg.context_gen_feat_dim*4, cfg.context_gen_feat_dim*4, 3, 1, 1)
        self.pmconv6 = gen_conv(cfg.context_gen_feat_dim*4, cfg.context_gen_feat_dim*4, 3, 1, 1, activation='relu')
        self.contextul_attention = ContextualAttention(ksize=3, stride=1, rate=2, fuse_k=3, softmax_scale=10,
                                                       fuse=True, use_cuda=self.use_cuda, device_ids=self.device_ids)
        self.pmconv9 = gen_conv(cfg.context_gen_feat_dim*4, cfg.context_gen_feat_dim*4, 3, 1, 1)
        self.pmconv10 = gen_conv(cfg.context_gen_feat_dim*4, cfg.context_gen_feat_dim*4, 3, 1, 1)
        self.allconv11 = gen_conv(cfg.context_gen_feat_dim*8, cfg.context_gen_feat_dim*4, 3, 1, 1)
        self.allconv12 = gen_conv(cfg.context_gen_feat_dim*4, cfg.context_gen_feat_dim*4, 3, 1, 1)
        self.allconv13 = gen_conv(cfg.context_gen_feat_dim*4, cfg.context_gen_feat_dim*2, 3, 1, 1)
        self.allconv14 = gen_conv(cfg.context_gen_feat_dim*2, cfg.context_gen_feat_dim*2, 3, 1, 1)
        self.allconv15 = gen_conv(cfg.context_gen_feat_dim*2, cfg.context_gen_feat_dim, 3, 1, 1)
        self.allconv16 = gen_conv(cfg.context_gen_feat_dim, cfg.context_gen_feat_dim//2, 3, 1, 1)
        self.allconv17 = gen_conv(cfg.context_gen_feat_dim//2, cfg.context_input_dim, 3, 1, 1, activation='none')

    def forward(self, xin, x_stage1, mask):
        x1_inpaint = x_stage1 * mask + xin * (1. - mask)
        # For indicating the boundaries of images
        ones = torch.ones(xin.size(0), 1, xin.size(2), xin.size(3))
        if self.use_cuda:
            ones = ones.cuda()
            mask = mask.cuda()
        # conv branch
        xnow = torch.cat([x1_inpaint, ones, mask], dim=1)
        x = self.conv1(xnow)
        x = self.conv2_downsample(x)
        x = self.conv3(x)
        x = self.conv4_downsample(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7_atrous(x)
        x = self.conv8_atrous(x)
        x = self.conv9_atrous(x)
        x = self.conv10_atrous(x)
        x_hallu = x
        # attention branch
        x = self.pmconv1(xnow)
        x = self.pmconv2_downsample(x)
        x = self.pmconv3(x)
        x = self.pmconv4_downsample(x)
        x = self.pmconv5(x)
        x = self.pmconv6(x)
        x, offset_flow = self.contextul_attention(x, x, mask)
        x = self.pmconv9(x)
        x = self.pmconv10(x)
        pm = x
        x = torch.cat([x_hallu, pm], dim=1)
        # merge two branches
        x = self.allconv11(x)
        x = self.allconv12(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.allconv13(x)
        x = self.allconv14(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.allconv15(x)
        x = self.allconv16(x)
        x = self.allconv17(x)
        x_stage2 = torch.clamp(x, -1., 1.)

        return x_stage2, offset_flow

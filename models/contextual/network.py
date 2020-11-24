import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as spectral_norm_fn
from scripts.config import Config
from utils.utils import extract_image_patches, flow_to_image, reduce_sum, reduce_mean, same_padding

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
            """
            Input:
                none
            Output:
                none
            Description:
                Creates a conv layer with desired input and output sizes with other features.
            """
            return Conv2dBlock(input_dim, output_dim, kernel_size, stride,
                        conv_padding=padding, dilation=rate,
                        activation=activation)


def dis_conv(input_dim = cfg.context_input_dim, output_dim = cfg.context_gen_feat_dim,
            kernel_size=5, stride=2, padding=0, rate=1, activation='lrelu'):
            return Conv2dBlock(input_dim, output_dim, kernel_size, stride,
                       conv_padding=padding, dilation=rate,
                       activation=activation)

class ContextualAttention(nn.Module):
    def __init__(self, ksize=3, stride=1, rate=1, fuse_k=3, softmax_scale=10,
                 fuse=False, use_cuda=False, device_ids=None):
        super(ContextualAttention, self).__init__()
        self.ksize = ksize
        self.stride = stride
        self.rate = rate
        self.fuse_k = fuse_k
        self.softmax_scale = softmax_scale
        self.fuse = fuse
        self.use_cuda = cfg.use_cuda
        self.device_ids = device_ids

    def forward(self, f, b, mask=None):
        """ Contextual attention layer implementation.
        Contextual attention is first introduced in publication:
            Generative Image Inpainting with Contextual Attention, Yu et al.
        Args:
            f: Input feature to match (foreground).
            b: Input feature for match (background).
            mask: Input mask for b, indicating patches not available.
            ksize: Kernel size for contextual attention.
            stride: Stride for extracting patches from b.
            rate: Dilation for matching.
            softmax_scale: Scaled softmax for attention.
        Returns:
            torch.tensor: output
        """
        # get shapes
        raw_int_fs = list(f.size())   # b*c*h*w
        raw_int_bs = list(b.size())   # b*c*h*w

        # extract patches from background with stride and rate
        kernel = 2 * self.rate
        # raw_w is extracted for reconstruction
        raw_w = extract_image_patches(b, ksizes=[kernel, kernel],
                                      strides=[self.rate*self.stride,
                                               self.rate*self.stride],
                                      rates=[1, 1],
                                      padding='same') # [N, C*k*k, L]
        # raw_shape: [N, C, k, k, L]
        raw_w = raw_w.view(raw_int_bs[0], raw_int_bs[1], kernel, kernel, -1)
        raw_w = raw_w.permute(0, 4, 1, 2, 3)    # raw_shape: [N, L, C, k, k]
        raw_w_groups = torch.split(raw_w, 1, dim=0)

        # downscaling foreground option: downscaling both foreground and
        # background for matching and use original background for reconstruction.
        f = F.interpolate(f, scale_factor=1./self.rate, mode='nearest')
        b = F.interpolate(b, scale_factor=1./self.rate, mode='nearest')
        int_fs = list(f.size())     # b*c*h*w
        int_bs = list(b.size())
        f_groups = torch.split(f, 1, dim=0)  # split tensors along the batch dimension
        # w shape: [N, C*k*k, L]
        w = extract_image_patches(b, ksizes=[self.ksize, self.ksize],
                                  strides=[self.stride, self.stride],
                                  rates=[1, 1],
                                  padding='same')
        # w shape: [N, C, k, k, L]
        w = w.view(int_bs[0], int_bs[1], self.ksize, self.ksize, -1)
        w = w.permute(0, 4, 1, 2, 3)    # w shape: [N, L, C, k, k]
        w_groups = torch.split(w, 1, dim=0)

        # process mask
        if mask is None:
            mask = torch.zeros([int_bs[0], 1, int_bs[2], int_bs[3]])
            if self.use_cuda:
                mask = mask.cuda()
        else:
            mask = F.interpolate(mask, scale_factor=1./(4*self.rate),  mode='nearest')
        int_ms = list(mask.size())
        # m shape: [N, C*k*k, L]
        m = extract_image_patches(mask, ksizes=[self.ksize, self.ksize],
                                  strides=[self.stride, self.stride],
                                  rates=[1, 1],
                                  padding='same')
        # m shape: [N, C, k, k, L]
        m = m.view(int_ms[0], int_ms[1], self.ksize, self.ksize, -1)
        m = m.permute(0, 4, 1, 2, 3)    # m shape: [N, L, C, k, k]
        m = m[0]    # m shape: [L, C, k, k]
        # mm shape: [L, 1, 1, 1]
        mm = (reduce_mean(m, axis=[1, 2, 3], keepdim=True)==0.).to(torch.float32)
        mm = mm.permute(1, 0, 2, 3) # mm shape: [1, L, 1, 1]

        y = []
        offsets = []
        k = self.fuse_k
        scale = self.softmax_scale    # to fit the PyTorch tensor image value range
        fuse_weight = torch.eye(k).view(1, 1, k, k)  # 1*1*k*k
        if self.use_cuda:
            fuse_weight = fuse_weight.cuda()

        for xi, wi, raw_wi in zip(f_groups, w_groups, raw_w_groups):
            '''
            O => output channel as a conv filter
            I => input channel as a conv filter
            xi : separated tensor along batch dimension of front; (B=1, C=128, H=32, W=32)
            wi : separated patch tensor along batch dimension of back; (B=1, O=32*32, I=128, KH=3, KW=3)
            raw_wi : separated tensor along batch dimension of back; (B=1, I=32*32, O=128, KH=4, KW=4)
            '''
            # conv for compare
            escape_NaN = torch.FloatTensor([1e-4])
            if self.use_cuda:
                escape_NaN = escape_NaN.cuda()
            wi = wi[0]  # [L, C, k, k]
            max_wi = torch.sqrt(reduce_sum(torch.pow(wi, 2) + escape_NaN, axis=[1, 2, 3], keepdim=True))
            wi_normed = wi / max_wi
            # xi shape: [1, C, H, W], yi shape: [1, L, H, W]
            xi = same_padding(xi, [self.ksize, self.ksize], [1, 1], [1, 1])  # xi: 1*c*H*W
            yi = F.conv2d(xi, wi_normed, stride=1)   # [1, L, H, W]
            # conv implementation for fuse scores to encourage large patches
            if self.fuse:
                # make all of depth to spatial resolution
                yi = yi.view(1, 1, int_bs[2]*int_bs[3], int_fs[2]*int_fs[3])  # (B=1, I=1, H=32*32, W=32*32)
                yi = same_padding(yi, [k, k], [1, 1], [1, 1])
                yi = F.conv2d(yi, fuse_weight, stride=1)  # (B=1, C=1, H=32*32, W=32*32)
                yi = yi.contiguous().view(1, int_bs[2], int_bs[3], int_fs[2], int_fs[3])  # (B=1, 32, 32, 32, 32)
                yi = yi.permute(0, 2, 1, 4, 3)
                yi = yi.contiguous().view(1, 1, int_bs[2]*int_bs[3], int_fs[2]*int_fs[3])
                yi = same_padding(yi, [k, k], [1, 1], [1, 1])
                yi = F.conv2d(yi, fuse_weight, stride=1)
                yi = yi.contiguous().view(1, int_bs[3], int_bs[2], int_fs[3], int_fs[2])
                yi = yi.permute(0, 2, 1, 4, 3).contiguous()
            yi = yi.view(1, int_bs[2] * int_bs[3], int_fs[2], int_fs[3])  # (B=1, C=32*32, H=32, W=32)
            # softmax to match
            yi = yi * mm
            yi = F.softmax(yi*scale, dim=1)
            yi = yi * mm  # [1, L, H, W]

            offset = torch.argmax(yi, dim=1, keepdim=True)  # 1*1*H*W

            if int_bs != int_fs:
                # Normalize the offset value to match foreground dimension
                times = float(int_fs[2] * int_fs[3]) / float(int_bs[2] * int_bs[3])
                offset = ((offset + 1).float() * times - 1).to(torch.int64)
            offset = torch.cat([offset//int_fs[3], offset%int_fs[3]], dim=1)  # 1*2*H*W

            # deconv for patch pasting
            wi_center = raw_wi[0]
            # yi = F.pad(yi, [0, 1, 0, 1])    # here may need conv_transpose same padding
            yi = F.conv_transpose2d(yi, wi_center, stride=self.rate, padding=1) / 4.  # (B=1, C=128, H=64, W=64)
            y.append(yi)
            offsets.append(offset)

        y = torch.cat(y, dim=0)  # back to the mini-batch
        y.contiguous().view(raw_int_fs)

        offsets = torch.cat(offsets, dim=0)
        offsets = offsets.view(int_fs[0], 2, *int_fs[2:])

        # case1: visualize optical flow: minus current position
        h_add = torch.arange(int_fs[2]).view([1, 1, int_fs[2], 1]).expand(int_fs[0], -1, -1, int_fs[3])
        w_add = torch.arange(int_fs[3]).view([1, 1, 1, int_fs[3]]).expand(int_fs[0], -1, int_fs[2], -1)
        ref_coordinate = torch.cat([h_add, w_add], dim=1)
        if self.use_cuda:
            ref_coordinate = ref_coordinate.cuda()

        offsets = offsets - ref_coordinate
        # flow = pt_flow_to_image(offsets)

        flow = torch.from_numpy(flow_to_image(offsets.permute(0, 2, 3, 1).cpu().data.numpy())) / 255.
        flow = flow.permute(0, 3, 1, 2)
        if self.use_cuda:
            flow = flow.cuda()
        # case2: visualize which pixels are attended
        # flow = torch.from_numpy(highlight_flow((offsets * mask.long()).cpu().data.numpy()))

        if self.rate != 1:
            flow = F.interpolate(flow, scale_factor=self.rate*4,  mode='nearest')

        return y, flow

class CoarseGenerator(nn.Module):
    def __init__(self):
        super(CoarseGenerator, self).__init__()
        self.use_cuda = cfg.use_cuda
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
        x = F.interpolate(x, scale_factor=2,  mode='nearest')
        # cfg.context_gen_feat_dim*2 x 128 x 128
        x = self.conv13(x)
        x = self.conv14(x)
        x = F.interpolate(x, scale_factor=2,  mode='nearest')
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
        self.use_cuda = cfg.use_cuda
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
        self.pmconv1 = gen_conv(cfg.context_input_dim + 2, cfg.context_gen_feat_dim, 5, 1, 2)
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
        x = F.interpolate(x, scale_factor=2,  mode='nearest')
        x = self.allconv13(x)
        x = self.allconv14(x)
        x = F.interpolate(x, scale_factor=2,  mode='nearest')
        x = self.allconv15(x)
        x = self.allconv16(x)
        x = self.allconv17(x)
        x_stage2 = torch.clamp(x, -1., 1.)

        return x_stage2, offset_flow

class DisConvModule(nn.Module):
    def __init__(self):
        super(DisConvModule, self).__init__()
        self.use_cuda = cfg.use_cuda
        self.device_ids = None

        self.conv1 = dis_conv(cfg.context_input_dim, cfg.context_dis_feat_dim, 5, 2, 2)
        self.conv2 = dis_conv(cfg.context_dis_feat_dim, cfg.context_dis_feat_dim*2, 5, 2, 2)
        self.conv3 = dis_conv(cfg.context_dis_feat_dim*2, cfg.context_dis_feat_dim*4, 5, 2, 2)
        self.conv4 = dis_conv(cfg.context_dis_feat_dim*4, cfg.context_dis_feat_dim*4, 5, 2, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        return x

class LocalDis(nn.Module):
    def __init__(self):
        super(LocalDis, self).__init__()
        self.use_cuda = cfg.use_cuda
        self.device_ids = None

        self.dis_conv_module = DisConvModule()
        self.linear = nn.Linear(cfg.context_dis_feat_dim*4*8*8, 1)

    def forward(self, x):
        x = self.dis_conv_module(x)
        x = x.view(x.size()[0], -1)
        x = self.linear(x)

        return x

class GlobalDis(nn.Module):
    def __init__(self):
        super(GlobalDis, self).__init__()
        self.use_cuda = cfg.use_cuda
        self.device_ids = None

        self.dis_conv_module = DisConvModule()
        self.linear = nn.Linear(cfg.context_dis_feat_dim*4*16*16, 1)

    def forward(self, x):
        x = self.dis_conv_module(x)
        x = x.view(x.size()[0], -1)
        x = self.linear(x)

        return x



class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.coarse_generator = CoarseGenerator()
        self.fine_generator = FineGenerator()

    def forward(self, x, mask):
        x_stage1 = self.coarse_generator(x, mask)
        x_stage2, offset_flow = self.fine_generator(x, x_stage1, mask)
        return x_stage1, x_stage2, offset_flow

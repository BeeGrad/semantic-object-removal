import argparse
import os
import time

class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--seed', type=int, default=1, help='random seed')
        self.parser.add_argument('--model', type=str, default='gmcnn')
        self.parser.add_argument('--random_mask', type=int, default=0, help='using random mask')
        self.parser.add_argument('--img_shapes', type=str, default='256,256,3', help='given shape parameters: h,w,c or h,w')
        self.parser.add_argument('--mask_shapes', type=str, default='128,128', help='given mask parameters: h,w')
        self.parser.add_argument('--mask_type', type=str, default='rect')
        self.parser.add_argument('--phase', type=str, default='test')

        # for generator
        self.parser.add_argument('--g_cnum', type=int, default=32, help='# of generator filters in first conv layer')
        self.parser.add_argument('--d_cnum', type=int, default=32, help='# of discriminator filters in first conv layer')

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        assert self.opt.random_mask in [0, 1]
        self.opt.random_mask = True if self.opt.random_mask == 1 else False

        assert self.opt.mask_type in ['rect', 'stroke']

        str_img_shapes = self.opt.img_shapes.split(',')
        self.opt.img_shapes = [int(x) for x in str_img_shapes]

        str_mask_shapes = self.opt.mask_shapes.split(',')
        self.opt.mask_shapes = [int(x) for x in str_mask_shapes]

        return self.opt
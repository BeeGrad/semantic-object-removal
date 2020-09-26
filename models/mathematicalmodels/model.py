import cv2
import numpy as np
from scripts.config import Config

cfg = Config()

class InpaintMathematical():
    def __init__(self, source_image, mask, radius):
        super(InpaintMathematical, self).__init__()
        self.input = source_image
        self.method = cfg.mathematical_method
        self.mask = mask
        self.radius = radius

    def run(self):
        if self.method == 'navier-strokes':
            self.output = cv2.inpaint(
                                    self.input,
                                    self.mask,
                                    cfg.freely_select_mask_size,
                                    cv2.INPAINT_NS
                                        )

        if self.method == 'fast-marching':
            self.output = cv2.inpaint(
                                    self.input,
                                    self.mask,
                                    self.radius,
                                    cv2.INPAINT_TELEA
                                        )

        return self.output

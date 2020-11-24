import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
import numpy as np
import tensorflow as tf
from utils.gmcnn_utils import generate_rect_mask, generate_stroke_mask
from scripts.gmcnn_options import Options
from scripts.config import Config
from models.gmcnn.network import GMCNNModel

class GenerativeCNN():
    def __init__(self):
        self.options = Options().parse()
        self.path_dataset = Config().gmcnn_model_path
        self.model = GMCNNModel()
    
    def single_test(self, image, mask_1d):
        """
        Input:
            image: Input image that will be used with network
            mask: Mask that is used to mask the original image
        Output:
            output_image: Result of the network
        Description:
            Use pretrained version of the gmcnn model to eval for only 1 image
        """
        with tf.Session(config=tf.ConfigProto()) as sess:
            input_image_tf = tf.placeholder(dtype=tf.float32, shape=[1, 256, 256, 3])
            input_mask_tf = tf.placeholder(dtype=tf.float32, shape=[1, 256, 256, 1])

            output = self.model.evaluate(input_image_tf, input_mask_tf, config=self.options, reuse=False)
            output = (output + 1) * 127.5
            output = tf.minimum(tf.maximum(output[:, :, :, ::-1], 0), 255)
            output = tf.cast(output, tf.uint8)

            # Load Pretrained Model
            vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            assign_ops = list(map(lambda x: tf.assign(x, tf.contrib.framework.load_variable(self.path_dataset, x.name)), vars_list))
            sess.run(assign_ops)

            # Transform Mask
            mask = np.zeros((mask_1d.shape[0], mask_1d.shape[1], 1))
            mask[:, :, 0] = mask_1d
            mask = mask.astype(np.float32)
            mask = mask / 255.0

            # Apply Mask
            image = image * (1 - mask) + 255 * mask

            # Get Ready
            image = np.expand_dims(image, axis=0)
            mask = np.expand_dims(mask, axis=0)

            # Output Image
            result = sess.run(output, feed_dict={input_image_tf: image, input_mask_tf: mask})
            output_image = result[0][:, :, ::-1]

        return output_image

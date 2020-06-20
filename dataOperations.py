import pickle
import torch
import numpy as np
from matplotlib import pyplot as plt
import random
import cv2
import os, os.path


class dataRead():
    def __init__(self, dataset="cifar10", masking_type="lines"):
        self.dataset = dataset
        self.masking_type = masking_type

    def get_data(self):
        """
        Input:
            none
        Output:
            none
        Description:
            Save the desired dataset as numpy array
        Dataset Properties:
            Cifar10 dataset's loaded dict keys:
                b'batch_label' - Name of the batch
                b'labels' - Class type of every image
                b'data' - Image
                b'filenames' - Name of the image files

            creates a self.data that containts the dataset
        """
        # Cifar10 dataset load part
        if self.dataset == 'cifar10':
            self.path = "../../datasets/cifar10"

            self.train_files = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4",
                            "data_batch_5"]

            data = np.empty((50000,32,32,3), int)
            file_num = 0

            for file in self.train_files:
                file_path = self.path + "/" + file

                with open(file_path, 'rb') as f:
                    dict = pickle.load(f, encoding='bytes')
                    batch_data = dict[b'data']

                    for img in range(batch_data.shape[0]):
                        im1 = batch_data[img,:1024]
                        im1 = np.reshape(im1,(32,32))
                        im2 = batch_data[img,1024:2048]
                        im2 = np.reshape(im2,(32,32))
                        im3 = batch_data[img,2048:3072]
                        im3 = np.reshape(im3,(32,32))

                        rgb = (im1[..., np.newaxis], im2[..., np.newaxis], im3[..., np.newaxis])
                        data[img + file_num*10000] = np.concatenate(rgb, axis=-1)
                    file_num += 1


            print(f"Data Array has been created from cifar10 dataset with shape: {data.shape}")
            self.data = data

        elif self.dataset == 'places2':
            self.path = "../../datasets/places2"
            imgs = []
            ctr = 0

            for filename in os.listdir(self.path):
                img = cv2.imread(os.path.join(self.path,filename))
                if img is not None:
                    imgs.append(img)
                    # print(f"Image {filename} is loaded")

                if ctr == 1000:
                    break
                ctr += 1

            self.data = np.array(imgs)
            print(f"Data Array has been created from places2 dataset with shape: {self.data.shape}")

    def show_sample_data(self, sample_type="batch"):
        """
        Input:
            sample_type: 'single' or 'batch'
                Determines how to show sample image or images
        Output:
            none
        Description:
            Shows some of the data as example
        """
        if sample_type == "single":
            sample = random.randint(0,self.data.shape[0]-1)
            im = self.data[sample]
            plt.imshow(im)
            plt.show()

        if sample_type == 'batch':
            w=32
            h=32
            fig=plt.figure(figsize=(8, 8))
            columns = 8
            rows = 8
            for i in range(1, columns*rows +1):
                sample = random.randint(0,self.data.shape[0]-1)
                img = self.data[sample]
                fig.add_subplot(rows, columns, i)
                plt.imshow(img)
            plt.show()

    def show_masked_and_original(self):
        """
        Input:
            none
        Output:
            none
        Description:
            Shows the orignal and masked image of same data
        """
        sample = random.randint(0,self.data.shape[0]-1)
        im = self.data[sample]
        masked_im = self.masked_data[sample]
        fig=plt.figure(figsize=(1, 2))
        fig.add_subplot(1, 2, 1)
        plt.imshow(im)
        fig.add_subplot(1, 2, 2)
        plt.imshow(masked_im)
        plt.show()

    def create_mask(self):
        """
        Input:
            none
        Output:
            none
        Description:
            Creates masks in the desired method and apply them to data.
            Creates self.masked_data.
        """
        self.masked_data = np.empty_like(self.data)

        ## Prepare masking matrix
        image_width = self.data.shape[1]
        image_heigth = self.data.shape[2]
        image_channel = self.data.shape[3]

        if self.masking_type == 'lines':
            for img in range(self.data.shape[0]):
                mask = np.full((image_width,image_heigth, image_channel), 255, np.uint8) ## White background

                for _ in range(np.random.randint(1, image_width // 4)):
                    # Get random x locations to start line
                    x1, x2 = np.random.randint(1, image_width), np.random.randint(1, image_width)
                    # Get random y locations to start line
                    y1, y2 = np.random.randint(1, image_heigth), np.random.randint(1, image_heigth)
                    # Get random thickness of the line drawn
                    thickness = np.random.randint(1, 3)
                    # Draw black line on the white mask
                    cv2.line(mask,(x1,y1),(x2,y2),(0,0,0),thickness)

                ## Mask the image
                masked_image = self.data[img].copy()
                masked_image[mask==0] = 255
                self.masked_data[img] = masked_image

if __name__ == "__main__":
    data_class = dataRead(dataset='places2')
    data_class.get_data()
    data_class.show_sample_data()
    data_class.create_mask()
    data_class.show_masked_and_original()

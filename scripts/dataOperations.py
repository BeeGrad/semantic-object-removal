import pickle
import torch
import numpy as np
import torchvision
from matplotlib import pyplot as plt
import random
import cv2
import os, os.path
from skimage.feature import canny
from skimage.color import rgb2gray
from scripts.config import Config

cfg = Config()

class DataRead():
    def __init__(self, dataset="cifar10", masking_type="lines", batch_size = 10):
        self.dataset = dataset
        self.masking_type = masking_type
        self.batch_size = batch_size

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
        if self.dataset == "cifar10":
            self.path = "../datasets/cifar10"

            self.train_files = [
                "data_batch_1",
                "data_batch_2",
                "data_batch_3",
                "data_batch_4",
                "data_batch_5",
            ]

            data = np.empty((50000, 32, 32, 3), int)
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

        elif self.dataset == "places2":
            self.org_path = "../../datasets/data_256"
            imgs = []
            ctr = 0
            for f in os.listdir(self.org_path):
                self.org2_path = os.path.join(self.org_path,f)
                for fold in os.listdir(self.org2_path):
                    self.path = os.path.join(self.org2_path,fold)
                    for filename in os.listdir(self.path):
                        img = cv2.imread(os.path.join(self.path,filename))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        if img is not None:
                            imgs.append(img)

                        ctr += 1

            self.data = np.array(imgs)

            print(
                f"Data Array has been created from places2 dataset with shape: {self.data.shape}"
            )

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

            fig=plt.figure(figsize=(8, 8))
            columns = 8
            rows = 8
            for i in range(1, columns*rows +1):
                sample = random.randint(0,self.data.shape[0]-1)
                img = self.data[sample]
                fig.add_subplot(rows, columns, i)
                plt.imshow(img.permute(1,2,0).numpy())
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
        for i in range(2):
            sample = random.randint(0,self.data.shape[0]-1)
            im = self.data[sample]
            masked_im = self.masked_data[sample]
            mask = self.masks[sample]
            edge = self.edges[sample]
            mask = mask.squeeze()
            edge = edge.squeeze()
            fig=plt.figure(figsize=(1, 2))
            fig.add_subplot(2, 2, 1)
            plt.imshow(im.permute(1,2,0).numpy())
            fig.add_subplot(2, 2, 2)
            plt.imshow(masked_im.permute(1,2,0).numpy())
            fig.add_subplot(2, 2, 3)
            plt.imshow(mask.numpy(), cmap='gray')
            fig.add_subplot(2, 2, 4)
            plt.imshow(edge.numpy(), cmap='gray')
            plt.show()

    def create_masked_data(self):
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
        self.masks = np.empty_like(self.data[:, :, :, 0])
        self.gray_data = np.empty_like(self.data[:, :, :, 0])
        self.edges = np.empty_like(self.data[:, :, :, 0])

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
                    cv2.line(mask,(x1,y1),(x2,y2),(255,255,255),thickness)

                ## Mask the image
                masked_image = self.data[img].copy()
                masked_image[mask == 255] = 255
                mask = mask[
                    :, :, 0
                ]  # Mask should be 2 dimensional for the rest of the operations
                self.masks[img] = mask
                self.gray_data[img] = cv2.cvtColor(self.data[img], cv2.COLOR_RGB2GRAY)
                self.edges[img] = cv2.Canny(self.gray_data[img], cfg.thresh1, cfg.thresh2)
                self.masked_data[img] = masked_image

        if self.masking_type == '10-20percentage':
            coverage = np.random.uniform(3.2,4.4)
            off_x = int(image_width // coverage)
            off_y = int(image_heigth // coverage)


            for img in range(self.data.shape[0]):
                start_x = np.random.randint(0, image_width-off_x)
                start_y = np.random.randint(0, image_heigth-off_y)

                start_point = (start_x, start_y)
                end_point = (start_x + off_x, start_y + off_y)

                mask = np.full((image_width,image_heigth, image_channel), 0, np.uint8) ## White background
                cv2.rectangle(mask, start_point, end_point, (255,255,255), -1)

                ## Mask the image
                masked_image = self.data[img].copy()
                masked_image[mask == 255] = 255
                mask = mask[
                    :, :, 0
                ]  # Mask should be 2 dimensional for the rest of the operations
                self.masks[img] = mask
                self.gray_data[img] = cv2.cvtColor(self.data[img], cv2.COLOR_RGB2GRAY)
                self.edges[img] = cv2.Canny(self.gray_data[img], cfg.thresh1, cfg.thresh2)
                self.edges[img] = np.abs(self.edges[img] - 255)
                self.masked_data[img] = masked_image



    def create_data_loaders(self):
        """
        Input:
            none
        Output:
            none
        Description:
            Creates necessary data laoders for pytorch with specified batch size.
        """

        self.get_data()
        self.create_masked_data()

        self.data = torch.FloatTensor(self.data) / 255
        self.masked_data = torch.FloatTensor(self.masked_data) / 255
        self.gray_data = torch.FloatTensor(self.gray_data) / 255
        self.masks = torch.FloatTensor(self.masks) / 255
        self.edges = torch.FloatTensor(self.edges) / 255

        self.gray_data = self.gray_data.unsqueeze(1)
        self.edges = self.edges.unsqueeze(1)
        self.masks = self.masks.unsqueeze(1)
        self.data = self.data.permute(0, 3, 1, 2)
        self.masked_data = self.masked_data.permute(0, 3, 1, 2)

        print(f"Masks shape: {self.masks.shape}")
        print(f"Edges shape: {self.edges.shape}")
        print(f"Gray_data shape: {self.gray_data.shape}")
        print(f"masked_data shape: {self.masked_data.shape}")
        print(f"data shape: {self.data.shape}")

        dataset = torch.utils.data.TensorDataset(
            self.data, self.masked_data, self.gray_data, self.masks, self.edges
        )
        self.train_data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size
        )

        self.data = torch.FloatTensor(self.data)
        self.masked_data = torch.FloatTensor(self.masked_data)
        dataset = torch.utils.data.TensorDataset(self.masked_data) ## Buraya daha sonra bir bak hata var gibi
        self.test_data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size
        )

    def create_data_loaders_edgeconnect(self):
        """
        Input:
            none
        Output:
            none
        Description:
            Creates necessary data laoders for pytorch with specified batch size.
        """

        dataset = torchvision.datasets.ImageFolder(root='../../datasets/data_256')
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=True)
        print(len(train_loader))
        exit()
        self.get_data()
        self.create_masked_data()

        self.data = torch.FloatTensor(self.data) / 255
        self.masked_data = torch.FloatTensor(self.masked_data) / 255
        self.gray_data = torch.FloatTensor(self.gray_data) / 255
        self.masks = torch.FloatTensor(self.masks) / 255
        self.edges = torch.FloatTensor(self.edges) / 255

        self.gray_data = self.gray_data.unsqueeze(1)
        self.edges = self.edges.unsqueeze(1)
        self.masks = self.masks.unsqueeze(1)
        self.data = self.data.permute(0, 3, 1, 2)
        self.masked_data = self.masked_data.permute(0, 3, 1, 2)

        print(f"Masks shape: {self.masks.shape}")
        print(f"Edges shape: {self.edges.shape}")
        print(f"Gray_data shape: {self.gray_data.shape}")
        print(f"masked_data shape: {self.masked_data.shape}")
        print(f"data shape: {self.data.shape}")

        dataset = torch.utils.data.TensorDataset(
            self.data, self.gray_data, self.masks, self.edges
        )
        self.train_data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size
        )

        self.data = torch.FloatTensor(self.data)
        self.masked_data = torch.FloatTensor(self.masked_data)
        dataset = torch.utils.data.TensorDataset(self.masked_data) ## Buraya daha sonra bir bak hata var gibi
        self.test_data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size
        )


if __name__ == "__main__":
    data_class = dataRead(dataset='places2')
    data_class.get_data()
    data_class.show_sample_data()
    data_class.create_masked_data()
    data_class.show_masked_and_original()

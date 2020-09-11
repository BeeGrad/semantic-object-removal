import torch

class Config():
    def __init__(self):
        """
        Input:
            none
        Output:
            none
        Description:
            Choose the preferences that will be used throughout the training
            process
        """
        # Train Parameters
        self.DEVICE = torch.device("cpu")
        ''' Choose to train on cpu or gpu '''
        self.epoch_num = 100
        ''' Choose number of epochs '''

        # Data Parameters
        self.dataset = "places2"
        ''' Current Choices for datasets:
                -places2
                -cifar10
                '''
        self.batch_size = 2
        ''' Batch Size for DataLoader '''
        self.masking_type = "lines"
        ''' Current Choices for masking types:
                -lines
                '''
        self.show_sample_data = False
        ''' Choose if a sample from dataset will be shown before training'''
        self.show_masked_data = False
        ''' Choose if a sample from masked data will be shown before training'''
        self.SIGMA = 2
        ''' Parameter for canny edge detector '''
        self.max_pixel_value = 255.0
        ''' Maximum value in an image, necessary to calculate PSNR '''

        # Model Parameters
        self.model = "EdgeConnect"
        ''' Current Choices for Deep Learning Models:
                -EdgeConnect
                '''
        self.saveName = f"{self.model}Model"
        ''' Save name that is going to be used while training '''
        self.loadModel = True
        ''' Choose to load model '''
        self.loadName = f"{self.model}Model"
        ''' Load name to load the pre-trained model '''
        self.LR = 0.0001
        self.BETA1 = 0.0
        self.BETA2 = 0.9
        self.D2G_LR = 0.1
        ''' Learnin rate and Beta parameters for training'''
        self.GAN_LOSS = "nsgan"
        ''' Possible Choices :nsgan | lsgan | hinge '''
        self.FM_LOSS_WEIGHT = 10
        ''' Feature-matching loss weight '''
        self.L1_LOSS_WEIGHT = 1
        self.STYLE_LOSS_WEIGHT = 1
        self.CONTENT_LOSS_WEIGHT = 1
        self.INPAINT_ADV_LOSS_WEIGHT = 0.01
        ''' Loss weights '''
        self.edge_gen_path = f"../saves/{self.saveName}/EdgeGenerator.pt"
        self.edge_disc_path = f"../saves/{self.saveName}/EdgeDiscriminator.pt"
        self.inpaint_gen_path = f"../saves/{self.saveName}/InpaintGenerator.pt"
        self.inpaint_disc_path = f"../saves/{self.saveName}/InpaintDiscriminator.pt"
        ''' Save locations for generator and discriminator for edge and inpaint models '''

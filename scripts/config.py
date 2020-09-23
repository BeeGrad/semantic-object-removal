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
        self.batch_size = 1
        ''' Batch Size for DataLoader '''
        self.masking_type = "10-20percentage"
        ''' Current Choices for masking types:
                -lines
                -10-20percentage
                '''
        self.show_sample_data = False
        ''' Choose if a sample from dataset will be shown before training'''
        self.show_masked_data = True
        ''' Choose if a sample from masked data will be shown before training'''
        self.SIGMA = 1
        ''' Parameter for canny edge detector '''
        self.max_pixel_value = 255.0
        ''' Maximum value in an image, necessary to calculate PSNR '''

        # Model Parameters
        self.model = "EdgeConnect"
        ''' Current Choices for Deep Learning Models:
                -EdgeConnect
                -Contextual
                '''
        self.saveName = f"{self.model}Model"
        ''' Save name that is going to be used while training '''
        self.loadModel = True
        ''' Choose to load model '''
        self.loadName = f"{self.model}Model"
        ''' Load name to load the pre-trained model '''
        self.edge_LR = 0.0001
        self.edge_BETA1 = 0.0
        self.edge_BETA2 = 0.9
        self.edge_D2G_LR = 0.1
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
        self.edge_gen_path = f"../saves/{self.saveName}/pretrainedOur/EdgeGenerator.pt"
        self.edge_disc_path = f"../saves/{self.saveName}/pretrainedOur/EdgeDiscriminator.pt"
        self.inpaint_gen_path = f"../saves/{self.saveName}/pretrainedOur/InpaintGenerator.pt"
        self.inpaint_disc_path = f"../saves/{self.saveName}/pretrainedOur/InpaintDiscriminator.pt"
        ''' Save locations for generator and discriminator for edge and inpaint models for train'''

        # Test parameters
        self.test_im_path = '../foreground-substraction/test/test005.jpg'
        ''' Location of the image that is used to eval the program '''
        self.test_mask_method = 'freely_select_from_image'
        ''' Method to mask the test image
            - freely_select_from_image
            - select_by_edge
         '''
        self.freely_select_mask_size = 15
        ''' Size of the brush for freely select method '''
        self.thresh1 = 50
        self.thresh2 = 100
        ''' Threshold values for canny edge detection '''
        self.test_inpaint_method = 'EdgeConnect'
        ''' Method to inpaint the test image
            - Mathematical
            - EdgeConnect
        '''
        self.test_edge_gen_path = f"../saves/{self.saveName}/pretrainedPaper/EdgeGenerator.pth"
        self.test_edge_disc_path = f"../saves/{self.saveName}/pretrainedPaper/EdgeDiscriminator.pth"
        self.test_inpaint_gen_path = f"../saves/{self.saveName}/pretrainedPaper/InpaintGenerator.pth"
        self.test_inpaint_disc_path = f"../saves/{self.saveName}/pretrainedPaper/InpaintDiscriminator.pth"
        ''' Save locations for generator and discriminator for edge and inpaint models for test'''

        # Traditional Models parameters
        self.mathematical_method = "navier-strokes"
        ''' Method to use in mathematical inpainting
            - navier-strokes
            - fast-marching
         '''

         # Contextual Generative Model parameters
        self.context_activation = 'elu'
        ''' Activation function that is going to be used in training of generative contual CNN
            -elu
            -relu
            -lrelu
        '''
        self.context_conv_type = 'normal'
        ''' Convolution layer type for contextual model
         -normal
         -transpose
        '''
        self.context_input_dim = 3
        self.context_gen_feat_dim = 32
        self.context_dis_feat_dim = 32
        ''' Input and output sizes for gen and dis networks '''
        self.use_cuda = False
        ''' Choose to use cuda '''
        self.context_LR = 0.0001
        self.context_BETA1 = 0.5
        self.context_BETA2 = 0.9
        ''' Optimizer parameters '''
        self.context_image_shape = [256,256,3]
        self.context_mask_shape = [128,128]
        self.context_margin = [0,0]
        self.mask_batch_same = True
        self.context_batch_size = 1
        ''' Random bbox parameters '''
        self.context_global_wgan_loss_alpha = 1.0
        ''' Context train parameters '''
        self.n_critic = 5
        ''' iteration number to compute g loss '''
        self.spatial_discounting_mask = 0.9
        self.discounted_mask = True
        ''' Spatial discounting mask parameters '''
        self.coarse_l1_alpha = 1.2
        self.context_l1_loss_alpha =1.2
        self.context_ae_loss_alpha =1.2
        self.context_gan_loss_alpha =0.001
        self.context_wgan_gp_lambda = 10
        ''' Context train parameters '''

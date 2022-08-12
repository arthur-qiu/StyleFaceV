import torch
from .base_model import BaseModel
from . import networks
from . import lmcode_networks
from . import diy_networks
from . import resnet

import dnnlib
import legacy
import torch.nn.functional as F
import numpy as np
import random

def draw_points(image, pose):
    new_image = image.clone()
    size = 256
    for i in range(pose.shape[0]):
        for j in range(pose.shape[1]):
            pose_w = int(pose[i, j, 0] * size)
            pose_h = int(pose[i, j, 1] * size)
            new_image[i, 0, pose_h-3:pose_h+3, pose_w-3:pose_w+3] = -1
            new_image[i, 1, pose_h - 3:pose_h + 3, pose_w - 3:pose_w + 3] = -1
            new_image[i, 2, pose_h - 3:pose_h + 3, pose_w - 3:pose_w + 3] = -1

    return new_image

def make_transform(translate, angle):
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m

def sample_trans():
    translate = (0.5 * (random.random() - 0.5), 0.5 * (random.random() - 0.5))
    rotate = 90 * (random.random() - 0.5)
    m = make_transform(translate, rotate)
    m = np.linalg.inv(m)
    return m

class StylePoseModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='jointset')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--num_point', type=int, default=14)
            parser.add_argument('--dataroot2', type=str, default='../data/realign1024x1024_random-shift0.1')
            parser.add_argument('--lambda_L1', type=float, default=1.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_L2', 'G_L2B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.visual_names = ['real_A', 'real_B', 'fake_A', 'fake_B']
        if self.isTrain:
            self.model_names = ['FE']
        else:  # during test time, only load G
            self.model_names = ['FE']
            self.visual_names = ['real_A', 'real_B', 'fake_A', 'fake_B', 'real_A_map', 'real_B_map']
        # define networks (both generator and discriminator)
        with dnnlib.util.open_url(opt.network_pkl) as f:
            self.netG = legacy.load_network_pkl(f)['G_ema'].to(self.gpu_ids[0])  # type: ignore

        lm_path = 'pretrain/wing.ckpt'
        self.netFE_lm = lmcode_networks.FAN(fname_pretrained=lm_path).eval().to(self.gpu_ids[0])

        self.netFE = diy_networks._resposenet(num_point=opt.num_point).to(self.gpu_ids[0])

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL2 = torch.nn.MSELoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_FE = torch.optim.Adam(self.netFE.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_FE)

            # Load VGG16 feature detector.
            url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
            with dnnlib.util.open_url(url) as f:
                self.vgg16 = torch.jit.load(f).eval().to(self.gpu_ids[0])

        self.m_zero = make_transform((0.0,0.0),(0.0))


    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.image_paths = input['B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

        if self.real_A.shape[2] != 256:
            self.real_A = F.interpolate(self.real_A, size=(256, 256), mode='area')
        self.real_A_heat = self.netFE_lm.get_heatmap(self.real_A, b_preprocess=False)
        self.real_A_pose = self.netFE(self.real_A_heat).view(-1, 14, 2)
        self.real_A_lm = self.netFE_lm.get_landmark(self.real_A).detach().to(self.device) / 256
        self.real_A_key = torch.cat([
            torch.sum(self.real_A_lm[:, 33:42, ...], dim=1, keepdim=True) / 9,
            torch.sum(self.real_A_lm[:, 42:51, ...], dim=1, keepdim=True) / 9,
            self.real_A_lm[:, [62], ...],
            self.real_A_lm[:, [96], ...],
            self.real_A_lm[:, [66], ...],
            self.real_A_lm[:, [70], ...],
            self.real_A_lm[:, [97], ...],
            self.real_A_lm[:, [74], ...],
            self.real_A_lm[:, [54], ...],
            self.real_A_lm[:, [79], ...],
            self.real_A_lm[:, [85], ...],
            self.real_A_lm[:, [88], ...],
            self.real_A_lm[:, [92], ...],
            torch.sum(self.real_A_lm[:, [90,94], ...], dim=1, keepdim=True) / 2,
        ], 1)

        self.fake_A = draw_points(self.real_A.clone(), self.real_A_pose)

    def forward_B(self):
        if self.real_B.shape[2] != 256:
            self.real_B = F.interpolate(self.real_B, size=(256, 256), mode='area')
        self.real_B_heat = self.netFE_lm.get_heatmap(self.real_B, b_preprocess=False)
        self.real_B_pose = self.netFE(self.real_B_heat).view(-1, 14, 2)
        self.real_B_lm = self.netFE_lm.get_landmark(self.real_B).detach().to(self.device) / 256
        self.real_B_key = torch.cat([
            torch.sum(self.real_B_lm[:, 33:42, ...], dim=1, keepdim=True) / 9,
            torch.sum(self.real_B_lm[:, 42:51, ...], dim=1, keepdim=True) / 9,
            self.real_B_lm[:, [62], ...],
            self.real_B_lm[:, [96], ...],
            self.real_B_lm[:, [66], ...],
            self.real_B_lm[:, [70], ...],
            self.real_B_lm[:, [97], ...],
            self.real_B_lm[:, [74], ...],
            self.real_B_lm[:, [54], ...],
            self.real_B_lm[:, [79], ...],
            self.real_B_lm[:, [85], ...],
            self.real_B_lm[:, [88], ...],
            self.real_B_lm[:, [92], ...],
            torch.sum(self.real_B_lm[:, [90, 94], ...], dim=1, keepdim=True) / 2,
        ], 1)

        self.fake_B = draw_points(self.real_B.clone(), self.real_B_pose)

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""

        # Second, G(A) = B
        self.loss_G_L2 = 10 * self.criterionL2(self.real_A_pose, self.real_A_key)
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_L2
        self.loss_G.backward()

    def backward_G_B(self):
        """Calculate GAN and L1 loss for the generator"""

        # Second, G(A) = B
        self.loss_G_L2B = 10 * self.criterionL2(self.real_B_pose, self.real_B_key)
        # combine loss and calculate gradients
        self.loss_G_B = self.loss_G_L2B
        self.loss_G_B.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update G
        self.optimizer_FE.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_FE.step()  # udpate G's weights

        self.forward_B()
        self.optimizer_FE.zero_grad()        # set G's gradients to zero
        self.backward_G_B()                   # calculate graidents for G
        self.optimizer_FE.step()  # udpate G's weights

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        self.forward_B()
        self.criterionL2 = torch.nn.MSELoss()
        self.loss_G_L2 = 10 * self.criterionL2(self.real_A_pose, self.real_A_key)
        self.loss_G_L2B = 10 * self.criterionL2(self.real_B_pose, self.real_B_key)
        self.fake_A = draw_points(self.real_A, self.real_A_pose)
        self.fake_B = draw_points(self.real_B, self.real_B_pose)
        self.real_A = draw_points(self.real_A, self.real_A_key)
        self.real_B = draw_points(self.real_B, self.real_B_key)

        self.real_A_map = self.netFE(self.real_A_heat, mode=1).detach()
        self.real_A_map = (self.real_A_map - torch.min(self.real_A_map)) / (torch.max(self.real_A_map) -torch.min(self.real_A_map)) * 2 -1
        self.real_B_map = self.netFE(self.real_B_heat, mode=1).detach()
        self.real_B_map = (self.real_B_map - torch.min(self.real_B_map)) / (
                    torch.max(self.real_B_map) - torch.min(self.real_B_map)) * 2 - 1
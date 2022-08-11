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
import os

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

class StylePreModel(BaseModel):
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
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='noise')
        parser.add_argument('--pose_path', type=str, default='', help='path for pose net')
        parser.add_argument('--num_point', type=int, default=14)
        parser.add_argument('--pre_path', type=str, default='', help='path for pretrain')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=1.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_L1', 'G_VGG', 'G_W']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_B', 'real_A', 'fake_B']
        # self.visual_names = ['real_A', 'fake_B', 'real_B', 'fake_C', 'real_C', 'fake_AB', 'fake_AC', 'fake_BA', 'fake_CA']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['FE']
        else:  # during test time, only load G
            self.model_names = ['FE']
        # define networks (both generator and discriminator)
        with dnnlib.util.open_url(opt.network_pkl) as f:
            self.netG = legacy.load_network_pkl(f)['G_ema'].eval().to(self.gpu_ids[0])  # type: ignore

        lm_path = 'pretrained_models/wing.ckpt'
        self.netFE_lm = lmcode_networks.FAN(fname_pretrained=lm_path).eval().to(self.gpu_ids[0])
        # self.netFE_lm = torch.nn.DataParallel(self.netFE_lm, self.gpu_ids)
        self.netFE_pose = diy_networks._resposenet(num_point=opt.num_point).eval().to(self.gpu_ids[0])
        if opt.pose_path != '':
            # pose_path = 'checkpoints/ffhq_stylevideopose/latest_net_FE.pth'
            self.netFE_pose.load_state_dict(torch.load(opt.pose_path))
        # self.netFE_pose = torch.nn.DataParallel(self.netFE_pose, self.gpu_ids)

        self.netFE = resnet.wide_resdisnet50_2(num_classes=512 * 16).to(self.gpu_ids[0])
        # self.netFE = networks.init_net(self.netFE, opt.init_type, opt.init_gain, self.gpu_ids)
        if opt.pre_path != '':
            try:
                self.netFE.load_state_dict(torch.load(opt.pre_path), strict=True)
            except:
                import collections
                model_dic = torch.load(opt.pre_path)
                new_state_dict = collections.OrderedDict()
                for k, v in model_dic.items():
                    name = k.replace('module.', '')
                    new_state_dict[name] = v
                self.netFE.load_state_dict(new_state_dict, strict=True)

        # self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
        #                               not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            # self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_FE = torch.optim.Adam(self.netFE.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            # self.optimizer_FD = torch.optim.Adam(self.netFD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            # self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            # self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_FE)
            # self.optimizers.append(self.optimizer_FD)
            # self.optimizers.append(self.optimizer_D)

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
        self.real_z = input['A'].to(self.device)
        self.image_paths = input['A_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if hasattr(self.netG.synthesis, 'input'):
            self.netG.synthesis.input.transform.copy_(torch.from_numpy(self.m_zero))
        # label = torch.zeros([self.rresneteal_z.shape[0], 0], device=self.gpu_ids[0])

        # with torch.no_grad():
            # self.netG.synthesis.input.transform.copy_(torch.from_numpy(self.m_zero))
        with torch.no_grad():
            self.real_A_w = self.netG.mapping(self.real_z, None)
            self.real_A = self.netG.synthesis(self.real_A_w, noise_mode='const').detach().clamp(-1, 1)
            if self.real_A.shape[2] != 256:
                self.real_A = F.interpolate(self.real_A, size=(256, 256), mode='area')
            self.real_A_heat = self.netFE_lm.get_heatmap(self.real_A, b_preprocess=False)
            self.real_A_pose = self.netFE_pose(self.real_A_heat, mode=1).detach()
            # self.real_A_app = self.netFE(self.real_A, mode=1).detach()
            # self.fake_A_w = self.netFE(self.real_A, mode=2).view(-1, 16, 512)

            m = sample_trans()
            self.netG.synthesis.input.transform.copy_(torch.from_numpy(m))
            self.real_B = self.netG.synthesis(self.real_A_w, noise_mode='const').detach().clamp(-1, 1)
            if self.real_B.shape[2] != 256:
                self.real_B = F.interpolate(self.real_B, size=(256, 256), mode='area')
            # self.real_B_heat = self.netFE_lm.get_heatmap(self.real_B, b_preprocess=False)
            # self.real_B_pose = self.netFE_pose(self.real_B_heat, mode=1).detach()
        self.real_B_app = self.netFE(self.real_B, mode=1)
        self.fake_B_w = self.netFE(self.real_B_app, self.real_A_pose, mode=2).view(-1, 16, 512)

        self.netG.synthesis.input.transform.copy_(torch.from_numpy(self.m_zero))
        self.fake_B = self.netG.synthesis(self.fake_B_w, noise_mode='const')  # G(A)
        if self.fake_B.shape[2] != 256:
            self.fake_B = F.interpolate(self.fake_B, size=(256, 256), mode='area')

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""

        # Second, G(A) = B
        self.loss_G_L1 = 1 * self.opt.lambda_L1 * self.criterionL1(self.fake_B, self.real_A)

        self.loss_G_VGG = 100 * self.opt.lambda_L1 * self.criterionL1(self.vgg16(self.fake_B), self.vgg16(self.real_A))

        # self.loss_G_APP = self.opt.lambda_L1 * self.criterionL1(self.real_B_app, self.real_A_app)

        self.loss_G_W = 100 * self.opt.lambda_L1 * self.criterionL1(self.fake_B_w[:,1:,:], self.real_A_w[:,1:,:])

        # combine loss and calculate gradients
        self.loss_G = self.loss_G_L1 + self.loss_G_VGG + self.loss_G_W
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update G
        self.optimizer_FE.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_FE.step()  # udpate G's weights

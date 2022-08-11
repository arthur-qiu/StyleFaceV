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

from . import rnn_net

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

class SampleModel(BaseModel):
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
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='noiseshufflevideo', num_test = 32)
        parser.add_argument('--pose_path', type=str, default='', help='path for pose net')
        parser.add_argument('--rnn_path', type=str, default='', help='path for rnn net')
        parser.add_argument('--n_frames_G', type=int, default=60)
        parser.add_argument('--w_residual', type=float, default=0.2)
        parser.add_argument('--num_point', type=int, default=14)
        parser.add_argument('--model_names', type=str, default='')
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
        self.visual_names = ['real_vid_B', 'fake_vid_AR', 'fake_vid_BR', 'fake_vid_AR1', 'fake_vid_BR1', 'fake_vid_AR2', 'fake_vid_BR2', 'fake_vid_AB', 'fake_vid_B', 'fake_vid']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['FE']
        else:  # during test time, only load G
            self.model_names = ['FE']
        if opt.model_names != '':
            str_models = opt.model_names.split(',')
            self.model_names = []
            for str_model in str_models:
                self.model_names.append(str_model)
        # define networks (both generator and discriminator)
        with dnnlib.util.open_url(opt.network_pkl) as f:
            self.netG = legacy.load_network_pkl(f)['G_ema'].eval().to(self.gpu_ids[0])  # type: ignore

        lm_path = 'pretrain/wing.ckpt'
        self.netFE_lm = lmcode_networks.FAN(fname_pretrained=lm_path).eval().to(self.gpu_ids[0])
        self.netFE_pose = diy_networks._resposenet(num_point=opt.num_point).eval().to(self.gpu_ids[0])
        if opt.pose_path != '':
            self.netFE_pose.load_state_dict(torch.load(opt.pose_path))

        self.netFE = resnet.wide_resdisnet50_2(num_classes=512 * 16).to(self.gpu_ids[0])
        self.netFE = networks.init_net(self.netFE, opt.init_type, opt.init_gain, self.gpu_ids)

        self.netR = rnn_net.RNNModule(w_residual = opt.w_residual).to(self.gpu_ids[0])
        if opt.rnn_path != '':
            self.netR.load_state_dict(torch.load(opt.rnn_path))
        self.n_frames_G = opt.n_frames_G
        self.style_gan_size = 8

        self.m_zero = make_transform((0.0,0.0),(0.0))
        self.count = 0


    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        self.real_Bs = input['A'].to(self.device)
        self.image_paths = input['A_paths']
        self.count += 1
        self.image_paths[0] = os.path.split(self.image_paths[0])[0] + '/' + str(self.count) + '.png'

        real_v_list = []
        with torch.no_grad():
            for i in range(self.real_Bs.shape[1]):
                real_v_list.append(self.netFE_pose(self.netFE_lm.get_heatmap(self.real_Bs[:,i,...], b_preprocess=False), mode = 1).unsqueeze(1))

        self.real_v = torch.cat(real_v_list, 1).detach()

        self.real_z = input['B'].to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

        self.real_A_w = self.netG.mapping(self.real_z, None)
        self.real_A = self.netG.synthesis(self.real_A_w, noise_mode='const').detach().clamp(-1, 1)
        if self.real_A.shape[2] != 256:
            self.real_A = F.interpolate(self.real_A, size=(256, 256), mode='area')
        self.real_A_heat = self.netFE_lm.get_heatmap(self.real_A, b_preprocess=False)
        self.real_A_pose = self.netFE_pose(self.real_A_heat, mode=1).detach()
        self.real_A_app = self.netFE(self.real_A, mode=1).detach()
        self.fake_A_w = self.netFE(self.real_A_app, self.real_A_pose, mode=2).view(-1, 16, 512)
        self.fake_A = self.netG.synthesis(self.fake_A_w, noise_mode='const')  # G(A)

        self.real_B_app = self.netFE(self.real_Bs[:, 0, ...], mode=1)

        x_fake, self.rand_in, self.rand_rec = self.netR(self.real_v[:, 0].view(self.opt.batch_size, self.style_gan_size * self.style_gan_size), self.n_frames_G)
        x_fake = x_fake.view(self.opt.batch_size, self.n_frames_G, 1, self.style_gan_size,
                             self.style_gan_size)

        self.real_R_pose = x_fake.clone()

        x_fake, self.rand_in, self.rand_rec = self.netR(self.real_v[:, 29].view(self.opt.batch_size, self.style_gan_size * self.style_gan_size), self.n_frames_G)
        x_fake = x_fake.view(self.opt.batch_size, self.n_frames_G, 1, self.style_gan_size,
                             self.style_gan_size)

        self.real_R1_pose = x_fake.clone()

        x_fake, self.rand_in, self.rand_rec = self.netR(self.real_v[:, 59].view(self.opt.batch_size, self.style_gan_size * self.style_gan_size), self.n_frames_G)
        x_fake = x_fake.view(self.opt.batch_size, self.n_frames_G, 1, self.style_gan_size,
                             self.style_gan_size)

        self.real_R2_pose = x_fake.clone()

        x_fake_A, self.rand_in, self.rand_rec = self.netR(self.real_A_pose.view(self.opt.batch_size, self.style_gan_size * self.style_gan_size), self.n_frames_G)
        x_fake_A = x_fake_A.view(self.opt.batch_size, self.n_frames_G, 1, self.style_gan_size,
                             self.style_gan_size)

        self.real_R_pose_A = x_fake_A

        if hasattr(self.netG.synthesis, 'input'):
            self.netG.synthesis.input.transform.copy_(torch.from_numpy(self.m_zero))

        self.real_A_list = []
        self.real_B_list = []
        self.fake_AR_list = []
        self.fake_BR_list = []
        self.fake_AR1_list = []
        self.fake_BR1_list = []
        self.fake_AR2_list = []
        self.fake_BR2_list = []
        self.fake_AB_list = []
        self.fake_B_list = []
        # for i in range(self.real_Bs.shape[1]):
        self.real_B_app = self.netFE(self.real_Bs[:,0,...], mode=1)
        for i in range(self.n_frames_G):
            self.real_B = self.real_Bs[:,i,...]
            if self.real_B.shape[2] != 256:
                self.real_B = F.interpolate(self.real_B, size=(256, 256), mode='area')

            self.fake_AR_w = self.netFE(self.real_A_app, self.real_R_pose[:,i,...], mode=2).view(-1, 16, 512)
            self.fake_BR_w = self.netFE(self.real_B_app, self.real_R_pose[:,i,...], mode=2).view(-1, 16, 512)
            self.fake_AR1_w = self.netFE(self.real_A_app, self.real_R1_pose[:,i,...], mode=2).view(-1, 16, 512)
            self.fake_BR1_w = self.netFE(self.real_B_app, self.real_R1_pose[:,i,...], mode=2).view(-1, 16, 512)
            self.fake_AR2_w = self.netFE(self.real_A_app, self.real_R2_pose[:,i,...], mode=2).view(-1, 16, 512)
            self.fake_BR2_w = self.netFE(self.real_B_app, self.real_R2_pose[:,i,...], mode=2).view(-1, 16, 512)
            self.fake_AB_w = self.netFE(self.real_A_app, self.real_R_pose_A[:,i,...], mode=2).view(-1, 16, 512)
            self.fake_B_w = self.netFE(self.real_B_app, self.real_R_pose_A[:,i,...], mode=2).view(-1, 16, 512)

            self.fake_AR = self.netG.synthesis(self.fake_AR_w, noise_mode='const')  # G(A)
            self.fake_BR = self.netG.synthesis(self.fake_BR_w, noise_mode='const')  # G(A)
            self.fake_AR1 = self.netG.synthesis(self.fake_AR1_w, noise_mode='const')  # G(A)
            self.fake_BR1 = self.netG.synthesis(self.fake_BR1_w, noise_mode='const')  # G(A)
            self.fake_AR2 = self.netG.synthesis(self.fake_AR2_w, noise_mode='const')  # G(A)
            self.fake_BR2 = self.netG.synthesis(self.fake_BR2_w, noise_mode='const')  # G(A)
            self.fake_AB = self.netG.synthesis(self.fake_AB_w, noise_mode='const')  # G(A)
            self.fake_B = self.netG.synthesis(self.fake_B_w, noise_mode='const')  # G(A)

            self.real_A_list.append(self.real_A.clamp(-1, 1))
            self.real_B_list.append(self.real_B.clamp(-1, 1))
            self.fake_AR_list.append(self.fake_AR.clamp(-1, 1))
            self.fake_BR_list.append(self.fake_BR.clamp(-1, 1))
            self.fake_AR1_list.append(self.fake_AR1.clamp(-1, 1))
            self.fake_BR1_list.append(self.fake_BR1.clamp(-1, 1))
            self.fake_AR2_list.append(self.fake_AR2.clamp(-1, 1))
            self.fake_BR2_list.append(self.fake_BR2.clamp(-1, 1))
            self.fake_AB_list.append(self.fake_AB.clamp(-1, 1))
            self.fake_B_list.append(self.fake_B.clamp(-1, 1))

    def compute_visuals(self):

        self.real_vid_A = torch.cat(self.real_A_list, 0)
        self.real_vid_B = torch.cat(self.real_B_list, 0)
        self.fake_vid_AR = torch.cat(self.fake_AR_list, 0)
        self.fake_vid_BR = torch.cat(self.fake_BR_list, 0)
        self.fake_vid_AR1 = torch.cat(self.fake_AR1_list, 0)
        self.fake_vid_BR1 = torch.cat(self.fake_BR1_list, 0)
        self.fake_vid_AR2 = torch.cat(self.fake_AR2_list, 0)
        self.fake_vid_BR2 = torch.cat(self.fake_BR2_list, 0)
        self.fake_vid_AB = torch.cat(self.fake_AB_list, 0)
        self.fake_vid_B = torch.cat(self.fake_B_list, 0)

        self.fake_vid = torch.cat([torch.cat([self.real_vid_B, self.fake_vid_BR, self.fake_vid_BR1, self.fake_vid_BR2, self.fake_vid_B], dim = 3), torch.cat([self.real_vid_A, self.fake_vid_AR, self.fake_vid_AR1, self.fake_vid_AR2, self.fake_vid_AB], dim = 3)], dim = 2)


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

def sample_trans():
    translate = (0.5 * (random.random() - 0.5), 0.5 * (random.random() - 0.5))
    rotate = 90 * (random.random() - 0.5)
    m = make_transform(translate, rotate)
    m = np.linalg.inv(m)
    return m

class SampleModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        """
        # changing the default values
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='noiseshufflevideo', num_test = 32)
        parser.add_argument('--pose_path', type=str, default='', help='path for pose net')
        parser.add_argument('--rnn_path', type=str, default='', help='path for rnn net')
        parser.add_argument('--n_frames_G', type=int, default=30)
        parser.add_argument('--w_residual', type=float, default=0.2)
        parser.add_argument('--num_point', type=int, default=12)
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
        self.visual_names = ['real_B', 'fake_B', 'fake_BA', 'fake_BR', 'real_A', 'fake_A', 'fake_AB', 'fake_AR', 'real_vid_B', 'fake_vid_B', 'fake_vid_AB', 'fake_vid_AR', 'fake_vid_BR', 'fake_vid']
        # self.visual_names = ['real_A', 'fake_B', 'real_B', 'fake_C', 'real_C', 'fake_AB', 'fake_AC', 'fake_BA', 'fake_CA']
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
            # pose_path = 'checkpoints/ffhq_stylevideopose/latest_net_FE.pth'
            self.netFE_pose.load_state_dict(torch.load(opt.pose_path))

        self.netFE = resnet.wide_resdisnet50_2(num_classes=512 * 16).to(self.gpu_ids[0])
        self.netFE = networks.init_net(self.netFE, opt.init_type, opt.init_gain, self.gpu_ids)

        self.netR = rnn_net.RNNModule(w_residual = opt.w_residual).to(self.gpu_ids[0])
        if opt.rnn_path != '':
            self.netR.load_state_dict(torch.load(opt.rnn_path))
        self.n_frames_G = opt.n_frames_G
        self.style_gan_size = 8

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.optimizer_FE = torch.optim.Adam(self.netFE.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_FE)

            # Load VGG16 feature detector.
            url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
            with dnnlib.util.open_url(url) as f:
                self.vgg16 = torch.jit.load(f).eval().to(self.gpu_ids[0])

            if opt.continue_train and os.path.exists('checkpoints/' + opt.name + '/latest_net_FE.pth'):
                self.netFE.load_state_dict(torch.load('checkpoints/' + opt.name + '/latest_net_FE.pth'))

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
        self.image_paths[0] = self.image_paths[0][:-4] + str(self.count) + '.png'

        real_v_list = []
        with torch.no_grad():
            for i in range(self.real_Bs.shape[1]):
                real_v_list.append(self.netFE_pose(self.netFE_lm.get_heatmap(self.real_Bs[:,i,...], b_preprocess=False), mode = 1).unsqueeze(1))

        self.real_v = torch.cat(real_v_list, 1).detach()

        self.real_z = input['B'].to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

        x_fake, self.rand_in, self.rand_rec = self.netR(self.real_v[:, 0].view(self.opt.batch_size, self.style_gan_size * self.style_gan_size), self.n_frames_G)
        x_fake = x_fake.view(self.opt.batch_size, self.n_frames_G, 1, self.style_gan_size,
                             self.style_gan_size)

        if hasattr(self.netG.synthesis, 'input'):
            self.netG.synthesis.input.transform.copy_(torch.from_numpy(self.m_zero))

        self.real_A_w = self.netG.mapping(self.real_z, None)
        self.real_A = self.netG.synthesis(self.real_A_w, noise_mode='const').detach().clamp(-1, 1)
        self.real_A_heat = self.netFE_lm.get_heatmap(self.real_A, b_preprocess=False)
        self.real_A_pose = self.netFE_pose(self.real_A_heat, mode=1).detach()
        self.real_A_app = self.netFE(self.real_A, mode=1).detach()
        self.fake_A_w = self.netFE(self.real_A_app, self.real_A_pose, mode=2).view(-1, 16, 512)
        self.fake_A = self.netG.synthesis(self.fake_A_w, noise_mode='const')  # G(A)

        self.real_B_list = []
        self.fake_B_list = []
        self.real_A_list = []
        self.fake_A_list = []
        self.fake_AB_list = []
        self.fake_BA_list = []
        self.fake_AR_list = []
        self.fake_BR_list = []

        self.real_B_app = self.netFE(self.real_Bs[:,0,...], mode=1)
        for i in range(self.n_frames_G):
            self.real_B = self.real_Bs[:,i,...]
            self.real_B_heat = self.netFE_lm.get_heatmap(self.real_B, b_preprocess=False)
            self.real_B_pose = self.netFE_pose(self.real_B_heat, mode=1).detach()

            self.fake_B_w = self.netFE(self.real_B_app, self.real_B_pose, mode=2).view(-1, 16, 512)
            self.fake_AB_w = self.netFE(self.real_A_app, self.real_B_pose, mode=2).view(-1, 16, 512)
            self.fake_BA_w = self.netFE(self.real_B_app, self.real_A_pose, mode=2).view(-1, 16, 512)

            self.fake_AR_w = self.netFE(self.real_A_app, self.real_R_pose[:,i,...], mode=2).view(-1, 16, 512)
            self.fake_BR_w = self.netFE(self.real_B_app, self.real_R_pose[:,i,...], mode=2).view(-1, 16, 512)

            self.fake_B = self.netG.synthesis(self.fake_B_w, noise_mode='const')  # G(A)
            self.fake_AB = self.netG.synthesis(self.fake_AB_w, noise_mode='const')  # G(A)
            self.fake_BA = self.netG.synthesis(self.fake_BA_w, noise_mode='const')  # G(A)
            self.fake_AR = self.netG.synthesis(self.fake_AR_w, noise_mode='const')  # G(A)
            self.fake_BR = self.netG.synthesis(self.fake_BR_w, noise_mode='const')  # G(A)

            self.real_B_list.append(self.real_B.clamp(-1, 1))
            self.fake_B_list.append(self.fake_B.clamp(-1, 1))
            self.fake_AB_list.append(self.fake_AB.clamp(-1, 1))
            self.fake_BA_list.append(self.fake_BA.clamp(-1, 1))
            self.fake_AR_list.append(self.fake_AR.clamp(-1, 1))
            self.fake_BR_list.append(self.fake_BR.clamp(-1, 1))
            self.real_A_list.append(self.real_A.clamp(-1, 1))
            self.fake_A_list.append(self.fake_A.clamp(-1, 1))

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""

        # Second, G(A) = B
        self.loss_G_L1 = self.opt.lambda_L1 * self.criterionL1(self.fake_B, self.real_A)

        self.loss_G_VGG = 100 * self.opt.lambda_L1 * self.criterionL1(self.vgg16(F.interpolate(self.fake_B, size=(256, 256), mode='area')), self.vgg16(F.interpolate(self.real_A, size=(256, 256), mode='area')))

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

    def compute_visuals(self):
        self.real_A = self.real_A.clamp(-1, 1)
        self.fake_A = self.fake_A.clamp(-1, 1)
        self.fake_AB = self.fake_AB.clamp(-1, 1)
        self.real_B = self.real_B.clamp(-1, 1)
        self.fake_B = self.fake_B.clamp(-1, 1)
        self.fake_BA = self.fake_BA.clamp(-1, 1)
        self.fake_AR = self.fake_AR.clamp(-1, 1)
        self.fake_BR = self.fake_BR.clamp(-1, 1)

        self.real_vid_B = torch.cat(self.real_B_list, 0)
        self.fake_vid_B = torch.cat(self.fake_B_list, 0)
        self.fake_vid_AB = torch.cat(self.fake_AB_list, 0)
        self.fake_vid_BA = torch.cat(self.fake_BA_list, 0)
        self.fake_vid_AR = torch.cat(self.fake_AR_list, 0)
        self.fake_vid_BR = torch.cat(self.fake_BR_list, 0)
        self.real_vid_A = torch.cat(self.real_A_list, 0)
        self.fake_vid_A = torch.cat(self.fake_A_list, 0)

        self.fake_vid = torch.cat([torch.cat([self.real_vid_B, self.fake_vid_B, self.fake_vid_BA, self.fake_vid_BR], dim = 3), torch.cat([self.real_vid_A, self.fake_vid_A, self.fake_vid_AB, self.fake_vid_AR], dim = 3)], dim = 2)


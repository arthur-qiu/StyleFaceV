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
from . import rnn_dnet
from . import rnn_losses

def flip_video(x):
    num = random.randint(0, 1)
    if num == 0:
        return torch.flip(x, [2])
    else:
        return x

class StyleRNNSModel(BaseModel):
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
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='randomvideo', no_flip=True)
        parser.add_argument('--pose_path', type=str, default='', help='path for pose net')
        parser.add_argument('--lm_path', type=str, default='', help='path for lm net')
        parser.add_argument('--n_frames_G', type=int, default=30)
        parser.add_argument('--w_residual', type=float, default=0.2)
        parser.add_argument('--num_point', type=int, default=14)
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
        self.loss_names = ['G_2d', 'G_3d', 'G_mutual', 'D_real', 'D_fake', 'D_real_3d', 'D_fake_3d']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A0', 'fake_A0', 'real_A_final', 'fake_A_final']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D_2d', 'D_3d']
        else:  # during test time, only load G
            self.model_names = ['G']

        self.netG = rnn_net.RNNModule(w_residual = opt.w_residual).to(self.gpu_ids[0])
        # self.netG.init_optim(lr=0.0001, beta1=0.5, beta2=0.999)
        self.n_frames_G = opt.n_frames_G
        self.style_gan_size = 8

        lm_path = 'pretrained_models/wing.ckpt'
        self.netFE_lm = lmcode_networks.FAN(fname_pretrained=lm_path).eval().to(self.gpu_ids[0])
        self.netFE_pose = diy_networks._resposenet(num_point=opt.num_point).eval().to(self.gpu_ids[0])

        if opt.lm_path != '':
            self.netFE_lm.load_state_dict(torch.load(opt.lm_path))
        if opt.pose_path != '':
            self.netFE_pose.load_state_dict(torch.load(opt.pose_path))


        if self.isTrain:
            self.netD_2d = rnn_dnet.ModelD_img().to(self.gpu_ids[0])
            self.netD_3d = rnn_dnet.ModelD_3d(n_frames_G=self.n_frames_G).to(self.gpu_ids[0])
            # define loss functions
            self.criterionGAN = rnn_losses.Relativistic_Average_LSGAN().to(self.device)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_2D = torch.optim.Adam(self.netD_2d.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_3D = torch.optim.Adam(self.netD_3d.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D_2D)
            self.optimizers.append(self.optimizer_D_3D)


    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        real_v_list = []
        self.real_A = input['A'].to(self.device)
        start_index = random.randint(0, self.real_A.shape[1] - self.n_frames_G)
        with torch.no_grad():
            for i in range(start_index, start_index + self.n_frames_G):
                real_v_list.append(self.netFE_pose(self.netFE_lm.get_heatmap(self.real_A[:,i,...], b_preprocess=False), mode = 1).unsqueeze(1))

        self.real_v = torch.cat(real_v_list, 1).detach()

        self.image_paths = input['A_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

        x_fake, self.rand_in, self.rand_rec = self.netG(self.real_v[:, 0].view(self.opt.batch_size, self.style_gan_size * self.style_gan_size), self.n_frames_G)
        x_fake = x_fake.view(self.opt.batch_size, self.n_frames_G, 1, self.style_gan_size,
                             self.style_gan_size)

        frame_id = random.randint(1, self.n_frames_G - 1)
        self.D_fake = self.netD_2d(torch.cat((x_fake[:, 0], x_fake[:, frame_id]), dim=1))
        self.D_real = self.netD_2d(torch.cat((self.real_v[:, 0], self.real_v[:, frame_id]), dim=1).detach())

        x_in = torch.cat((self.real_v[:, 0].unsqueeze(1).repeat(1, self.n_frames_G - 1, 1, 1,
                                                      1), self.real_v[:, 1:]), dim=2)

        x_fake_in = torch.cat((x_fake[:, 0].unsqueeze(1).repeat(
            1, self.n_frames_G - 1, 1, 1, 1), x_fake[:, 1:]),
            dim=2)

        self.D_real_3d = self.netD_3d(flip_video(torch.transpose(x_in, 1, 2)))
        self.D_fake_3d = self.netD_3d(flip_video(torch.transpose(x_fake_in, 1, 2)))

        self.real_A0 = self.real_v[:, 0]
        self.fake_A0 = x_fake[:, 0]
        self.real_A_final = self.real_v[:, -1]
        self.fake_A_final = self.x_fake[:, -1]
        self.real_A0 = (self.real_A0 - torch.min(self.real_A0)) / (
                    torch.max(self.real_A0) - torch.min(self.real_A0)) * 2 - 1
        self.fake_A0 = (self.fake_A0 - torch.min(self.fake_A0)) / (
                    torch.max(self.fake_A0) - torch.min(self.fake_A0)) * 2 - 1
        self.real_A_final = (self.real_A_final - torch.min(self.real_A_final)) / (
                    torch.max(self.real_A_final) - torch.min(self.real_A_final)) * 2 - 1
        self.fake_A_final = (self.fake_A_final - torch.min(self.fake_A_final)) / (
                    torch.max(self.fake_A_final) - torch.min(self.fake_A_final)) * 2 - 1

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""

        # Second, G(A) = B
        self.loss_G_2d = (self.criterionGAN(self.D_fake, self.D_real, True) +
                     self.criterionGAN(self.D_real, self.D_fake, False)) * 0.5

        self.loss_G_3d = (self.criterionGAN(self.D_fake_3d, self.D_real_3d, True) +
                     self.criterionGAN(self.D_real_3d, self.D_fake_3d, False)) * 0.5

        self.loss_G_mutual = -torch.mean(F.cosine_similarity(self.rand_rec, self.rand_in.detach()))

        self.loss_G = self.loss_G_3d + self.loss_G_2d + self.loss_G_mutual
        self.loss_G.backward()

    def forward_D(self):

        x_fake, _, _ = self.netG(self.real_v[:, 0].view(self.opt.batch_size, self.style_gan_size * self.style_gan_size), self.n_frames_G)
        x_fake = x_fake.view(self.opt.batch_size, self.n_frames_G, 1,
                             self.style_gan_size, self.style_gan_size)

        frame_id = random.randint(1, self.n_frames_G - 1)
        self.D_real = self.netD_2d(torch.cat((self.real_v[:, 0], self.real_v[:, frame_id]), dim=1).detach())
        self.D_fake = self.netD_2d(
            torch.cat((x_fake[:, 0], x_fake[:, frame_id]), dim=1).detach())

        self.x_in = torch.cat((self.real_v[:, 0].unsqueeze(1).repeat(1, self.n_frames_G - 1, 1, 1,
                                                      1), self.real_v[:, 1:]),
                         dim=2)
        self.x_fake_in = torch.cat((x_fake[:, 0].unsqueeze(1).repeat(
            1, self.n_frames_G - 1, 1, 1, 1), x_fake[:, 1:]),
            dim=2)

        self.D_fake_3d = self.netD_3d(flip_video(
            torch.transpose(self.x_fake_in, 1, 2).detach()))
        self.D_real_3d = self.netD_3d(flip_video(torch.transpose(self.x_in, 1, 2)))

    def backward_D(self):

        self.loss_D_real_3d = self.criterionGAN(self.D_real_3d, self.D_fake_3d, True)
        self.loss_D_fake_3d = self.criterionGAN(self.D_fake_3d, self.D_real_3d, False)

        self.loss_D_3d = (self.loss_D_real_3d + self.loss_D_fake_3d) * 0.5

        loss_GP_3d = rnn_losses.compute_gradient_penalty_T(
            torch.transpose(self.x_in, 1, 2), torch.transpose(self.x_fake_in, 1, 2),
            self.netD_3d, 2)
        self.loss_D_3d += loss_GP_3d

        self.optimizer_D_3D.zero_grad()  # set D's gradients to zero
        self.loss_D_3d.backward(retain_graph=True)
        self.optimizer_D_3D.step()  # update D's weights

        self.loss_D_real = self.criterionGAN(self.D_real, self.D_fake, True)
        self.loss_D_fake = self.criterionGAN(self.D_fake, self.D_real, False)
        self.loss_D = (self.loss_D_real + self.loss_D_fake) * 0.5

        self.optimizer_D_2D.zero_grad()  # set D's gradients to zero
        self.loss_D.backward()
        self.optimizer_D_2D.step()  # update D's weight

    def optimize_parameters(self):
        # update D
        self.set_requires_grad(self.netD_2d, True)  # enable backprop for D
        self.set_requires_grad(self.netD_3d, True)  # enable backprop for D
        self.forward_D()
        self.backward_D()                # calculate gradients for D

        # update G
        self.forward()  # compute fake images: G(A)
        self.set_requires_grad(self.netD_2d, False)  # D requires no gradients when optimizing G
        self.set_requires_grad(self.netD_3d, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

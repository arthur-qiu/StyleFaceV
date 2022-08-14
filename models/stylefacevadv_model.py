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

class StyleFaceVadvModel(BaseModel):

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
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='noiseframe')
        parser.add_argument('--num_frame', type=int, default=2, help='number of frames')
        parser.add_argument('--seq_frame', type=int, default=3, help='squence of frames')
        parser.add_argument('--max_gap', type=int, default=15, help='max gap')
        parser.add_argument('--pose_path', type=str, default='', help='path for pose net')
        parser.add_argument('--num_point', type=int, default=14)
        parser.add_argument('--pre_path', type=str, default='', help='path for pretrain')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla', epoch_gan = 20)
            parser.add_argument('--lambda_VGG', type=float, default=100.0, help='weight for VGG loss')
            parser.add_argument('--lambda_L1', type=float, default=10.0, help='weight for L1 loss')
            parser.add_argument('--lambda_L2', type=float, default=1.0, help='weight for L2 loss')
            parser.add_argument('--lambda_GAN', type=float, default=0.01, help='weight for videoGAN loss')
            parser.add_argument('--lambda_W', type=float, default=10.0, help='weight for W loss')
            parser.add_argument('--lambda_B', type=float, default=0.1, help='weight for B loss')
            parser.add_argument('--lambda_APP', type=float, default=0.01, help='weight for APP loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_L1', 'G_VGG', 'G_L2', 'G_L1_B', 'G_VGG_B', 'G_L2_B', 'G_APP_B', 'G_W', 'G_GAN', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_B', 'fake_B', 'real_B2', 'real_A_ori', 'real_A', 'real_A2', 'fake_A', 'fake_A2', 'pose_A', 'pose_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['FE', 'FE_pose', 'FE_lm']
        else:  # during test time, only load G
            self.model_names = ['FE', 'FE_pose', 'FE_lm']
        # define networks (both generator and discriminator)
        with dnnlib.util.open_url(opt.network_pkl) as f:
            self.netG = legacy.load_network_pkl(f)['G_ema'].eval().to(self.gpu_ids[0])  # type: ignore

        lm_path = 'pretrained_models/wing.ckpt'
        self.netFE_lm = lmcode_networks.FAN(fname_pretrained=lm_path).to(self.gpu_ids[0])
        self.netFE_lm_fix = lmcode_networks.FAN(fname_pretrained=lm_path).eval().to(self.gpu_ids[0])
        self.netFE_pose = diy_networks._resposenet(num_point=opt.num_point).to(self.gpu_ids[0])
        if opt.pose_path != '':
            self.netFE_pose.load_state_dict(torch.load(opt.pose_path))

        self.netFE = resnet.wide_resdisnet50_2(num_classes=512 * 16).to(self.gpu_ids[0])
        self.netFE = networks.init_net(self.netFE, opt.init_type, opt.init_gain, self.gpu_ids)
        if opt.pre_path != '':
            try:
                self.netFE.load_state_dict(torch.load(opt.pre_path), strict=True)
            except:
                import collections
                model_dic = torch.load(opt.pre_path)
                new_state_dict = collections.OrderedDict()
                for k, v in model_dic.items():
                    name = 'module.' + k
                    new_state_dict[name] = v
                self.netFE.load_state_dict(new_state_dict, strict=True)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.optimizer_FE = torch.optim.Adam(self.netFE.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_FE_pose = torch.optim.Adam(self.netFE_pose.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_FE_lm = torch.optim.Adam(self.netFE_lm.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr * 0.2, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_FE)
            self.optimizers.append(self.optimizer_FE_pose)
            self.optimizers.append(self.optimizer_FE_lm)
            self.optimizers.append(self.optimizer_D)

            # Load VGG16 feature detector.
            url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
            with dnnlib.util.open_url(url) as f:
                self.vgg16 = torch.jit.load(f).eval().to(self.gpu_ids[0])

        self.m_zero = make_transform((0.0,0.0),(0.0))
        self.use_gan_loss = False

    def Use_GAN_Loss(self):
        self.use_gan_loss = True

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        self.real_A_ori = input['A'].to(self.device)
        self.real_As_ori = input['As'].to(self.device)
        self.image_paths = input['A_paths']
        self.real_z = input['B'].to(self.device)
        self.label = (torch.zeros([self.real_z.shape[0], 1]) + random.randint(0, 1)).to(self.device)

    def forward_pre(self):
        self.bs, self.vs, self.c, self.h, self.w = self.real_As_ori.shape
        self.real_As = self.real_As_ori.view(-1, self.c, self.h, self.w)
        with torch.no_grad():
            self.real_As_lm = self.netFE_lm_fix.get_landmark(self.real_As).detach().to(self.device) / 256
            self.real_As_key = torch.cat([
                torch.sum(self.real_As_lm[:, 33:42, ...], dim=1, keepdim=True) / 9,
                torch.sum(self.real_As_lm[:, 42:51, ...], dim=1, keepdim=True) / 9,
                self.real_As_lm[:, [62], ...],
                self.real_As_lm[:, [96], ...],
                self.real_As_lm[:, [66], ...],
                self.real_As_lm[:, [70], ...],
                self.real_As_lm[:, [97], ...],
                self.real_As_lm[:, [74], ...],
                self.real_As_lm[:, [54], ...],
                self.real_As_lm[:, [79], ...],
                self.real_As_lm[:, [85], ...],
                self.real_As_lm[:, [88], ...],
                self.real_As_lm[:, [92], ...],
                torch.sum(self.real_As_lm[:, [90, 94], ...], dim=1, keepdim=True) / 2,
            ], 1)

    def forward_D(self):

        if hasattr(self.netG.synthesis, 'input'):
            self.netG.synthesis.input.transform.copy_(torch.from_numpy(self.m_zero))

        self.real_As_heat = self.netFE_lm.get_heatmap(self.real_As, b_preprocess=False)
        self.real_As_pose = self.netFE_pose(self.real_As_heat, mode=1)
        self.fake_As_pose = self.netFE_pose(self.real_As_pose, mode=2).view(-1, 14, 2)

        if self.bs > 1:
            self.real_As_app = self.netFE(self.real_A_ori, mode=1).unsqueeze(1).repeat(1, self.bs, 1, 1, 1).view(-1, self.c, self.h, self.w)
        else:
            self.real_As_app = self.netFE(self.real_A_ori, mode=1).repeat(self.bs, 1, 1, 1)
        self.fake_As_w = self.netFE(self.real_As_app, self.real_As_pose, mode=2).view(-1, 16, 512)
        self.fake_As = self.netG.synthesis(self.fake_As_w, noise_mode='const')  # G(A)

    def forward(self):

        if hasattr(self.netG.synthesis, 'input'):
            self.netG.synthesis.input.transform.copy_(torch.from_numpy(self.m_zero))

        self.real_As_heat = self.netFE_lm.get_heatmap(self.real_As, b_preprocess=False)
        self.real_As_pose = self.netFE_pose(self.real_As_heat, mode=1)
        self.fake_As_pose = self.netFE_pose(self.real_As_pose, mode=2).view(-1, 14, 2)

        if self.bs > 1:
            self.real_As_app = self.netFE(self.real_A_ori, mode=1).unsqueeze(1).repeat(1, self.bs, 1, 1, 1).view(-1, self.c, self.h, self.w)
        else:
            self.real_As_app = self.netFE(self.real_A_ori, mode=1).repeat(self.bs, 1, 1, 1)
        self.fake_As_w = self.netFE(self.real_As_app, self.real_As_pose, mode=2).view(-1, 16, 512)
        self.fake_As = self.netG.synthesis(self.fake_As_w, noise_mode='const')  # G(A)

    def forward_B(self):

        with torch.no_grad():
            self.real_B_w = self.netG.mapping(self.real_z, self.label)
            self.real_B = self.netG.synthesis(self.real_B_w, noise_mode='const').detach().clamp(-1, 1)
            self.real_B_lm = self.netFE_lm_fix.get_landmark(self.real_B).detach().to(self.device) / 256
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

        self.real_B_heat = self.netFE_lm.get_heatmap(self.real_B, b_preprocess=False)
        self.real_B_pose = self.netFE_pose(self.real_B_heat, mode=1)
        self.fake_B_pose = self.netFE_pose(self.real_B_pose, mode=2).view(-1, 14, 2)

        m = sample_trans()
        self.netG.synthesis.input.transform.copy_(torch.from_numpy(m))
        with torch.no_grad():
            self.real_B2 = self.netG.synthesis(self.real_B_w, noise_mode='const').detach().clamp(-1, 1)

        self.real_B2_app = self.netFE(self.real_B2, mode=1)
        self.fake_B_w = self.netFE(self.real_B2_app, self.real_B_pose, mode=2).view(-1, 16, 512)
        self.netG.synthesis.input.transform.copy_(torch.from_numpy(self.m_zero))
        self.fake_B = self.netG.synthesis(self.fake_B_w, noise_mode='const')  # G(A)
        self.fake_B_app = self.netFE(self.fake_B, mode=1)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        self.D_fake_As = self.netD(self.fake_As.detach())
        self.loss_D_fake = self.opt.lambda_GAN * self.criterionGAN(self.D_fake_As, False)
        # Real
        self.D_real_As = self.netD(self.real_As)
        self.loss_D_real = self.opt.lambda_GAN * self.criterionGAN(self.D_real_As, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""

        # Second, G(A) = B
        self.loss_G_L1 = self.opt.lambda_L1 * self.criterionL1(self.fake_As, self.real_As)
        self.loss_G_VGG = self.opt.lambda_VGG * self.criterionL1(self.vgg16(self.fake_As),
                                                                     self.vgg16(self.real_As).detach())

        self.loss_G_L2 = self.opt.lambda_L2 * self.criterionL2(self.fake_As_pose, self.real_As_key)

        if self.use_gan_loss:
            self.G_fake_As = self.netD(self.fake_As)
            self.loss_G_GAN = self.opt.lambda_GAN * self.criterionGAN(self.G_fake_As, True)
            # combine loss and calculate gradients
            self.loss_G = self.loss_G_L1 + self.loss_G_VGG + self.loss_G_L2 + self.loss_G_GAN
        else:
            # combine loss and calculate gradients
            self.loss_G = self.loss_G_L1 + self.loss_G_VGG + self.loss_G_L2
        self.loss_G.backward()

    def backward_G_B(self):
        self.loss_G_L1_B = self.opt.lambda_L1 * self.criterionL1(self.fake_B, self.real_B)
        self.loss_G_VGG_B = self.opt.lambda_VGG * self.criterionL1(self.vgg16(self.fake_B),
                                                                     self.vgg16(self.real_B).detach())

        self.loss_G_L2_B = self.opt.lambda_L2 * self.criterionL2(self.fake_B_pose, self.real_B_key)

        # self.loss_G_APP_B = self.opt.lambda_APP * self.criterionL1(self.fake_B_app, self.real_B2_app.detach())

        self.loss_G_W = self.opt.lambda_W * self.criterionL1(self.fake_B_w[:,1:,:], self.real_B_w.detach()[:,1:,:])

        # combine loss and calculate gradients
        # self.loss_G_B = self.opt.lambda_B * (self.loss_G_L1_B + self.loss_G_VGG_B + self.loss_G_L2_B + self.loss_G_W + self.loss_G_APP_B)
        self.loss_G_B = self.opt.lambda_B * (self.loss_G_L1_B + self.loss_G_VGG_B + self.loss_G_L2_B + self.loss_G_W)
        self.loss_G_B.backward()

    def optimize_parameters(self):

        self.forward_pre()
        if self.use_gan_loss:
            self.forward_D()                   # compute fake images: G(A)
            # update G
            self.set_requires_grad(self.netD, True)  # D requires no gradients when optimizing G
            self.optimizer_D.zero_grad()     # set D's gradients to zero
            self.backward_D()                # calculate gradients for D
            self.optimizer_D.step()          # update D's weights

        self.forward()                   # compute fake images: G(A)
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_FE.zero_grad()        # set G's gradients to zero
        self.optimizer_FE_pose.zero_grad()  # set G's gradients to zero
        self.optimizer_FE_lm.zero_grad()  # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_FE.step()  # udpate G's weights
        self.optimizer_FE_pose.step()
        self.optimizer_FE_lm.step()

        self.forward_B()                   # compute fake images: G(A)
        # update G
        # self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_FE.zero_grad()        # set G's gradients to zero
        self.optimizer_FE_pose.zero_grad()  # set G's gradients to zero
        self.optimizer_FE_lm.zero_grad()  # set G's gradients to zero
        self.backward_G_B()                   # calculate graidents for G
        self.optimizer_FE.step()  # udpate G's weights
        self.optimizer_FE_pose.step()
        self.optimizer_FE_lm.step()

    def compute_visuals(self):
        self.real_A = self.real_As[[0],...].clone()
        self.real_A2 = self.real_As[[-1], ...].clone()
        self.fake_A = self.fake_As[[0], ...].clone()
        self.fake_A2 = self.fake_As[[-1], ...].clone()

        self.pose_A = draw_points(self.real_As[[0],...], self.fake_As_pose[[0],...])
        self.pose_B = draw_points(self.real_B[[0], ...], self.fake_B_pose[[0], ...])
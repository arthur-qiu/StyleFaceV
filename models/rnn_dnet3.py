import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import functools
import torch.nn.functional as F

import torch.nn.utils.spectral_norm as spectral_norm


class DBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        which_conv=nn.Conv2d,
        wide=True,
        preactivation=False,
        activation=None,
        downsample=None,
    ):
        super(DBlock, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.hidden_channels = self.out_channels if wide else self.in_channels
        self.which_conv = which_conv
        self.preactivation = preactivation
        self.activation = activation
        self.downsample = downsample

        self.conv1 = self.which_conv(in_channels=self.in_channels,
                                     out_channels=self.hidden_channels)
        self.conv1 = spectral_norm(self.conv1)
        self.conv2 = self.which_conv(in_channels=self.hidden_channels,
                                     out_channels=self.out_channels)
        self.conv2 = spectral_norm(self.conv2)

        self.learnable_sc = True if (
            in_channels != out_channels) or downsample else False
        if self.learnable_sc:
            self.conv_sc = nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=1,
                                     padding=0)
            self.conv_sc = spectral_norm(self.conv_sc)

    def shortcut(self, x):
        if self.preactivation:
            if self.learnable_sc:
                x = self.conv_sc(x)
            if self.downsample:
                x = self.downsample(x)
        else:
            if self.downsample:
                x = self.downsample(x)
            if self.learnable_sc:
                x = self.conv_sc(x)
        return x

    def forward(self, x):
        if self.preactivation:
            h = F.relu(x)
        else:
            h = x
        h = self.conv1(h)
        h = self.conv2(self.activation(h))
        if self.downsample:
            h = self.downsample(h)

        return h + self.shortcut(x)


def D_arch(ch=32):
    arch = {}
    arch[256] = {
        'in_channels': [3] + [ch * item for item in [1, 2, 4, 8, 8, 16]],
        'out_channels': [item * ch for item in [1, 2, 4, 8, 8, 16, 16]],
        'downsample': [True] * 6 + [False],
        'resolution': [128, 64, 32, 16, 8, 4, 4],
    }
    arch[128] = {
        'in_channels': [3] + [ch * item for item in [1, 2, 4, 8, 16]],
        'out_channels': [item * ch for item in [1, 2, 4, 8, 16, 16]],
        'downsample': [True] * 5 + [False],
        'resolution': [64, 32, 16, 8, 4, 4],
    }
    arch[64] = {
        'in_channels': [3] + [ch * item for item in [1, 2, 4, 8]],
        'out_channels': [item * ch for item in [1, 2, 4, 8, 16]],
        'downsample': [True] * 4 + [False],
        'resolution': [32, 16, 8, 4, 4],
    }
    arch[32] = {
        'in_channels': [3] + [item * ch for item in [4, 4, 4]],
        'out_channels': [item * ch for item in [4, 4, 4, 4]],
        'downsample': [True, True, False, False],
        'resolution': [16, 16, 16, 16],
    }
    return arch


class BigGAN_Discriminator(nn.Module):
    def __init__(self,
                 D_ch=96,
                 D_wide=True,
                 resolution=128,
                 D_activation=nn.ReLU(inplace=False),
                 output_dim=1,
                 proj_dim=256,
                 D_init='ortho',
                 skip_init=False,
                 D_param='SN'):
        super(BigGAN_Discriminator, self).__init__()
        self.ch = D_ch
        self.D_wide = D_wide
        self.activation = D_activation
        self.init = D_init

        self.arch = D_arch(self.ch)[resolution]

        self.which_conv = functools.partial(nn.Conv2d,
                                            kernel_size=3,
                                            padding=1)

        self.blocks = []
        for index in range(len(self.arch['out_channels'])):
            self.blocks += [[
                DBlock(in_channels=self.arch['in_channels'][index],
                       out_channels=self.arch['out_channels'][index],
                       which_conv=self.which_conv,
                       wide=self.D_wide,
                       activation=self.activation,
                       preactivation=(index > 0),
                       downsample=(nn.AvgPool2d(2) if
                                   self.arch['downsample'][index] else None))
            ]]
        self.blocks = nn.ModuleList(
            [nn.ModuleList(block) for block in self.blocks])

        self.proj0 = spectral_norm(
            nn.Linear(self.arch['out_channels'][-1], proj_dim))
        self.proj1 = spectral_norm(nn.Linear(proj_dim, proj_dim))
        self.proj2 = spectral_norm(nn.Linear(proj_dim, proj_dim))

        self.linear = spectral_norm(nn.Linear(proj_dim, output_dim))

        if not skip_init:
            self.init_weights()

    def init_weights(self):
        self.param_count = 0
        for module in self.modules():
            if (isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)
                    or isinstance(module, nn.Embedding)):
                if self.init == 'ortho':
                    init.orthogonal_(module.weight)
                elif self.init == 'N02':
                    init.normal_(module.weight, 0, 0.02)
                elif self.init in ['glorot', 'xavier']:
                    init.xavier_uniform_(module.weight)
                else:
                    print('Init style not recognized...')
                self.param_count += sum(
                    [p.data.nelement() for p in module.parameters()])
        print('Param count for D'
              's initialized parameters: %d' % self.param_count)

    def forward(self, x, proj_only=False):
        h = x
        for index, blocklist in enumerate(self.blocks):
            for block in blocklist:
                h = block(h)
        h = torch.sum(self.activation(h), [2, 3])
        h = self.activation(self.proj0(h))
        out = self.linear(h)

        proj_head = self.proj2(self.activation(self.proj1(h)))

        if proj_only:
            return proj_head
        return out, proj_head

def pair_cos_sim(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class ModelD_img(nn.Module):
    def __init__(self, moco_m=0.999, video_frame_size=128, l_len=256, q_len=4096, moco_t=0.07, batchSize=1):
        super(ModelD_img, self).__init__()

        self.moco_m = moco_m

        self.modelD = BigGAN_Discriminator(resolution=video_frame_size,
                                             proj_dim=l_len)
        self.modelD_ema = BigGAN_Discriminator(
            resolution=video_frame_size, proj_dim=l_len)

        self.register_buffer("q_real", torch.randn(q_len, l_len))
        self.q_real = F.normalize(self.q_real, dim=0)
        self.register_buffer("q_fake", torch.randn(q_len, l_len))
        self.q_fake = F.normalize(self.q_fake, dim=0)

        self.q_len = q_len // batchSize * batchSize

        self.register_buffer("q_ptr", torch.zeros(1, dtype=torch.long))

        self.T = moco_t
        self.batchSize = batchSize

        for param, param_ema in zip(self.modelD.parameters(),
                                    self.modelD_ema.parameters()):
            param_ema.data.copy_(param.data)
            param_ema.requires_grad = False

    @torch.no_grad()
    def _momentum_update_dis(self):
        """
        Momentum update of the discriminator
        """
        for p, p_ema in zip(self.modelD.parameters(),
                            self.modelD_ema.parameters()):
            p_ema.data = p_ema.data * self.moco_m + p.data * (1. - self.moco_m)

    @torch.no_grad()
    def update_memory_bank(self, logits_real, logits_fake):
        logits_real = concat_all_gather(logits_real)
        logits_fake = concat_all_gather(logits_fake)

        batch_size_t = logits_real.shape[0]

        ptr = int(self.q_ptr)
        self.q_real[ptr:ptr + batch_size_t, :] = logits_real
        self.q_fake[ptr:ptr + batch_size_t, :] = logits_fake
        ptr = (ptr + batch_size_t) % self.q_len
        self.q_ptr[0] = ptr

    def get_cntr_loss_cross_domain(self, logits_real, logits_real2,
                                   logits_fake, logits_fake2):
        T = self.T
        cos_sim_real = pair_cos_sim(
            torch.cat((logits_real, logits_real2), dim=0),
            torch.cat(
                (logits_real, logits_real2, self.q_real[:self.q_len].detach()),
                dim=0))
        m = torch.ones_like(cos_sim_real) / T
        m.fill_diagonal_(0.)

        cos_sim_reg_real = F.softmax(cos_sim_real * m)

        cos_sim_fake = pair_cos_sim(
            torch.cat((logits_fake, logits_fake2), dim=0),
            torch.cat(
                (logits_fake, logits_fake2, self.q_fake[:self.q_len].detach()),
                dim=0))
        cos_sim_reg_fake = F.softmax(cos_sim_fake * m)

        cntr_loss_real = 0.

        for i in range(self.batchSize):
            cntr_loss_real += -torch.log(
                cos_sim_reg_real[i][i + self.batchSize])
            cntr_loss_real += -torch.log(
                cos_sim_reg_real[i + self.batchSize][i])

        cntr_loss_real = cntr_loss_real / (2. * self.batchSize)

        cntr_loss_fake = 0.

        for i in range(self.batchSize):
            cntr_loss_fake += -torch.log(
                cos_sim_reg_fake[i][i + self.batchSize])
            cntr_loss_fake += -torch.log(
                cos_sim_reg_fake[i + self.batchSize][i])

        cntr_loss_fake = cntr_loss_fake / (2. * self.batchSize)

        cntr_loss = cntr_loss_real + cntr_loss_fake
        return cntr_loss

    def forward(self, x, ema=False, proj_only=False):
        if ema:
            return self.modelD_ema(x, proj_only)
        return self.modelD(x, proj_only)



class ModelD_3d(nn.Module):
    def __init__(self, nc=3, n_frames_G=16, num_D = 2):
        super(ModelD_3d, self).__init__()
        norm_D_3d = 'instance'

        nc = nc * 2
        n_frames_G = n_frames_G - 1

        self.netD = MultiscaleDiscriminator_3d(input_nc=nc,
                                            n_frames=n_frames_G,
                                            norm_layer=get_norm_layer_3d(norm_D_3d),
                                            num_D=num_D)
        self.netD.apply(weights_init_3d)


    def forward(self, x):
        return self.netD.forward(x)


def weights_init_3d(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and hasattr(m, 'weight'):
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm3d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer_3d(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm3d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm3d,
                                       affine=False,
                                       track_running_stats=True)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' %
                                  norm_type)
    return norm_layer


class MultiscaleDiscriminator_3d(nn.Module):
    def __init__(self,
                 input_nc,
                 ndf=64,
                 n_layers=3,
                 n_frames=16,
                 norm_layer=nn.InstanceNorm3d,
                 num_D=2,
                 getIntermFeat=True):
        super(MultiscaleDiscriminator_3d, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
        ndf_max = 64

        for i in range(num_D):
            netD = NLayerDiscriminator_3d(
                input_nc, min(ndf_max, ndf * (2**(num_D - 1 - i))), n_layers,
                norm_layer, getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers + 2):
                    setattr(self, 'scale' + str(i) + '_layer' + str(j),
                            getattr(netD, 'model' + str(j)))
            else:
                setattr(self, 'layer' + str(i), netD.model)
        if n_frames > 16:
            self.downsample = nn.AvgPool3d(3,
                                           stride=2,
                                           padding=[1, 1, 1],
                                           count_include_pad=False)
        else:
            self.downsample = nn.AvgPool3d(3,
                                           stride=[1, 2, 2],
                                           padding=[1, 1, 1],
                                           count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [
                    getattr(self,
                            'scale' + str(num_D - 1 - i) + '_layer' + str(j))
                    for j in range(self.n_layers + 2)
                ]
            else:
                model = getattr(self, 'layer' + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result


class NLayerDiscriminator_3d(nn.Module):
    def __init__(self,
                 input_nc,
                 ndf=64,
                 n_layers=3,
                 norm_layer=nn.InstanceNorm3d,
                 getIntermFeat=True):
        super(NLayerDiscriminator_3d, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[
            nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv3d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf),
                nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv3d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[
            nn.Conv3d(nf, 1, kernel_size=kw, stride=1, padding=padw)
        ]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, 'model' + str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)

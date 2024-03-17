
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from torch.nn.utils import spectral_norm
import torch.nn.functional as F

# from medseg.models.custom_layers import DomainSpecificBatchNorm2d


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


class res_convdown(nn.Module):
    '''
    '''

    def __init__(self, in_ch, out_ch, norm=nn.InstanceNorm2d, if_SN=False, bias=True, dropout=None):
        super(res_convdown, self).__init__()
        # down-> conv3->prelu->conv
        if if_SN:
            self.down = spectral_norm(
                nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1, bias=bias))

            self.conv = nn.Sequential(
                spectral_norm(
                    nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=bias)),
                norm(out_ch),
                nn.LeakyReLU(0.2),
                spectral_norm(nn.Conv2d(out_ch, out_ch,
                                        3, padding=1, bias=bias)),
                norm(out_ch),
            )
        else:
            self.down = (nn.Conv2d(in_ch, in_ch, 3,
                                   stride=2, padding=1, bias=bias))

            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=bias),
                norm(out_ch),
                nn.LeakyReLU(0.2),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=bias),
                norm(out_ch),
            )
        if if_SN:
            self.conv_input = spectral_norm(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True))
        else:
            self.conv_input = nn.Conv2d(
                in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True)

        self.last_act = nn.LeakyReLU(0.2)
        self.dropout = dropout
        if not self.dropout is None:
            self.drop = nn.Dropout2d(p=dropout)

    def forward(self, x):
        x = self.down(x)
        res_x = self.last_act(self.conv_input(x) + self.conv(x))
        if not self.dropout is None:
            res_x = self.drop(res_x)
        # appl
        return res_x


# class res_convup(nn.Module):
#     '''
#     upscale
#     '''

#     def __init__(self, in_ch, out_ch, norm=nn.InstanceNorm2d, if_SN=False, bias=True, dropout=None):
#         super(res_convup, self).__init__()
#         # up-> conv3->prelu->conv

#         if if_SN:
#             self.up = nn.Sequential(
#                 spectral_norm(nn.ConvTranspose2d(in_ch, in_ch, 4, padding=1, stride=2), dim=1),

#             )
#             self.conv = nn.Sequential(
#                 spectral_norm(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=bias), dim=1),
#                 norm(out_ch),
#                 nn.LeakyReLU(0.2),
#                 spectral_norm(nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=bias), dim=1),
#                 norm(out_ch),
#             )
#         else:
#             self.up = nn.ConvTranspose2d(in_ch, in_ch, 4, padding=1, stride=2)
#             self.conv = nn.Sequential(
#                 nn.ConvTranspose2d(in_ch, out_ch, 3, padding=1, bias=bias),
#                 norm(out_ch),
#                 nn.LeakyReLU(0.2),
#                 nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=bias),
#                 norm(out_ch),
#             )

#         if if_SN:
#             self.conv_input = spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size=1,
#                                                       stride=1, padding=0, bias=bias), dim=1)
#         else:
#             self.conv_input = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=bias)

#         self.last_act = nn.LeakyReLU(0.2)
#         self.dropout = dropout
#         if not self.dropout is None:
#             self.drop = nn.Dropout2d(p=dropout)

#     def forward(self, x):
#         x = self.up(x)
#         res_x = self.last_act(self.conv_input(x) + self.conv(x))
#         if not self.dropout is None:
#             res_x = self.drop(res_x)
#         # appl
#         return res_x


# class res_convup_2(nn.Module):
#     '''
#     upscale
#     '''

#     def __init__(self, in_ch, out_ch, norm=nn.InstanceNorm2d, if_SN=False, bias=True, dropout=None):
#         super(res_convup_2, self).__init__()
#         # up-> conv3->prelu->conv

#         if if_SN:
#             self.up = nn.Sequential(
#                 spectral_norm(nn.ConvTranspose2d(in_ch, in_ch, 4, padding=1, stride=2), dim=1),

#             )
#             self.conv = nn.Sequential(
#                 spectral_norm(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=bias), dim=1),
#                 norm(out_ch),
#                 nn.LeakyReLU(0.2),
#                 spectral_norm(nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=bias), dim=1),
#                 norm(out_ch),
#             )
#         else:
#             self.up = nn.ConvTranspose2d(in_ch, in_ch, 4, padding=1, stride=2)
#             self.conv = nn.Sequential(
#                 nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=bias),
#                 norm(out_ch),
#                 nn.LeakyReLU(0.2),
#                 nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=bias),
#                 norm(out_ch),
#             )

#         if if_SN:
#             self.conv_input = spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size=1,
#                                                       stride=1, padding=0, bias=bias), dim=1)
#         else:
#             self.conv_input = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=bias)

#         self.last_act = nn.LeakyReLU(0.2)
#         self.dropout = dropout
#         if not self.dropout is None:
#             self.drop = nn.Dropout2d(p=dropout)

#     def forward(self, x):
#         x = self.up(x)
#         res_x = self.last_act(self.conv_input(x) + self.conv(x))
#         if not self.dropout is None:
#             res_x = self.drop(res_x)
#         # appl
#         return res_x


# class res_bilinear_up(nn.Module):
#     '''
#     upscale
#     '''

#     def __init__(self, in_ch, out_ch, norm=nn.InstanceNorm2d, if_SN=False, bias=True, dropout=None):
#         super(res_bilinear_up, self).__init__()
#         # up-> conv3->prelu->conv

#         if if_SN:
#             self.up = nn.Sequential(
#                 nn.UpsamplingBilinear2d(scale_factor=2),
#                 nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0, bias=bias)
#             )

#             self.conv = nn.Sequential(
#                 spectral_norm(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=bias), dim=1),
#                 norm(out_ch),
#                 nn.LeakyReLU(0.2),
#                 spectral_norm(nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=bias), dim=1),
#                 norm(out_ch),
#             )
#         else:
#             self.up = nn.Sequential(
#                 nn.UpsamplingBilinear2d(scale_factor=2),
#                 nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, bias=bias)
#             )

#             self.conv = nn.Sequential(
#                 nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=bias),
#                 norm(out_ch),
#                 nn.LeakyReLU(0.2),
#                 nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=bias),
#                 norm(out_ch),
#             )

#         if if_SN:
#             self.conv_input = spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size=1,
#                                                       stride=1, padding=0, bias=bias), dim=1)
#         else:
#             self.conv_input = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=bias)

#         self.last_act = nn.LeakyReLU(0.2)
#         self.dropout = dropout
#         if not self.dropout is None:
#             self.drop = nn.Dropout2d(p=dropout)

#     def forward(self, x):
#         x = self.up(x)
#         res_x = self.last_act(self.conv_input(x) + self.conv(x))
#         if not self.dropout is None:
#             res_x = self.drop(res_x)
#         # appl
#         return res_x


class res_NN_up(nn.Module):
    '''
    upscale with NN upsampling followed by conv
    '''

    def __init__(self, in_ch, out_ch, norm=nn.InstanceNorm2d, if_SN=False, bias=True, dropout=None):
        super(res_NN_up, self).__init__()
        # up-> conv3->prelu->conv
        self.up = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_ch, in_ch, kernel_size=3,
                      stride=1, padding=1, bias=True)
        )

        if if_SN:

            self.conv = nn.Sequential(
                spectral_norm(nn.Conv2d(in_ch, out_ch, 3,
                                        padding=1, bias=bias), dim=1),
                norm(out_ch),
                nn.LeakyReLU(0.2),
                spectral_norm(nn.Conv2d(out_ch, out_ch, 3,
                                        padding=1, bias=bias), dim=1),
                norm(out_ch),
            )
        else:

            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=bias),
                norm(out_ch),
                nn.LeakyReLU(0.2),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=bias),
                norm(out_ch),
            )

        if if_SN:
            self.conv_input = spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size=1,
                                                      stride=1, padding=0, bias=True), dim=1)
        else:
            self.conv_input = nn.Conv2d(
                in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True)

        self.last_act = nn.LeakyReLU(0.2)
        self.dropout = dropout
        if not self.dropout is None:
            self.drop = nn.Dropout2d(p=dropout)

    def forward(self, x):
        x = self.up(x)
        res_x = self.last_act(self.conv_input(x) + self.conv(x))
        if not self.dropout is None:
            res_x = self.drop(res_x)
        # appl
        return res_x


class res_up_family(nn.Module):
    '''
    upscale with different upsampling methods
    '''

    def __init__(self, in_ch, out_ch, norm=nn.InstanceNorm2d, if_SN=False, bias=True, dropout=None, up_type='bilinear'):
        super(res_up_family, self).__init__()
        # up-> conv3->prelu->conv
        if up_type == 'NN':
            self.up = nn.Sequential(
                nn.UpsamplingNearest2d(scale_factor=2),
            )
        elif up_type == 'bilinear':
            self.up = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
            )
        elif up_type == 'Conv2':
            self.up = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
        elif up_type == 'Conv4':
            self.up = nn.ConvTranspose2d(
                in_ch, in_ch, kernel_size=4, stride=2, padding=1)
        else:
            raise NotImplementedError

        if if_SN:

            self.conv = nn.Sequential(
                spectral_norm(nn.Conv2d(in_ch, out_ch, 3,
                                        padding=1, bias=bias), dim=1),
                norm(out_ch),
                nn.LeakyReLU(0.2),
                spectral_norm(nn.Conv2d(out_ch, out_ch, 3,
                                        padding=1, bias=bias), dim=1),
                norm(out_ch),
            )
        else:

            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=bias),
                norm(out_ch),
                nn.LeakyReLU(0.2),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=bias),
                norm(out_ch),
            )

        if if_SN:
            self.conv_input = spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size=1,
                                                      stride=1, padding=0, bias=True), dim=1)
        else:
            self.conv_input = nn.Conv2d(
                in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True)

        self.last_act = nn.LeakyReLU(0.2)
        self.dropout = dropout
        if not self.dropout is None:
            self.drop = nn.Dropout2d(p=dropout)

    def forward(self, x):
        x = self.up(x)
        res_x = self.last_act(self.conv_input(x) + self.conv(x))
        if not self.dropout is None:
            res_x = self.drop(res_x)
        # appl
        return res_x


class MyEncoder(nn.Module):
    '''
    Naive Encoder
    '''

    def __init__(self, input_channel, output_channel=None, feature_reduce=1, encoder_dropout=None, norm=nn.InstanceNorm2d, if_SN=False, act=torch.nn.Sigmoid()):
        super(MyEncoder, self).__init__()

        if if_SN:
            self.inc = nn.Sequential(
                spectral_norm(nn.Conv2d(input_channel, 64 //
                                        feature_reduce, 3, padding=1, bias=True)),
                norm(64 // feature_reduce),
                nn.LeakyReLU(0.2),
                nn.Conv2d(64 // feature_reduce, 64 //
                          feature_reduce, 3, padding=1, bias=True),
                norm(64 // feature_reduce),
            )
        else:
            self.inc = nn.Sequential(
                nn.Conv2d(input_channel, 64 // feature_reduce,
                          3, padding=1, bias=True),
                norm(64 // feature_reduce),
                nn.LeakyReLU(0.2),
                nn.Conv2d(64 // feature_reduce, 64 //
                          feature_reduce, 3, padding=1, bias=True),
                norm(64 // feature_reduce),
            )

        self.down1 = res_convdown(64 // feature_reduce, 128 // feature_reduce,
                                  norm=norm, if_SN=if_SN, dropout=encoder_dropout)
        self.down2 = res_convdown(128 // feature_reduce, 264 // feature_reduce,
                                  norm=norm, if_SN=if_SN, dropout=encoder_dropout)
        self.down3 = res_convdown(264 // feature_reduce, 512 // feature_reduce,
                                  norm=norm, if_SN=if_SN, dropout=encoder_dropout)
        self.down4 = res_convdown(512 // feature_reduce, 512 // feature_reduce,
                                  norm=norm, if_SN=if_SN, dropout=encoder_dropout)
        if output_channel is None:
            self.final_conv = nn.Sequential(
                nn.Conv2d(512 // feature_reduce, 512 // feature_reduce,
                          kernel_size=1, stride=1, padding=0),
                norm(512 // feature_reduce))
        else:
            self.final_conv = nn.Sequential(
                nn.Conv2d(512 // feature_reduce, 512 // feature_reduce,
                          kernel_size=1, stride=1, padding=0),
                norm(512 // feature_reduce))

        self.act = act
        for m in self._modules:
            normal_init(self._modules[m], 0, 0.02)

    def forward(self, x):
        x1 = self.inc(x)
        x1 = F.leaky_relu(x1, negative_slope=0.2)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x5 = self.final_conv(x5)
        if self.act is not None:
            x5 = self.act(x5)

        return x5


class MyDecoder(nn.Module):
    '''

    '''

    def __init__(self, input_channel, output_channel, feature_reduce=1, decoder_dropout=None, norm=nn.InstanceNorm2d, up_type='bilinear', if_SN=False, last_act=None):
        super(MyDecoder, self).__init__()
        self.up1 = res_up_family(input_channel, 264 // feature_reduce, norm=norm,
                                 up_type=up_type, dropout=decoder_dropout, if_SN=if_SN)
        self.up2 = res_up_family(264 // feature_reduce, 128 // feature_reduce, norm=norm,
                                 up_type=up_type, dropout=decoder_dropout, if_SN=if_SN)
        self.up3 = res_up_family(128 // feature_reduce, 64 // feature_reduce, norm=norm,
                                 up_type=up_type, dropout=decoder_dropout, if_SN=if_SN)
        self.up4 = res_up_family(64 // feature_reduce, 64 // feature_reduce, norm=norm,
                                 up_type=up_type, dropout=decoder_dropout, if_SN=if_SN)

        # final conv
        if if_SN:
            self.final_conv = spectral_norm(
                nn.Conv2d(64 // feature_reduce, output_channel, kernel_size=1, stride=1, padding=0))
        else:
            self.final_conv = nn.Conv2d(
                64 // feature_reduce, output_channel, kernel_size=1, stride=1, padding=0)
        self.last_act = last_act
        for m in self._modules:
            normal_init(self._modules[m], 0, 0.02)

    def forward(self, x):
        x2 = self.up1(x)
        x3 = self.up2(x2)
        x4 = self.up3(x3)
        x5 = self.up4(x4)
        x5 = self.final_conv(x5)
        if self.last_act is not None:
            x5 = self.last_act(x5)
        return x5


class Dual_Branch_Encoder(nn.Module):
    '''
    FTN's encoder, produces two latent codes, z_i and z_s
    '''

    def __init__(self, input_channel, z_level_1_channel=None, z_level_2_channel=None, feature_reduce=1, encoder_dropout=None, norm=nn.InstanceNorm2d, if_SN=False):
        super(Dual_Branch_Encoder, self).__init__()

        self.general_encoder = MyEncoder(input_channel, output_channel=z_level_1_channel,
                                         feature_reduce=feature_reduce, encoder_dropout=encoder_dropout, norm=norm, if_SN=if_SN, act=torch.nn.ReLU())

        if not if_SN:
            self.code_decoupler = nn.Sequential(
                nn.Conv2d(z_level_1_channel, z_level_2_channel,
                          3, padding=1, bias=True),
                norm(z_level_2_channel),
                nn.LeakyReLU(0.2),
                nn.Conv2d(z_level_2_channel, z_level_2_channel,
                          3, padding=1, bias=True),
                norm(z_level_2_channel),
                nn.ReLU(),


            )
        else:
            self.code_decoupler = nn.Sequential(
                spectral_norm(nn.Conv2d(z_level_1_channel,
                                        z_level_2_channel, 3, padding=1, bias=True)),
                norm(z_level_2_channel),
                nn.LeakyReLU(0.2),
                spectral_norm(nn.Conv2d(z_level_2_channel,
                                        z_level_2_channel, 3, padding=1, bias=True)),
                norm(z_level_2_channel),
                nn.ReLU(),

            )

        for m in self._modules:
            normal_init(self._modules[m], 0, 0.02)

    def filter_code(self, z):
        z_s = self.code_decoupler(z)
        return z_s

    def forward(self, x):
        z_i = self.general_encoder(x)
        z_s = self.filter_code(z_i)
        return z_i, z_s


class ds_res_convdown(nn.Module):
    '''
    res conv down with domain specific layers
    '''

    def __init__(self, in_ch, out_ch, num_domains=2, if_SN=False, bias=True, dropout=None):
        super(ds_res_convdown, self).__init__()
        # down-> conv3->prelu->conv
        if if_SN:
            self.down = spectral_norm(
                nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1, bias=True))

            self.conv_1 = spectral_norm(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=bias))
            self.norm_1 = DomainSpecificBatchNorm2d(
                out_ch, num_domains=num_domains)
            self.act_1 = nn.LeakyReLU(0.2)
            self.conv_2 = spectral_norm(nn.Conv2d(out_ch, out_ch,
                                                  3, padding=1, bias=bias))

            self.norm_2 = DomainSpecificBatchNorm2d(
                out_ch, num_domains=num_domains)
        else:
            self.down = nn.Conv2d(
                in_ch, in_ch, 3, stride=2, padding=1, bias=True)

            self.conv_1 = spectral_norm(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=bias))
            self.norm_1 = DomainSpecificBatchNorm2d(
                out_ch, num_domains=num_domains)
            self.act_1 = nn.LeakyReLU(0.2)
            self.conv_2 = nn.Conv2d(out_ch, out_ch,
                                    3, padding=1, bias=bias)

            self.norm_2 = DomainSpecificBatchNorm2d(
                out_ch, num_domains=num_domains)

        if if_SN:
            self.conv_input = spectral_norm(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True))
        else:
            self.conv_input = nn.Conv2d(
                in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True)

        self.last_act = nn.LeakyReLU(0.2)
        self.dropout = dropout
        if not self.dropout is None:
            self.drop = nn.Dropout2d(p=dropout)

    def forward(self, x, domain_id=0):
        x = self.down(x)
        f = self.conv_1(x)
        f = self.norm_1(f, domain_id)
        f = self.act_1(f)
        f = self.conv_2(f)
        f = self.norm_2(f, domain_id)
        res_x = self.last_act(self.conv_input(x) + f)
        if not self.dropout is None:
            res_x = self.drop(res_x)
        return res_x


class DomainSpecificEncoder(nn.Module):
    '''
    Encoder with domain specific batch normalization layers.
    '''

    def __init__(self, input_channel, output_channel=None, num_domains=2, feature_reduce=1, encoder_dropout=None, if_SN=False, act=torch.nn.Sigmoid()):
        super(DomainSpecificEncoder, self).__init__()

        if if_SN:
            self.inc_conv_1 = spectral_norm(nn.Conv2d(input_channel, 64 //
                                                      feature_reduce, 3, padding=1, bias=True))
            self.norm_1 = DomainSpecificBatchNorm2d(
                64 // feature_reduce, num_domains)
            self.act_1 = nn.LeakyReLU(0.2)
            self.inc_conv_2 = spectral_norm(nn.Conv2d(64 // feature_reduce, 64 // feature_reduce,
                                                      3, padding=1, bias=True))
            self.norm_2 = DomainSpecificBatchNorm2d(
                64 // feature_reduce, num_domains)

        else:
            self.inc_conv_1 = nn.Conv2d(input_channel, 64 //
                                        feature_reduce, 3, padding=1, bias=True)
            self.norm_1 = DomainSpecificBatchNorm2d(
                64 // feature_reduce, num_domains)
            self.act_1 = nn.LeakyReLU(0.2)
            self.inc_conv_2 = nn.Conv2d(64 // feature_reduce, 64 // feature_reduce,
                                        3, padding=1, bias=True)
            self.norm_2 = DomainSpecificBatchNorm2d(
                64 // feature_reduce, num_domains)

        self.down1 = ds_res_convdown(
            64 // feature_reduce, 128 // feature_reduce, num_domains=num_domains, if_SN=if_SN, dropout=encoder_dropout)
        self.down2 = ds_res_convdown(
            128 // feature_reduce, 264 // feature_reduce, num_domains=num_domains, if_SN=if_SN, dropout=encoder_dropout)
        self.down3 = ds_res_convdown(
            264 // feature_reduce, 512 // feature_reduce, num_domains=num_domains, if_SN=if_SN, dropout=encoder_dropout)
        self.down4 = ds_res_convdown(
            512 // feature_reduce, 512 // feature_reduce, num_domains=num_domains, if_SN=if_SN, dropout=encoder_dropout)

        if output_channel is None:
            self.final_conv = nn.Conv2d(512 // feature_reduce, 512 // feature_reduce,
                                        kernel_size=1, stride=1, padding=0)
            self.final_norm = DomainSpecificBatchNorm2d(
                512 // feature_reduce, num_domains=num_domains)
        else:
            self.final_conv = nn.Conv2d(512 // feature_reduce, output_channel,
                                        kernel_size=1, stride=1, padding=0)
            self.final_norm = DomainSpecificBatchNorm2d(
                512 // feature_reduce, num_domains=num_domains)

        # apply sigmoid activation
        self.act = act  # torch.nn.LeakyReLU(0.2)

        for m in self._modules:
            normal_init(self._modules[m], 0, 0.02)

    def forward(self, x, domain_id=0):
        x1 = self.inc_conv_1(x)
        x1 = self.norm_1(x1, domain_id)
        x1 = self.act_1(x1)
        x1 = self.inc_conv_2(x1)
        x1 = self.norm_2(x1, domain_id)

        x1 = F.leaky_relu(x1, negative_slope=0.2)
        x2 = self.down1(x1, domain_id)
        x3 = self.down2(x2, domain_id)
        x4 = self.down3(x3, domain_id)
        x5 = self.down4(x4, domain_id)

        x5 = self.final_conv(x5)
        x5 = self.final_norm(x5, domain_id)

        if self.act is not None:
            x5 = self.act(x5)

        return x5


if __name__ == '__main__':

    encoder = Dual_Branch_Encoder(input_channel=1, z_level_1_channel=512 // 4,
                                  z_level_2_channel=512 // 8, feature_reduce=4, if_SN=True, encoder_dropout=None)
    # decoder= Decoder(input_channel=512//4,output_channel=4,feature_reduce=4,if_SN=True,decoder_dropout=None)

    encoder.train()
    image = torch.autograd.Variable(torch.randn(2, 1, 192, 192))
    z1, z2 = encoder(image)

    decoder_1 = MyDecoder(input_channel=128, output_channel=1, feature_reduce=4,
                          decoder_dropout=None, norm=nn.InstanceNorm2d, if_SN=False)
    decoder_2 = MyDecoder(input_channel=z2.size(1), output_channel=1, feature_reduce=8,
                          decoder_dropout=None, norm=nn.InstanceNorm2d, if_SN=False)

    print(z1.size(), z2.size())

    decoder_1.train()
    result = decoder_1(z1)
    print(result.size())

    decoder_2.train()
    result = decoder_2(z2)
    print(result.size())
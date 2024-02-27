import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor

import warnings
from compressai.ans import BufferedRansEncoder, RansDecoder

# pylint: disable=E0611,E0401
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN, MaskedConv2d, last_STB, ResBottleneck

from .utils import conv, deconv, update_registered_buffers
from .priors import CompressionModel
from .gain_utils import get_scale_table, ResBlock, NonLocalAttention, SFT, UpConv2d
from .google import MeanScaleHyperprior
# pylint: enable=E0611,E0401



class GainedScaleHyperprior(CompressionModel):
    '''
    Bottleneck scaling version.
    '''
    def __init__(self, N = 192, M = 320, **kwargs):
        super().__init__(entropy_bottleneck_channels=N, **kwargs)

        self.g_a = nn.Sequential(
            conv(3, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 3),
        )

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
            conv(N, N),
            nn.ReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, N),
            nn.ReLU(inplace=True),
            deconv(N, N),
            nn.ReLU(inplace=True),
            conv(N, M, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
        )

        self.gaussian_conditional = GaussianConditional(None)

        self.lmbda = [0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.001, 0.0003]  # mxh add from HUAWEI CVPR2021 Gained...

        # Condition on Latent y, so the gain vector length M
        # e.g.: self.levels = 6 means we have 6 pairs gain vectors corresponding to 6 level RD performance
        # treat all channels the same in initialization
        self.levels = len(self.lmbda) # 8
        self.Gain = torch.nn.Parameter(torch.ones(size=[self.levels, M]), requires_grad=True)
        self.InverseGain = torch.nn.Parameter(torch.ones(size=[self.levels, M]), requires_grad=True)
        self.HyperGain = torch.nn.Parameter(torch.ones(size=[self.levels, N]), requires_grad=True)
        self.InverseHyperGain = torch.nn.Parameter(torch.ones(size=[self.levels, N]), requires_grad=True)


    def forward(self, x, s):
        '''
            x: input image
            s: random num to choose gain vector
        '''
        y = self.g_a(x)
        y = y * torch.abs(self.Gain[s]).unsqueeze(0).unsqueeze(2).unsqueeze(3) # Gain[s]: [M]  -->  [1,M,1,1]
        z = self.h_a(y)
        z = z * torch.abs(self.HyperGain[s]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        z_hat = z_hat * torch.abs(self.InverseHyperGain[s]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        scales_hat = self.h_s(z_hat)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        y_hat = y_hat * torch.abs(self.InverseGain[s]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        x_hat = self.g_s(y_hat)

        return {
            "y": y,
            "y_hat": y_hat,
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x, s, l):
        assert s in range(0,self.levels-1), f"s should in range(0, {self.levels-1}), but get s:{s}"
        assert l >=0 and l <=1, "l should in [0,1]"
        # l = 0, Interpolated = s+1; l = 1, Interpolated = s
        InterpolatedGain = torch.abs(self.Gain[s]).pow(1-l) * torch.abs(self.Gain[s+1]).pow(l)
        # InterpolatedInverseGain = self.InverseGain[s].pow(l) * self.InverseGain[s+1].pow(1-l)
        InterpolatedHyperGain = torch.abs(self.HyperGain[s]).pow(1-l) * torch.abs(self.HyperGain[s+1]).pow(l)
        InterpolatedInverseHyperGain = torch.abs(self.InverseHyperGain[s]).pow(1-l) * torch.abs(self.InverseHyperGain[s+1]).pow(l)

        # Linear Interpolation can achieve the same result?
        # InterpolatedGain = torch.abs(self.Gain[s]) * (1 - l) + torch.abs(self.Gain[s + 1]) * l
        # InterpolatedHyperGain = torch.abs(self.HyperGain[s]) * (1 - l) + torch.abs(self.HyperGain[s + 1]) * l
        # InterpolatedInverseHyperGain = torch.abs(self.InverseHyperGain[s]) * (1 - l) + torch.abs(self.InverseHyperGain[s + 1]) * (l)

        y = self.g_a(x)
        ungained_y = y
        y = y * InterpolatedGain.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        z = self.h_a(y)
        z = z * InterpolatedHyperGain.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        z_hat = z_hat * InterpolatedInverseHyperGain.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes)  # y_string 是 list, 且只包含一个元素
        gained_y_hat = self.gaussian_conditional.quantize(y, "symbols")
        return {"strings": [y_strings, z_strings],
                "shape": z.size()[-2:],
                "ungained_y": ungained_y,
                "gained_y": y,
                "gained_y_hat": gained_y_hat}


    def decompress(self, strings, shape, s, l):
        assert isinstance(strings, list) and len(strings) == 2 # 保证有y和z
        assert s in range(0, self.levels - 1), f"s should in range(0,{self.levels - 1})"
        assert l >= 0 and l <= 1, "l should in [0,1]"
        # InterpolatedGain = self.Gain[s].pow(l) * self.Gain[s + 1].pow(1 - l)
        InterpolatedInverseGain = torch.abs(self.InverseGain[s]).pow(1-l) * torch.abs(self.InverseGain[s+1]).pow(l)
        # InterpolatedHyperGain = self.HyperGain[s].pow(l) * self.HyperGain[s + 1].pow(1 - l)
        InterpolatedInverseHyperGain = torch.abs(self.InverseHyperGain[s]).pow(1-l) * torch.abs(self.InverseHyperGain[s+1]).pow(l)

        # Linear Interpolation can achieve the same result
        InterpolatedInverseGain = torch.abs(self.InverseGain[s]) * (1 - l) + torch.abs(self.InverseGain[s + 1]) * (l)
        InterpolatedInverseHyperGain = torch.abs(self.InverseHyperGain[s])* (1 - l) + torch.abs(self.InverseHyperGain[s + 1]) * (l)

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        z_hat = z_hat * InterpolatedInverseHyperGain.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes
        )
        gained_y_hat = y_hat
        y_hat = y_hat * InterpolatedInverseGain.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat, "gained_y_hat": gained_y_hat}

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def getY(self, x, s, l):
        assert s in range(0, self.levels - 1), f"s should in range(0, {self.levels - 1}), but get s:{s}"
        assert l >= 0 and l <= 1, "l should in [0,1]"
        # l = 0, Interpolated = s+1; l = 1, Interpolated = s
        InterpolatedGain = torch.abs(self.Gain[s]).pow(1 - l) * torch.abs(self.Gain[s + 1]).pow(l)

        # 如果x不是64的倍数，就对x做padding
        h, w = x.size(2), x.size(3)
        p = 64  # maximum 6 strides of 2
        new_h = (h + p - 1) // p * p  # padding为64的倍数
        new_w = (w + p - 1) // p * p
        padding_left = (new_w - w) // 2
        padding_right = new_w - w - padding_left
        padding_top = (new_h - h) // 2
        padding_bottom = new_h - h - padding_top
        x_padded = F.pad(
            x,
            (padding_left, padding_right, padding_top, padding_bottom),
            mode="constant",
            value=0,
        )

        y = self.g_a(x_padded)
        ungained_y = y
        y = y * InterpolatedGain.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        y_quantized = self.gaussian_conditional.quantize(y, "noise")

        return {"ungained_y": ungained_y,
                "gained_y": y,
                "gained_y_hat": y_quantized}

        # return y, y_quantized

    def getScale(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        gaussian_params = self.h_s(z)
        scales, means = gaussian_params.chunk(2, 1)
        return scales

    def getX(self, y_hat, s, l):
        assert s in range(0, self.levels - 1), f"s should in range(0,{self.levels - 1})"
        assert l >= 0 and l <= 1, "l should in [0,1]"
        InterpolatedInverseGain = torch.abs(self.InverseGain[s]).pow(1 - l) * torch.abs(self.InverseGain[s + 1]).pow(l)
        InterpolatedInverseHyperGain = torch.abs(self.InverseHyperGain[s]).pow(1 - l) * torch.abs(
            self.InverseHyperGain[s + 1]).pow(l)

        y_hat = y_hat * InterpolatedInverseGain.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}


class GainedMSHyperprior(CompressionModel):
    '''
    Bottleneck scaling version.
    '''
    def __init__(self, N = 128, M = 192, **kwargs):
        super().__init__(entropy_bottleneck_channels=N, **kwargs)

        self.g_a = nn.Sequential(
            conv(3, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 3),
        )

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            # conv(N, N, stride=1, kernel_size=3),
            conv(N, N),
            nn.LeakyReLU(inplace=True),
            # conv(N, N, stride=1, kernel_size=3),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            # conv(N, N, stride=1, kernel_size=3),
            deconv(N, M),
            nn.LeakyReLU(inplace=True),
            # conv(M, M, stride=1, kernel_size=3),
            deconv(M, M * 3 // 2),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )

        self.gaussian_conditional = GaussianConditional(None)

        self.lmbda = [0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.001, 0.0003]  # mxh add from HUAWEI CVPR2021 Gained...

        # Condition on Latent y, so the gain vector length M
        # e.g.: self.levels = 6 means we have 6 pairs gain vectors corresponding to 6 level RD performance
        # treat all channels the same in initialization
        # 创建了一个形状为 [self.levels, M] 的张量，并将其包装在 torch.nn.Parameter 类中，表示它是一个模型参数，需要进行训练。
        # 参数 requires_grad=True 表示需要计算该参数的梯度，并在反向传播时更新该参数的值。
        # M:通道数
        self.levels = len(self.lmbda) # 8
        self.Gain = torch.nn.Parameter(torch.ones(size=[self.levels, M]), requires_grad=True)
        self.InverseGain = torch.nn.Parameter(torch.ones(size=[self.levels, M]), requires_grad=True)
        self.HyperGain = torch.nn.Parameter(torch.ones(size=[self.levels, N]), requires_grad=True)
        self.InverseHyperGain = torch.nn.Parameter(torch.ones(size=[self.levels, N]), requires_grad=True)


    def forward(self, x, s):
        '''
            x: input image
            s: random num to choose gain vector
        '''
        # ga->Gain->ha->HyperGain->InverseHyperGain->hs->gaussian_params->InverseGain->g_s
        y = self.g_a(x)
        y = y * torch.abs(self.Gain[s]).unsqueeze(0).unsqueeze(2).unsqueeze(3) # Gain[s]: [M]  -->  [1,M,1,1]
        # y_hat = self.gaussian_conditional.quantize(y, "noise")
        z = self.h_a(y)
        z = z * torch.abs(self.HyperGain[s]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        z_hat = z_hat * torch.abs(self.InverseHyperGain[s]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        y_hat = y_hat * torch.abs(self.InverseGain[s]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        x_hat = self.g_s(y_hat)

        return {
            "y": y,
            "y_hat": y_hat,
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x, s, l):
        assert s in range(0,self.levels-1), f"s should in range(0, {self.levels-1}), but get s:{s}"
        assert l >=0 and l <=1, "l should in [0,1]"
        # l = 0, Interpolated = s+1; l = 1, Interpolated = s
        # 计算l实现连续码率可变
        InterpolatedGain = torch.abs(self.Gain[s]).pow(1-l) * torch.abs(self.Gain[s+1]).pow(l)
        # InterpolatedInverseGain = self.InverseGain[s].pow(l) * self.InverseGain[s+1].pow(1-l)
        InterpolatedHyperGain = torch.abs(self.HyperGain[s]).pow(1-l) * torch.abs(self.HyperGain[s+1]).pow(l)
        InterpolatedInverseHyperGain = torch.abs(self.InverseHyperGain[s]).pow(1-l) * torch.abs(self.InverseHyperGain[s+1]).pow(l)

        # Linear Interpolation can achieve the same result?
        InterpolatedGain = torch.abs(self.Gain[s]) * (1 - l) + torch.abs(self.Gain[s + 1]) * l
        InterpolatedHyperGain = torch.abs(self.HyperGain[s]) * (1 - l) + torch.abs(self.HyperGain[s + 1]) * l
        InterpolatedInverseHyperGain = torch.abs(self.InverseHyperGain[s]) * (1 - l) + torch.abs(self.InverseHyperGain[s + 1]) * (l)

        y = self.g_a(x)
        ungained_y = y
        y = y * InterpolatedGain.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        # y_hat = self.gaussian_conditional.quantize(y, "symbols").type(torch.cuda.FloatTensor)
        z = self.h_a(y)
        z = z * InterpolatedHyperGain.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        z_hat = z_hat * InterpolatedInverseHyperGain.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)  # y_string 是 list, 且只包含一个元素
        gained_y_hat = self.gaussian_conditional.quantize(y, "symbols", means_hat)
        return {"strings": [y_strings, z_strings],
                "shape": z.size()[-2:],
                "ungained_y": ungained_y,
                "gained_y": y,
                "gained_y_hat": gained_y_hat}

    def decompress(self, strings, shape, s, l):
        assert isinstance(strings, list) and len(strings) == 2 # 保证有y和z
        assert s in range(0, self.levels - 1), f"s should in range(0,{self.levels - 1})"
        assert l >= 0 and l <= 1, "l should in [0,1]"
        # InterpolatedGain = self.Gain[s].pow(l) * self.Gain[s + 1].pow(1 - l)
        InterpolatedInverseGain = torch.abs(self.InverseGain[s]).pow(1-l) * torch.abs(self.InverseGain[s+1]).pow(l)
        # InterpolatedHyperGain = self.HyperGain[s].pow(l) * self.HyperGain[s + 1].pow(1 - l)
        InterpolatedInverseHyperGain = torch.abs(self.InverseHyperGain[s]).pow(1-l) * torch.abs(self.InverseHyperGain[s+1]).pow(l)

        # Linear Interpolation can achieve the same result
        InterpolatedInverseGain = torch.abs(self.InverseGain[s]) * (1 - l) + torch.abs(self.InverseGain[s + 1]) * (l)
        InterpolatedInverseHyperGain = torch.abs(self.InverseHyperGain[s])* (1 - l) + torch.abs(self.InverseHyperGain[s + 1]) * (l)

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        z_hat = z_hat * InterpolatedInverseHyperGain.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )
        gained_y_hat = y_hat
        y_hat = y_hat * InterpolatedInverseGain.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat, "gained_y_hat": gained_y_hat}

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def getY(self, x, s, l):
        assert s in range(0, self.levels - 1), f"s should in range(0, {self.levels - 1}), but get s:{s}"
        assert l >= 0 and l <= 1, "l should in [0,1]"
        # l = 0, Interpolated = s+1; l = 1, Interpolated = s
        InterpolatedGain = torch.abs(self.Gain[s]).pow(1 - l) * torch.abs(self.Gain[s + 1]).pow(l)

        # 如果x不是64的倍数，就对x做padding
        h, w = x.size(2), x.size(3)
        p = 64  # maximum 6 strides of 2
        new_h = (h + p - 1) // p * p  # padding为64的倍数
        new_w = (w + p - 1) // p * p
        padding_left = (new_w - w) // 2
        padding_right = new_w - w - padding_left
        padding_top = (new_h - h) // 2
        padding_bottom = new_h - h - padding_top
        x_padded = F.pad(
            x,
            (padding_left, padding_right, padding_top, padding_bottom),
            mode="constant",
            value=0,
        )

        y = self.g_a(x_padded)
        ungained_y = y
        y = y * InterpolatedGain.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        y_quantized = self.gaussian_conditional.quantize(y, "noise")

        return {"ungained_y": ungained_y,
                "gained_y": y,
                "gained_y_hat": y_quantized}

        # return y, y_quantized

    def getScale(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        gaussian_params = self.h_s(z)
        scales, means = gaussian_params.chunk(2, 1)
        return scales

    def getX(self, y_hat, s, l):
        assert s in range(0, self.levels - 1), f"s should in range(0,{self.levels - 1})"
        assert l >= 0 and l <= 1, "l should in [0,1]"
        InterpolatedInverseGain = torch.abs(self.InverseGain[s]).pow(1 - l) * torch.abs(self.InverseGain[s + 1]).pow(l)
        InterpolatedInverseHyperGain = torch.abs(self.InverseHyperGain[s]).pow(1 - l) * torch.abs(
            self.InverseHyperGain[s + 1]).pow(l)

        y_hat = y_hat * InterpolatedInverseGain.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}


class SCGainedMSHyperprior(CompressionModel):
    '''
    Bottleneck scaling version.
    modulate in both spatial and channel dim
    '''
    def __init__(self, N = 128, M = 192, **kwargs):
        super().__init__(entropy_bottleneck_channels=N, **kwargs)

        self.qmap_feature_ga0 = nn.Sequential(
            conv(4, N * 2, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(N * 2, N, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(N, N, 3, 1)
        )
        self.qmap_feature_ga1 = nn.Sequential(
            conv(N, N, 3),
            nn.LeakyReLU(0.1, True),
            conv(N, N, 1, 1)
        )
        self.ga_SFT1 = SFT(N, N, ks=3)
        self.qmap_feature_ga2 = nn.Sequential(
            conv(N, N, 3),
            nn.LeakyReLU(0.1, True),
            conv(N, N, 1, 1)
        )
        self.ga_SFT2 = SFT(N, N, ks=3)
        self.qmap_feature_ga3 = nn.Sequential(
            conv(N, N, 3),
            nn.LeakyReLU(0.1, True),
            conv(N, N, 1, 1)
        )
        self.ga_SFT3 = SFT(N, N, ks=3)

        # self.g_a = None
        self.g_a1 = nn.Sequential(
            conv(3, N),
            GDN(N),
        )
        self.g_a2 = nn.Sequential(
            conv(N, N),
            GDN(N),
        )
        self.g_a3 = nn.Sequential(
            conv(N, N),
            GDN(N),
        )
        self.g_a4 = nn.Sequential(
            conv(N, M),
        )

        # f_c
        self.qmap_feature_generation = nn.Sequential(
            UpConv2d(N, N // 2, 3),
            nn.LeakyReLU(0.1, True),
            UpConv2d(N // 2, N // 4),
            nn.LeakyReLU(0.1, True),
            conv(N // 4, N // 4, 3, 1)
        )

        self.qmap_feature_gs0 = nn.Sequential(
            conv(M + N // 4, N * 4, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(N * 4, N * 2, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(N * 2, N, 3, 1)
        )
        self.gs_SFT0 = SFT(M, N, ks=3)
        self.qmap_feature_gs1 = nn.Sequential(
            UpConv2d(N, N, 3),
            nn.LeakyReLU(0.1, True),
            conv(N, N, 1, 1)
        )
        self.gs_SFT1 = SFT(N, N, ks=3)
        self.qmap_feature_gs2 = nn.Sequential(
            UpConv2d(N, N, 3),
            nn.LeakyReLU(0.1, True),
            conv(N, N, 1, 1)
        )
        self.gs_SFT2 = SFT(N, N, ks=3)
        self.qmap_feature_gs3 = nn.Sequential(
            UpConv2d(N, N, 3),
            nn.LeakyReLU(0.1, True),
            conv(N, N, 1, 1)
        )
        self.gs_SFT3 = SFT(N, N, ks=3)


        # self.g_s = None
        self.g_s1 = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
        )
        self.g_s2 = nn.Sequential(
            deconv(N, N),
            GDN(N, inverse=True),
        )
        self.g_s3 = nn.Sequential(
            deconv(N, N),
            GDN(N, inverse=True),
        )
        self.g_s4 = nn.Sequential(
            deconv(N, 3),
        )

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, M),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )

        self.gaussian_conditional = GaussianConditional(None)

        self.lmbda = [0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.001, 0.0003]  # mxh add from HUAWEI CVPR2021 Gained...

        # Condition on Latent y, so the gain vector length M
        # e.g.: self.levels = 6 means we have 6 pairs gain vectors corresponding to 6 level RD performance
        # treat all channels the same in initialization
        self.levels = len(self.lmbda) # 8
        self.Gain = torch.nn.Parameter(torch.ones(size=[self.levels, M]), requires_grad=True)
        self.InverseGain = torch.nn.Parameter(torch.ones(size=[self.levels, M]), requires_grad=True)
        self.HyperGain = torch.nn.Parameter(torch.ones(size=[self.levels, N]), requires_grad=True)
        self.InverseHyperGain = torch.nn.Parameter(torch.ones(size=[self.levels, N]), requires_grad=True)

    def g_a(self, x, qmap):
        qmap = self.qmap_feature_ga0(torch.cat([qmap, x], dim=1))
        qmap = self.qmap_feature_ga1(qmap)
        x = self.g_a1(x)
        x = self.ga_SFT1(x, qmap)

        qmap = self.qmap_feature_ga2(qmap)
        x = self.g_a2(x)
        x = self.ga_SFT2(x, qmap)

        qmap = self.qmap_feature_ga3(qmap)
        x = self.g_a3(x)
        x = self.ga_SFT3(x, qmap)

        x = self.g_a4(x)

        return x

    def g_s(self, x, z):
        w = self.qmap_feature_generation(z)
        w = self.qmap_feature_gs0(torch.cat([w, x], dim=1))
        x = self.gs_SFT0(x, w)

        w = self.qmap_feature_gs1(w)
        x = self.g_s1(x)
        x = self.gs_SFT1(x, w)

        w = self.qmap_feature_gs2(w)
        x = self.g_s2(x)
        x = self.gs_SFT2(x, w)

        w = self.qmap_feature_gs3(w)
        x = self.g_s3(x)
        x = self.gs_SFT3(x, w)

        x = self.g_s4(x)

        return x

    def forward(self, x, s, qmap):
        '''
            x: input image
            s: random num to choose gain vector
            qmap: spatial quality map
        '''
        y = self.g_a(x, qmap)
        y = y * torch.abs(self.Gain[s]).unsqueeze(0).unsqueeze(2).unsqueeze(3) # Gain[s]: [M]  -->  [1,M,1,1]
        z = self.h_a(y)
        z = z * torch.abs(self.HyperGain[s]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        z_hat = z_hat * torch.abs(self.InverseHyperGain[s]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        y_hat = y_hat * torch.abs(self.InverseGain[s]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        x_hat = self.g_s(y_hat, z_hat)

        return {
            "y": y,
            "y_hat": y_hat,
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x, s, l, qmap):
        assert s in range(0,self.levels), f"s should in range(0, {self.levels}), but get s:{s}"
        assert l >=0 and l <=1, "l should in [0,1]"
        assert s + l <= self.levels-1
        # l = 0, Interpolated = s+1; l = 1, Interpolated = s
        if s == self.levels-1:
            InterpolatedGain = torch.abs(self.Gain[s])
            InterpolatedHyperGain = torch.abs(self.HyperGain[s])
            InterpolatedInverseHyperGain = torch.abs(self.InverseHyperGain[s])
        else:
            InterpolatedGain = torch.abs(self.Gain[s]).pow(1-l) * torch.abs(self.Gain[s+1]).pow(l)
            InterpolatedHyperGain = torch.abs(self.HyperGain[s]).pow(1-l) * torch.abs(self.HyperGain[s+1]).pow(l)
            InterpolatedInverseHyperGain = torch.abs(self.InverseHyperGain[s]).pow(1-l) * torch.abs(self.InverseHyperGain[s+1]).pow(l)

        y = self.g_a(x, qmap)
        ungained_y = y
        y = y * InterpolatedGain.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        # y_hat = self.gaussian_conditional.quantize(y, "symbols").type(torch.cuda.FloatTensor)
        z = self.h_a(y)
        z = z * InterpolatedHyperGain.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        z_hat = z_hat * InterpolatedInverseHyperGain.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        gained_y_hat = self.gaussian_conditional.quantize(y, "symbols", means_hat)
        return {"strings": [y_strings, z_strings],
                "shape": z.size()[-2:],
                "ungained_y": ungained_y,
                "gained_y": y,
                "gained_y_hat": gained_y_hat}

    def decompress(self, strings, shape, s, l):
        assert isinstance(strings, list) and len(strings) == 2 # 保证有y和z
        assert s in range(0, self.levels), f"s should in range(0,{self.levels})"
        assert l >= 0 and l <= 1, "l should in [0,1]"
        assert s + l <= self.levels-1
        if s == self.levels-1:
            InterpolatedInverseGain = torch.abs(self.InverseGain[s])
            InterpolatedInverseHyperGain = torch.abs(self.InverseHyperGain[s])
        else:
            InterpolatedInverseGain = torch.abs(self.InverseGain[s]).pow(1-l) * torch.abs(self.InverseGain[s+1]).pow(l)
            InterpolatedInverseHyperGain = torch.abs(self.InverseHyperGain[s]).pow(1-l) * torch.abs(self.InverseHyperGain[s+1]).pow(l)

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        z_hat = z_hat * InterpolatedInverseHyperGain.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )
        gained_y_hat = y_hat
        y_hat = y_hat * InterpolatedInverseGain.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        x_hat = self.g_s(y_hat, z_hat).clamp_(0, 1)
        return {"x_hat": x_hat, "gained_y_hat": gained_y_hat}

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated
    
class JointGain(MeanScaleHyperprior):
    # 基于JointAutoregressiveHierarchicalPriors更改的可变码率网络
    def __init__(self, N=192, M=192, **kwargs): # ll更改

        super().__init__(N=N, M=M, **kwargs)
        # torch.nn.Sequential是一个Sequential容器，模块将按照构造函数中传递的顺序添加到模块中
        # 神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行，同时以神经网络模块为元素的有序字典也可以作为传入参数。
        self.g_a = nn.Sequential(
            # conv(in_channels, out_channels, kernel_size=5, stride=2)
            conv(3, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, M, kernel_size=5, stride=2),
        )

        self.g_s = nn.Sequential(
            deconv(M, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, 3, kernel_size=5, stride=2),
        )

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=5),
        )

        self.h_s = nn.Sequential(
            deconv(N, M, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )

        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(M * 12 // 3, M * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 10 // 3, M * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 8 // 3, M * 6 // 3, 1),
        )

        self.context_prediction = MaskedConv2d(
            M, 2 * M, kernel_size=5, padding=2, stride=1
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)
        
        self.lmbda = [0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.001, 0.0003]  # mxh add from HUAWEI CVPR2021 Gained...

        # Condition on Latent y, so the gain vector length M
        # e.g.: self.levels = 6 means we have 6 pairs gain vectors corresponding to 6 level RD performance
        # treat all channels the same in initialization
        # 创建了一个形状为 [self.levels, M] 的张量，并将其包装在 torch.nn.Parameter 类中，表示它是一个模型参数，需要进行训练。
        # 参数 requires_grad=True 表示需要计算该参数的梯度，并在反向传播时更新该参数的值。
        # M:通道数
        self.levels = len(self.lmbda) # 8
        self.Gain = torch.nn.Parameter(torch.ones(size=[self.levels, M]), requires_grad=True)
        self.InverseGain = torch.nn.Parameter(torch.ones(size=[self.levels, M]), requires_grad=True)
        self.HyperGain = torch.nn.Parameter(torch.ones(size=[self.levels, N]), requires_grad=True)
        self.InverseHyperGain = torch.nn.Parameter(torch.ones(size=[self.levels, N]), requires_grad=True)

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)
    
    def forward(self, x, s):
        '''
            x: input image
            s: random num to choose gain vector
        '''
        # ga->Gain->ha->HyperGain->InverseHyperGain->hs->gaussian_params->InverseGain->g_s
        y = self.g_a(x)
        # Gain
        y = y * torch.abs(self.Gain[s]).unsqueeze(0).unsqueeze(2).unsqueeze(3) # Gain[s]: [M]  -->  [1,M,1,1]
        z = self.h_a(y)
        # HyperGain
        z = z * torch.abs(self.HyperGain[s]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        # InverseHyperGain
        z_hat = z_hat * torch.abs(self.InverseHyperGain[s]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        params = self.h_s(z_hat)
        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )
        ctx_params = self.context_prediction(y_hat) # φ
        # 混合高斯模型
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        # 方差、均值
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        # InverseGain
        y_hat = y_hat * torch.abs(self.InverseGain[s]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        x_hat = self.g_s(y_hat)
        
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }
        
    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x, s, l):
        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )

        assert s in range(0,self.levels-1), f"s should in range(0, {self.levels-1}), but get s:{s}"
        assert l >=0 and l <=1, "l should in [0,1]"
        
        # l = 0, Interpolated = s+1; l = 1, Interpolated = s
        # 计算l实现连续码率可变
        InterpolatedGain = torch.abs(self.Gain[s]).pow(1-l) * torch.abs(self.Gain[s+1]).pow(l)
        InterpolatedHyperGain = torch.abs(self.HyperGain[s]).pow(1-l) * torch.abs(self.HyperGain[s+1]).pow(l)
        InterpolatedInverseHyperGain = torch.abs(self.InverseHyperGain[s]).pow(1-l) * torch.abs(self.InverseHyperGain[s+1]).pow(l)

        # InterpolatedGain = torch.abs(self.Gain[s]) * (1 - l) + torch.abs(self.Gain[s + 1]) * l
        # InterpolatedHyperGain = torch.abs(self.HyperGain[s]) * (1 - l) + torch.abs(self.HyperGain[s + 1]) * l
        # InterpolatedInverseHyperGain = torch.abs(self.InverseHyperGain[s]) * (1 - l) + torch.abs(self.InverseHyperGain[s + 1]) * (l)
        
        
        y = self.g_a(x)
        # 记录没有改变的y
        ungained_y = y
        # InterpolatedGain
        y = y * InterpolatedGain.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        
        z = self.h_a(y)
        # InterpolatedHyperGain
        z = z * InterpolatedHyperGain.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        # InterpolatedInverseHyperGain
        z_hat = z_hat * InterpolatedInverseHyperGain.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        y_hat = F.pad(y, (padding, padding, padding, padding))

        y_strings = []
        for i in range(y.size(0)):
            string = self._compress_ar(
                y_hat[i : i + 1],
                params[i : i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )
            y_strings.append(string)
        gained_y_hat = self.gaussian_conditional.quantize(y, "symbols")
        return {"strings": [y_strings, z_strings], 
                "shape": z.size()[-2:],
                "gained_y_hat": gained_y_hat}

    def _compress_ar(self, y_hat, params, height, width, kernel_size, padding):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []

        # Warning, this is slow...
        # TODO: profile the calls to the bindings...
        masked_weight = self.context_prediction.weight * self.context_prediction.mask
        for h in range(height):
            for w in range(width):
                y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    masked_weight,
                    bias=self.context_prediction.bias,
                )

                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h : h + 1, w : w + 1]
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                gaussian_params = gaussian_params.squeeze(3).squeeze(2)
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.gaussian_conditional.build_indexes(scales_hat)

                y_crop = y_crop[:, :, padding, padding]
                y_q = self.gaussian_conditional.quantize(y_crop, "symbols", means_hat)
                y_hat[:, :, h + padding, w + padding] = y_q + means_hat

                symbols_list.extend(y_q.squeeze().tolist())
                indexes_list.extend(indexes.squeeze().tolist())

        encoder.encode_with_indexes(
            symbols_list, indexes_list, cdf, cdf_lengths, offsets
        )

        string = encoder.flush()
        return string

    def decompress(self, strings, shape, s, l):
        assert isinstance(strings, list) and len(strings) == 2

        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )

        # FIXME: we don't respect the default entropy coder and directly call the
        # range ANS decoder

        assert s in range(0, self.levels - 1), f"s should in range(0,{self.levels - 1})"
        assert l >= 0 and l <= 1, "l should in [0,1]"
        InterpolatedInverseGain = torch.abs(self.InverseGain[s]).pow(1-l) * torch.abs(self.InverseGain[s+1]).pow(l)
        InterpolatedInverseHyperGain = torch.abs(self.InverseHyperGain[s]).pow(1-l) * torch.abs(self.InverseHyperGain[s+1]).pow(l)
        # # Linear Interpolation can achieve the same result
        # InterpolatedInverseGain = torch.abs(self.InverseGain[s]) * (1 - l) + torch.abs(self.InverseGain[s + 1]) * (l)
        # InterpolatedInverseHyperGain = torch.abs(self.InverseHyperGain[s])* (1 - l) + torch.abs(self.InverseHyperGain[s + 1]) * (l)
        
        
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        # InterpolatedInverseHyperGain
        z_hat = z_hat * InterpolatedInverseHyperGain.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        # initialize y_hat to zeros, and pad it so we can directly work with
        # sub-tensors of size (N, C, kernel size, kernel_size)
        y_hat = torch.zeros(
            (z_hat.size(0), self.M, y_height + 2 * padding, y_width + 2 * padding),
            device=z_hat.device,
        )
        

        for i, y_string in enumerate(strings[0]):
            self._decompress_ar(
                y_string,
                y_hat[i : i + 1],
                params[i : i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )

        y_hat = F.pad(y_hat, (-padding, -padding, -padding, -padding))
        gained_y_hat = y_hat
        # InterpolatedInverseGain
        y_hat = y_hat * InterpolatedInverseGain.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat, "gained_y_hat": gained_y_hat}

    def _decompress_ar(
        self, y_string, y_hat, params, height, width, kernel_size, padding
    ):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        # Warning: this is slow due to the auto-regressive nature of the
        # decoding... See more recent publication where they use an
        # auto-regressive module on chunks of channels for faster decoding...
        for h in range(height):
            for w in range(width):
                # only perform the 5x5 convolution on a cropped tensor
                # centered in (h, w)
                y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    self.context_prediction.weight,
                    bias=self.context_prediction.bias,
                )
                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h : h + 1, w : w + 1]
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.gaussian_conditional.build_indexes(scales_hat)
                rv = decoder.decode_stream(
                    indexes.squeeze().tolist(), cdf, cdf_lengths, offsets
                )
                rv = torch.Tensor(rv).reshape(1, -1, 1, 1)
                rv = self.gaussian_conditional.dequantize(rv, means_hat)

                hp = h + padding
                wp = w + padding
                y_hat[:, :, hp : hp + 1, wp : wp + 1] = rv
                
                
# class TwoSegmentModel(MeanScaleHyperprior):
#     # 基于JointAutoregressiveHierarchicalPriors更改的可变码率网络
#     def __init__(self, N=192, M=192, **kwargs): # ll更改

#         super().__init__(N=N, M=M, **kwargs)
#         # torch.nn.Sequential是一个Sequential容器，模块将按照构造函数中传递的顺序添加到模块中
#         # 神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行，同时以神经网络模块为元素的有序字典也可以作为传入参数。
        
#         # 低码率点
#         self.g_a_lower = nn.Sequential(
#             conv(3, N, kernel_size=5, stride=2),
#             GDN(N),
#             conv(N, N, kernel_size=5, stride=2),
#             GDN(N),
#             conv(N, N, kernel_size=5, stride=2),
#             GDN(N),
#             conv(N, N, kernel_size=5, stride=2),
#         )

#         self.g_s_lower = nn.Sequential(
#             deconv(N, N, kernel_size=5, stride=2),
#             GDN(N, inverse=True),
#             deconv(N, N, kernel_size=5, stride=2),
#             GDN(N, inverse=True),
#             deconv(N, N, kernel_size=5, stride=2),
#             GDN(N, inverse=True),
#             deconv(N, 3, kernel_size=5, stride=2),
#         )
        
#         self.h_a_lower = nn.Sequential(
#             conv(N, N, stride=1, kernel_size=3),
#             nn.LeakyReLU(inplace=True),
#             conv(N, N, stride=2, kernel_size=5),
#             nn.LeakyReLU(inplace=True),
#             conv(N, N, stride=2, kernel_size=5),
#         )

#         self.h_s_lower = nn.Sequential(
#             deconv(N, N, stride=2, kernel_size=5),
#             nn.LeakyReLU(inplace=True),
#             deconv(N, N * 3 // 2, stride=2, kernel_size=5),
#             nn.LeakyReLU(inplace=True),
#             conv(N * 3 // 2, N * 2, stride=1, kernel_size=3),
#         )

#         self.entropy_parameters_lower = nn.Sequential(
#             nn.Conv2d(N * 12 // 3, N * 10 // 3, 1),
#             nn.LeakyReLU(inplace=True),
#             nn.Conv2d(N * 10 // 3, N * 8 // 3, 1),
#             nn.LeakyReLU(inplace=True),
#             nn.Conv2d(N * 8 // 3, N * 6 // 3, 1),
#         )

#         self.context_prediction_lower = MaskedConv2d(
#             N, 2 * N, kernel_size=5, padding=2, stride=1
#         )
        
        
#         # 高码率点
#         self.g_a_higher = nn.Sequential(
#             conv(3, N, kernel_size=5, stride=2),
#             GDN(N),
#             conv(N, N, kernel_size=5, stride=2),
#             GDN(N),
#             conv(N, N, kernel_size=5, stride=2),
#             GDN(N),
#             conv(N, M, kernel_size=5, stride=2),
#         )
        
#         self.g_s_higher = nn.Sequential(
#             deconv(M, N, kernel_size=5, stride=2),
#             GDN(N, inverse=True),
#             deconv(N, N, kernel_size=5, stride=2),
#             GDN(N, inverse=True),
#             deconv(N, N, kernel_size=5, stride=2),
#             GDN(N, inverse=True),
#             deconv(N, 3, kernel_size=5, stride=2),
#         )

#         self.h_a_higher = nn.Sequential(
#             conv(M, N, stride=1, kernel_size=3),
#             nn.LeakyReLU(inplace=True),
#             conv(N, N, stride=2, kernel_size=5),
#             nn.LeakyReLU(inplace=True),
#             conv(N, N, stride=2, kernel_size=5),
#         )

#         self.h_s_higher = nn.Sequential(
#             deconv(N, M, stride=2, kernel_size=5),
#             nn.LeakyReLU(inplace=True),
#             deconv(M, M * 3 // 2, stride=2, kernel_size=5),
#             nn.LeakyReLU(inplace=True),
#             conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
#         )

#         self.entropy_parameters_higher = nn.Sequential(
#             nn.Conv2d(M * 12 // 3, M * 10 // 3, 1),
#             nn.LeakyReLU(inplace=True),
#             nn.Conv2d(M * 10 // 3, M * 8 // 3, 1),
#             nn.LeakyReLU(inplace=True),
#             nn.Conv2d(M * 8 // 3, M * 6 // 3, 1),
#         )

#         self.context_prediction_higher = MaskedConv2d(
#             M, 2 * M, kernel_size=5, padding=2, stride=1
#         )

#         self.gaussian_conditional = GaussianConditional(None)
#         self.N = int(N)
#         self.M = int(M)
        
#         self.lmbda = [0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.001, 0.0003]  # mxh add from HUAWEI CVPR2021 Gained...

#         # Condition on Latent y, so the gain vector length M
#         # e.g.: self.levels = 6 means we have 6 pairs gain vectors corresponding to 6 level RD performance
#         # treat all channels the same in initialization
#         # 创建了一个形状为 [self.levels, M] 的张量，并将其包装在 torch.nn.Parameter 类中，表示它是一个模型参数，需要进行训练。
#         # 参数 requires_grad=True 表示需要计算该参数的梯度，并在反向传播时更新该参数的值。
#         # M:通道数
#         self.levels = len(self.lmbda) # 8
#         # self.Gain = torch.nn.Parameter(torch.ones(size=[self.levels, M]), requires_grad=True)
#         # self.InverseGain = torch.nn.Parameter(torch.ones(size=[self.levels, M]), requires_grad=True)
#         # self.HyperGain = torch.nn.Parameter(torch.ones(size=[self.levels, N]), requires_grad=True)
#         # self.InverseHyperGain = torch.nn.Parameter(torch.ones(size=[self.levels, N]), requires_grad=True)
#         # 低码率点
#         self.Gain_lower = torch.nn.Parameter(torch.ones(size=[self.levels, N]), requires_grad=True)
#         self.InverseGain_lower = torch.nn.Parameter(torch.ones(size=[self.levels, N]), requires_grad=True)
#         # 高码率点
#         self.Gain_higher = torch.nn.Parameter(torch.ones(size=[self.levels, M]), requires_grad=True)
#         self.InverseGain_higher = torch.nn.Parameter(torch.ones(size=[self.levels, M]), requires_grad=True)
#         self.HyperGain = torch.nn.Parameter(torch.ones(size=[self.levels, N]), requires_grad=True)
#         self.InverseHyperGain = torch.nn.Parameter(torch.ones(size=[self.levels, N]), requires_grad=True)

#     @property
#     def downsampling_factor(self) -> int:
#         return 2 ** (4 + 2)
    
#     def forward(self, x, s):
#         '''
#             x: input image
#             s: random num to choose gain vector
#         '''
#         if s >= 4: # 低码率点
#             g_a = self.g_a_lower
#             h_a = self.h_a_lower
#             h_s = self.h_s_lower
#             g_s = self.g_s_lower
#             context_prediction = self.context_prediction_lower
#             entropy_parameters = self.entropy_parameters_lower
#             Gain = self.Gain_lower
#             InverseGain = self.InverseGain_lower
#         else: # 高码率点
#             g_a = self.g_a_higher
#             h_a = self.h_a_higher
#             h_s = self.h_s_higher
#             g_s = self.g_s_higher
#             context_prediction = self.context_prediction_higher
#             entropy_parameters = self.entropy_parameters_higher
#             Gain = self.Gain_higher
#             InverseGain = self.InverseGain_higher
        
#         # ga->Gain->ha->HyperGain->InverseHyperGain->hs->gaussian_params->InverseGain->g_s
#         y = g_a(x)
#         # Gain
#         y = y * torch.abs(Gain[s]).unsqueeze(0).unsqueeze(2).unsqueeze(3) # Gain[s]: [M]  -->  [1,M,1,1]
#         z = h_a(y)
#         # HyperGain
#         z = z * torch.abs(self.HyperGain[s]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
#         z_hat, z_likelihoods = self.entropy_bottleneck(z)
#         # InverseHyperGain
#         z_hat = z_hat * torch.abs(self.InverseHyperGain[s]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
#         params = h_s(z_hat)
#         y_hat = self.gaussian_conditional.quantize(
#             y, "noise" if self.training else "dequantize"
#         )
#         ctx_params = context_prediction(y_hat) # φ
#         # 混合高斯模型
#         gaussian_params = entropy_parameters(
#             torch.cat((params, ctx_params), dim=1)
#         )
#         # 方差、均值
#         scales_hat, means_hat = gaussian_params.chunk(2, 1)
#         _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
#         # InverseGain
#         y_hat = y_hat * torch.abs(InverseGain[s]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
#         x_hat = g_s(y_hat)
        
#         return {
#             "x_hat": x_hat,
#             "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
#         }
        
#     @classmethod
#     def from_state_dict(cls, state_dict):
#         """Return a new model instance from `state_dict`."""
#         N = state_dict["g_a.0.weight"].size(0)
#         M = state_dict["g_a.6.weight"].size(0)
#         net = cls(N, M)
#         net.load_state_dict(state_dict)
#         return net

#     def compress(self, x, s, l):
#         if next(self.parameters()).device != torch.device("cpu"):
#             warnings.warn(
#                 "Inference on GPU is not recommended for the autoregressive "
#                 "models (the entropy coder is run sequentially on CPU)."
#             )

#         assert s in range(0,self.levels-1), f"s should in range(0, {self.levels-1}), but get s:{s}"
#         assert l >=0 and l <=1, "l should in [0,1]"
        
#         if s >= 4: # 低码率点
#             g_a = self.g_a_lower
#             h_a = self.h_a_lower
#             h_s = self.h_s_lower
#             Gain = self.Gain_lower
#         else: # 高码率点
#             g_a = self.g_a_higher
#             h_a = self.h_a_higher
#             h_s = self.h_s_higher
#             Gain = self.Gain_higher
        
#         # l = 0, Interpolated = s+1; l = 1, Interpolated = s
#         # 计算l实现连续码率可变
#         InterpolatedGain = torch.abs(Gain[s]).pow(1-l) * torch.abs(Gain[s+1]).pow(l)
#         InterpolatedHyperGain = torch.abs(self.HyperGain[s]).pow(1-l) * torch.abs(self.HyperGain[s+1]).pow(l)
#         InterpolatedInverseHyperGain = torch.abs(self.InverseHyperGain[s]).pow(1-l) * torch.abs(self.InverseHyperGain[s+1]).pow(l)

#         # InterpolatedGain = torch.abs(self.Gain[s]) * (1 - l) + torch.abs(self.Gain[s + 1]) * l
#         # InterpolatedHyperGain = torch.abs(self.HyperGain[s]) * (1 - l) + torch.abs(self.HyperGain[s + 1]) * l
#         # InterpolatedInverseHyperGain = torch.abs(self.InverseHyperGain[s]) * (1 - l) + torch.abs(self.InverseHyperGain[s + 1]) * (l)
        
        
#         y = g_a(x)
#         # 记录没有改变的y
#         ungained_y = y
#         # InterpolatedGain
#         y = y * InterpolatedGain.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        
#         z = h_a(y)
#         # InterpolatedHyperGain
#         z = z * InterpolatedHyperGain.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        
#         z_strings = self.entropy_bottleneck.compress(z)
#         z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
#         # InterpolatedInverseHyperGain
#         z_hat = z_hat * InterpolatedInverseHyperGain.unsqueeze(0).unsqueeze(2).unsqueeze(3)

#         params = h_s(z_hat)

#         s = 4  # scaling factor between z and y
#         kernel_size = 5  # context prediction kernel size
#         padding = (kernel_size - 1) // 2

#         y_height = z_hat.size(2) * s
#         y_width = z_hat.size(3) * s

#         y_hat = F.pad(y, (padding, padding, padding, padding))

#         y_strings = []
#         for i in range(y.size(0)):
#             string = self._compress_ar(
#                 y_hat[i : i + 1],
#                 params[i : i + 1],
#                 y_height,
#                 y_width,
#                 kernel_size,
#                 padding,
#                 s
#             )
#             y_strings.append(string)

#         gained_y_hat = self.gaussian_conditional.quantize(y, "symbols")
#         return {"strings": [y_strings, z_strings], 
#                 "shape": z.size()[-2:],
#                 "gained_y_hat": gained_y_hat}

#     def _compress_ar(self, y_hat, params, height, width, kernel_size, padding, s):
#         if s >= 4: # 低码率点
#             context_prediction = self.context_prediction_lower
#             entropy_parameters = self.entropy_parameters_lower
#         else: # 高码率点
#             context_prediction = self.context_prediction_higher
#             entropy_parameters = self.entropy_parameters_higher
#         cdf = self.gaussian_conditional.quantized_cdf.tolist()
#         cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
#         offsets = self.gaussian_conditional.offset.tolist()

#         encoder = BufferedRansEncoder()
#         symbols_list = []
#         indexes_list = []

#         # Warning, this is slow...
#         # TODO: profile the calls to the bindings...
#         masked_weight = context_prediction.weight * context_prediction.mask
#         for h in range(height):
#             for w in range(width):
#                 y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
#                 ctx_p = F.conv2d(
#                     y_crop,
#                     masked_weight,
#                     bias=context_prediction.bias,
#                 )

#                 # 1x1 conv for the entropy parameters prediction network, so
#                 # we only keep the elements in the "center"
#                 p = params[:, :, h : h + 1, w : w + 1]
#                 gaussian_params = entropy_parameters(torch.cat((p, ctx_p), dim=1))
#                 gaussian_params = gaussian_params.squeeze(3).squeeze(2)
#                 scales_hat, means_hat = gaussian_params.chunk(2, 1)

#                 indexes = self.gaussian_conditional.build_indexes(scales_hat)

#                 y_crop = y_crop[:, :, padding, padding]
#                 y_q = self.gaussian_conditional.quantize(y_crop, "symbols", means_hat)
#                 y_hat[:, :, h + padding, w + padding] = y_q + means_hat

#                 symbols_list.extend(y_q.squeeze().tolist())
#                 indexes_list.extend(indexes.squeeze().tolist())

#         encoder.encode_with_indexes(
#             symbols_list, indexes_list, cdf, cdf_lengths, offsets
#         )

#         string = encoder.flush()
#         return string

#     def decompress(self, strings, shape, s, l):
#         assert isinstance(strings, list) and len(strings) == 2

#         if next(self.parameters()).device != torch.device("cpu"):
#             warnings.warn(
#                 "Inference on GPU is not recommended for the autoregressive "
#                 "models (the entropy coder is run sequentially on CPU)."
#             )

#         # FIXME: we don't respect the default entropy coder and directly call the
#         # range ANS decoder
        
#         if s >= 4: # 低码率点
#             h_s = self.h_s_lower
#             g_s = self.g_s_lower
#             InverseGain = self.InverseGain_lower
#         else: # 高码率点
#             h_s = self.h_s_higher
#             g_s = self.g_s_higher
#             InverseGain = self.InverseGain_higher

#         assert s in range(0, self.levels - 1), f"s should in range(0,{self.levels - 1})"
#         assert l >= 0 and l <= 1, "l should in [0,1]"
#         InterpolatedInverseGain = torch.abs(InverseGain[s]).pow(1-l) * torch.abs(InverseGain[s+1]).pow(l)
#         InterpolatedInverseHyperGain = torch.abs(self.InverseHyperGain[s]).pow(1-l) * torch.abs(self.InverseHyperGain[s+1]).pow(l)
#         # # Linear Interpolation can achieve the same result
#         # InterpolatedInverseGain = torch.abs(self.InverseGain[s]) * (1 - l) + torch.abs(self.InverseGain[s + 1]) * (l)
#         # InterpolatedInverseHyperGain = torch.abs(self.InverseHyperGain[s])* (1 - l) + torch.abs(self.InverseHyperGain[s + 1]) * (l)
        
        
#         z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
#         # InterpolatedInverseHyperGain
#         z_hat = z_hat * InterpolatedInverseHyperGain.unsqueeze(0).unsqueeze(2).unsqueeze(3)
#         params = h_s(z_hat)

#         s = 4  # scaling factor between z and y
#         kernel_size = 5  # context prediction kernel size
#         padding = (kernel_size - 1) // 2

#         y_height = z_hat.size(2) * s
#         y_width = z_hat.size(3) * s

#         # initialize y_hat to zeros, and pad it so we can directly work with
#         # sub-tensors of size (N, C, kernel size, kernel_size)
#         y_hat = torch.zeros(
#             (z_hat.size(0), self.M, y_height + 2 * padding, y_width + 2 * padding),
#             device=z_hat.device,
#         )
        

#         for i, y_string in enumerate(strings[0]):
#             self._decompress_ar(
#                 y_string,
#                 y_hat[i : i + 1],
#                 params[i : i + 1],
#                 y_height,
#                 y_width,
#                 kernel_size,
#                 padding,
#                 s
#             )

#         y_hat = F.pad(y_hat, (-padding, -padding, -padding, -padding))
#         gained_y_hat = y_hat
#         # InterpolatedInverseGain
#         y_hat = y_hat * InterpolatedInverseGain.unsqueeze(0).unsqueeze(2).unsqueeze(3)
#         x_hat = g_s(y_hat).clamp_(0, 1)
#         return {"x_hat": x_hat, "gained_y_hat": gained_y_hat}

#     def _decompress_ar(
#         self, y_string, y_hat, params, height, width, kernel_size, padding, s
#     ):
#         if s >= 4: # 低码率点
#             context_prediction = self.context_prediction_lower
#             entropy_parameters = self.entropy_parameters_lower
#         else: # 高码率点
#             context_prediction = self.context_prediction_higher
#             entropy_parameters = self.entropy_parameters_higher
            
#         cdf = self.gaussian_conditional.quantized_cdf.tolist()
#         cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
#         offsets = self.gaussian_conditional.offset.tolist()

#         decoder = RansDecoder()
#         decoder.set_stream(y_string)

#         # Warning: this is slow due to the auto-regressive nature of the
#         # decoding... See more recent publication where they use an
#         # auto-regressive module on chunks of channels for faster decoding...
#         for h in range(height):
#             for w in range(width):
#                 # only perform the 5x5 convolution on a cropped tensor
#                 # centered in (h, w)
#                 y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
#                 ctx_p = F.conv2d(
#                     y_crop,
#                     context_prediction.weight,
#                     bias=context_prediction.bias,
#                 )
#                 # 1x1 conv for the entropy parameters prediction network, so
#                 # we only keep the elements in the "center"
#                 p = params[:, :, h : h + 1, w : w + 1]
#                 gaussian_params = entropy_parameters(torch.cat((p, ctx_p), dim=1))
#                 scales_hat, means_hat = gaussian_params.chunk(2, 1)

#                 indexes = self.gaussian_conditional.build_indexes(scales_hat)
#                 rv = decoder.decode_stream(
#                     indexes.squeeze().tolist(), cdf, cdf_lengths, offsets
#                 )
#                 rv = torch.Tensor(rv).reshape(1, -1, 1, 1)
#                 rv = self.gaussian_conditional.dequantize(rv, means_hat)

#                 hp = h + padding
#                 wp = w + padding
#                 y_hat[:, :, hp : hp + 1, wp : wp + 1] = rv
                
# class TwoSegmentGa(MeanScaleHyperprior):
#     # 基于JointAutoregressiveHierarchicalPriors更改的可变码率网络
#     def __init__(self, N=192, M=192, **kwargs): # ll更改

#         super().__init__(N=N, M=M, **kwargs)
#         self.g_a_lower = nn.Sequential(
#             conv(3, N, kernel_size=5, stride=2),
#             GDN(N),
#             conv(N, N, kernel_size=5, stride=2),
#             GDN(N),
#             conv(N, N, kernel_size=5, stride=2),
#             GDN(N),
#             conv(N, M, kernel_size=5, stride=2),
#         )
        
#         self.g_a_higher = nn.Sequential(
#             conv(3, N, kernel_size=5, stride=2),
#             GDN(N),
#             conv(N, N, kernel_size=3, stride=2),
#             GDN(N),
#             conv(N, N, kernel_size=3, stride=1),
#             GDN(N),
#             conv(N, N, kernel_size=3, stride=2),
#             GDN(N),
#             conv(N, N, kernel_size=3, stride=1),
#             GDN(N),
#             conv(N, M, kernel_size=5, stride=2),
#         )
        

#         self.g_s_lower = nn.Sequential(
#             deconv(M, N, kernel_size=5, stride=2),
#             GDN(N, inverse=True),
#             deconv(N, N, kernel_size=5, stride=2),
#             GDN(N, inverse=True),
#             deconv(N, N, kernel_size=5, stride=2),
#             GDN(N, inverse=True),
#             deconv(N, 3, kernel_size=5, stride=2),
#         )
        
#         self.g_s_higher = nn.Sequential(
#             deconv(M, N, kernel_size=5, stride=2),
#             GDN(N, inverse=True),
#             deconv(N, N, kernel_size=5, stride=2),
#             GDN(N, inverse=True),
#             deconv(N, N, kernel_size=5, stride=2),
#             GDN(N, inverse=True),
#             deconv(N, 3, kernel_size=5, stride=2),
#         )

#         self.h_a = nn.Sequential(
#             conv(M, N, stride=1, kernel_size=3),
#             nn.LeakyReLU(inplace=True),
#             conv(N, N, stride=2, kernel_size=5),
#             nn.LeakyReLU(inplace=True),
#             conv(N, N, stride=2, kernel_size=5),
#         )

#         self.h_s = nn.Sequential(
#             deconv(N, M, stride=2, kernel_size=5),
#             nn.LeakyReLU(inplace=True),
#             deconv(M, M * 3 // 2, stride=2, kernel_size=5),
#             nn.LeakyReLU(inplace=True),
#             conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
#         )

#         self.entropy_parameters = nn.Sequential(
#             nn.Conv2d(M * 12 // 3, M * 10 // 3, 1),
#             nn.LeakyReLU(inplace=True),
#             nn.Conv2d(M * 10 // 3, M * 8 // 3, 1),
#             nn.LeakyReLU(inplace=True),
#             nn.Conv2d(M * 8 // 3, M * 6 // 3, 1),
#         )

#         self.context_prediction = MaskedConv2d(
#             M, 2 * M, kernel_size=5, padding=2, stride=1
#         )

#         self.gaussian_conditional = GaussianConditional(None)
#         self.N = int(N)
#         self.M = int(M)
        
#         self.lmbda = [0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.001, 0.0003]  # mxh add from HUAWEI CVPR2021 Gained...

#         # Condition on Latent y, so the gain vector length M
#         # e.g.: self.levels = 6 means we have 6 pairs gain vectors corresponding to 6 level RD performance
#         # treat all channels the same in initialization
#         # 创建了一个形状为 [self.levels, M] 的张量，并将其包装在 torch.nn.Parameter 类中，表示它是一个模型参数，需要进行训练。
#         # 参数 requires_grad=True 表示需要计算该参数的梯度，并在反向传播时更新该参数的值。
#         # M:通道数
#         self.levels = len(self.lmbda) # 8
#         self.Gain = torch.nn.Parameter(torch.ones(size=[self.levels, M]), requires_grad=True)
#         self.InverseGain = torch.nn.Parameter(torch.ones(size=[self.levels, M]), requires_grad=True)
#         self.HyperGain = torch.nn.Parameter(torch.ones(size=[self.levels, N]), requires_grad=True)
#         self.InverseHyperGain = torch.nn.Parameter(torch.ones(size=[self.levels, N]), requires_grad=True)

#     @property
#     def downsampling_factor(self) -> int:
#         return 2 ** (4 + 2)
    
#     def forward(self, x, s):
#         '''
#             x: input image
#             s: random num to choose gain vector
#         '''
#         if s >= 4: # 低码率点
#             g_a = self.g_a_lower
#             g_s = self.g_s_lower
#         else: # 高码率点
#             g_a = self.g_a_higher
#             g_s = self.g_s_higher
            
#         # ga->Gain->ha->HyperGain->InverseHyperGain->hs->gaussian_params->InverseGain->g_s
#         y = g_a(x)
#         # Gain
#         y = y * torch.abs(self.Gain[s]).unsqueeze(0).unsqueeze(2).unsqueeze(3) # Gain[s]: [M]  -->  [1,M,1,1]
#         z = self.h_a(y)
#         # HyperGain
#         z = z * torch.abs(self.HyperGain[s]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
#         z_hat, z_likelihoods = self.entropy_bottleneck(z)
#         # InverseHyperGain
#         z_hat = z_hat * torch.abs(self.InverseHyperGain[s]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
#         params = self.h_s(z_hat)
#         y_hat = self.gaussian_conditional.quantize(
#             y, "noise" if self.training else "dequantize"
#         )
#         ctx_params = self.context_prediction(y_hat) # φ
#         # 混合高斯模型
#         gaussian_params = self.entropy_parameters(
#             torch.cat((params, ctx_params), dim=1)
#         )
#         # 方差、均值
#         scales_hat, means_hat = gaussian_params.chunk(2, 1)
#         _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
#         # InverseGain
#         y_hat = y_hat * torch.abs(self.InverseGain[s]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
#         x_hat = g_s(y_hat)
        
#         return {
#             "x_hat": x_hat,
#             "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
#         }
        
#     @classmethod
#     def from_state_dict(cls, state_dict):
#         """Return a new model instance from `state_dict`."""
#         N = state_dict["g_a.0.weight"].size(0)
#         M = state_dict["g_a.6.weight"].size(0)
#         net = cls(N, M)
#         net.load_state_dict(state_dict)
#         return net

#     def compress(self, x, s, l):
#         if next(self.parameters()).device != torch.device("cpu"):
#             warnings.warn(
#                 "Inference on GPU is not recommended for the autoregressive "
#                 "models (the entropy coder is run sequentially on CPU)."
#             )

#         assert s in range(0,self.levels-1), f"s should in range(0, {self.levels-1}), but get s:{s}"
#         assert l >=0 and l <=1, "l should in [0,1]"
        
#         # l = 0, Interpolated = s+1; l = 1, Interpolated = s
#         # 计算l实现连续码率可变
#         InterpolatedGain = torch.abs(self.Gain[s]).pow(1-l) * torch.abs(self.Gain[s+1]).pow(l)
#         InterpolatedHyperGain = torch.abs(self.HyperGain[s]).pow(1-l) * torch.abs(self.HyperGain[s+1]).pow(l)
#         InterpolatedInverseHyperGain = torch.abs(self.InverseHyperGain[s]).pow(1-l) * torch.abs(self.InverseHyperGain[s+1]).pow(l)

#         # InterpolatedGain = torch.abs(self.Gain[s]) * (1 - l) + torch.abs(self.Gain[s + 1]) * l
#         # InterpolatedHyperGain = torch.abs(self.HyperGain[s]) * (1 - l) + torch.abs(self.HyperGain[s + 1]) * l
#         # InterpolatedInverseHyperGain = torch.abs(self.InverseHyperGain[s]) * (1 - l) + torch.abs(self.InverseHyperGain[s + 1]) * (l)
        
#         if s >= 4: # 低码率点
#             g_a = self.g_a_lower
#             g_s = self.g_s_lower
#         else: # 高码率点
#             g_a = self.g_a_higher
#             g_s = self.g_s_higher
        
#         y = g_a(x)
#         # 记录没有改变的y
#         ungained_y = y
#         # InterpolatedGain
#         y = y * InterpolatedGain.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        
#         z = self.h_a(y)
#         # InterpolatedHyperGain
#         z = z * InterpolatedHyperGain.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        
#         z_strings = self.entropy_bottleneck.compress(z)
#         z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
#         # InterpolatedInverseHyperGain
#         z_hat = z_hat * InterpolatedInverseHyperGain.unsqueeze(0).unsqueeze(2).unsqueeze(3)

#         params = self.h_s(z_hat)

#         s = 4  # scaling factor between z and y
#         kernel_size = 5  # context prediction kernel size
#         padding = (kernel_size - 1) // 2

#         y_height = z_hat.size(2) * s
#         y_width = z_hat.size(3) * s

#         y_hat = F.pad(y, (padding, padding, padding, padding))

#         y_strings = []
#         for i in range(y.size(0)):
#             string = self._compress_ar(
#                 y_hat[i : i + 1],
#                 params[i : i + 1],
#                 y_height,
#                 y_width,
#                 kernel_size,
#                 padding,
#             )
#             y_strings.append(string)

#         gained_y_hat = self.gaussian_conditional.quantize(y, "symbols")
#         return {"strings": [y_strings, z_strings], 
#                 "shape": z.size()[-2:],
#                 "gained_y_hat": gained_y_hat}

#     def _compress_ar(self, y_hat, params, height, width, kernel_size, padding):
#         cdf = self.gaussian_conditional.quantized_cdf.tolist()
#         cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
#         offsets = self.gaussian_conditional.offset.tolist()

#         encoder = BufferedRansEncoder()
#         symbols_list = []
#         indexes_list = []

#         # Warning, this is slow...
#         # TODO: profile the calls to the bindings...
#         masked_weight = self.context_prediction.weight * self.context_prediction.mask
#         for h in range(height):
#             for w in range(width):
#                 y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
#                 ctx_p = F.conv2d(
#                     y_crop,
#                     masked_weight,
#                     bias=self.context_prediction.bias,
#                 )

#                 # 1x1 conv for the entropy parameters prediction network, so
#                 # we only keep the elements in the "center"
#                 p = params[:, :, h : h + 1, w : w + 1]
#                 gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
#                 gaussian_params = gaussian_params.squeeze(3).squeeze(2)
#                 scales_hat, means_hat = gaussian_params.chunk(2, 1)

#                 indexes = self.gaussian_conditional.build_indexes(scales_hat)

#                 y_crop = y_crop[:, :, padding, padding]
#                 y_q = self.gaussian_conditional.quantize(y_crop, "symbols", means_hat)
#                 y_hat[:, :, h + padding, w + padding] = y_q + means_hat

#                 symbols_list.extend(y_q.squeeze().tolist())
#                 indexes_list.extend(indexes.squeeze().tolist())

#         encoder.encode_with_indexes(
#             symbols_list, indexes_list, cdf, cdf_lengths, offsets
#         )

#         string = encoder.flush()
#         return string

#     def decompress(self, strings, shape, s, l):
#         assert isinstance(strings, list) and len(strings) == 2

#         if next(self.parameters()).device != torch.device("cpu"):
#             warnings.warn(
#                 "Inference on GPU is not recommended for the autoregressive "
#                 "models (the entropy coder is run sequentially on CPU)."
#             )

#         # FIXME: we don't respect the default entropy coder and directly call the
#         # range ANS decoder

#         assert s in range(0, self.levels - 1), f"s should in range(0,{self.levels - 1})"
#         assert l >= 0 and l <= 1, "l should in [0,1]"
#         InterpolatedInverseGain = torch.abs(self.InverseGain[s]).pow(1-l) * torch.abs(self.InverseGain[s+1]).pow(l)
#         InterpolatedInverseHyperGain = torch.abs(self.InverseHyperGain[s]).pow(1-l) * torch.abs(self.InverseHyperGain[s+1]).pow(l)
#         # # Linear Interpolation can achieve the same result
#         # InterpolatedInverseGain = torch.abs(self.InverseGain[s]) * (1 - l) + torch.abs(self.InverseGain[s + 1]) * (l)
#         # InterpolatedInverseHyperGain = torch.abs(self.InverseHyperGain[s])* (1 - l) + torch.abs(self.InverseHyperGain[s + 1]) * (l)
        
#         if s >= 4: # 低码率点
#             g_a = self.g_a_lower
#             g_s = self.g_s_lower
#         else: # 高码率点
#             g_a = self.g_a_higher
#             g_s = self.g_s_higher
        
#         z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
#         # InterpolatedInverseHyperGain
#         z_hat = z_hat * InterpolatedInverseHyperGain.unsqueeze(0).unsqueeze(2).unsqueeze(3)
#         params = self.h_s(z_hat)

#         s = 4  # scaling factor between z and y
#         kernel_size = 5  # context prediction kernel size
#         padding = (kernel_size - 1) // 2

#         y_height = z_hat.size(2) * s
#         y_width = z_hat.size(3) * s

#         # initialize y_hat to zeros, and pad it so we can directly work with
#         # sub-tensors of size (N, C, kernel size, kernel_size)
#         y_hat = torch.zeros(
#             (z_hat.size(0), self.M, y_height + 2 * padding, y_width + 2 * padding),
#             device=z_hat.device,
#         )
        

#         for i, y_string in enumerate(strings[0]):
#             self._decompress_ar(
#                 y_string,
#                 y_hat[i : i + 1],
#                 params[i : i + 1],
#                 y_height,
#                 y_width,
#                 kernel_size,
#                 padding,
#             )

#         y_hat = F.pad(y_hat, (-padding, -padding, -padding, -padding))
#         gained_y_hat = y_hat
#         # InterpolatedInverseGain
#         y_hat = y_hat * InterpolatedInverseGain.unsqueeze(0).unsqueeze(2).unsqueeze(3)
#         x_hat = g_s(y_hat).clamp_(0, 1)
#         return {"x_hat": x_hat, "gained_y_hat": gained_y_hat}

#     def _decompress_ar(
#         self, y_string, y_hat, params, height, width, kernel_size, padding
#     ):
#         cdf = self.gaussian_conditional.quantized_cdf.tolist()
#         cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
#         offsets = self.gaussian_conditional.offset.tolist()

#         decoder = RansDecoder()
#         decoder.set_stream(y_string)

#         # Warning: this is slow due to the auto-regressive nature of the
#         # decoding... See more recent publication where they use an
#         # auto-regressive module on chunks of channels for faster decoding...
#         for h in range(height):
#             for w in range(width):
#                 # only perform the 5x5 convolution on a cropped tensor
#                 # centered in (h, w)
#                 y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
#                 ctx_p = F.conv2d(
#                     y_crop,
#                     self.context_prediction.weight,
#                     bias=self.context_prediction.bias,
#                 )
#                 # 1x1 conv for the entropy parameters prediction network, so
#                 # we only keep the elements in the "center"
#                 p = params[:, :, h : h + 1, w : w + 1]
#                 gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
#                 scales_hat, means_hat = gaussian_params.chunk(2, 1)

#                 indexes = self.gaussian_conditional.build_indexes(scales_hat)
#                 rv = decoder.decode_stream(
#                     indexes.squeeze().tolist(), cdf, cdf_lengths, offsets
#                 )
#                 rv = torch.Tensor(rv).reshape(1, -1, 1, 1)
#                 rv = self.gaussian_conditional.dequantize(rv, means_hat)

#                 hp = h + padding
#                 wp = w + padding
#                 y_hat[:, :, hp : hp + 1, wp : wp + 1] = rv
                
                 
                
class twoSegmentGa_STB(MeanScaleHyperprior):
    # 基于JointAutoregressiveHierarchicalPriors更改的可变码率网络
    def __init__(self, N=192, M=192, **kwargs): # ll更改

        super().__init__(N=N, M=M, **kwargs)
        
        depths = [1, 2]
        num_heads = [4]
        window_size = 8
        mlp_ratio = 4.
        qkv_bias = True
        qk_scale = None
        drop_rate = 0.
        attn_drop_rate = 0.
        drop_path_rate = 0.2
        norm_layer = nn.LayerNorm
        use_checkpoint= False

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        
        
        # 低码率点g_a_lower
        # self.g_a_lower = nn.Sequential(
        #     conv(3, N, kernel_size=5, stride=2),
        #     ResBottleneck(N),
        #     conv(N, N, kernel_size=5, stride=2),
        #     ResBottleneck(N),
        #     conv(N, N, kernel_size=5, stride=2),
        #     ResBottleneck(N),
        #     conv(N, M, kernel_size=5, stride=2),
        # )
        
        # 低码率点
        self.g_a_low_0 = conv(3, N, kernel_size=5, stride=2)
        self.g_a_low_1 = ResBottleneck(N)
        self.g_a_low_2 = conv(N, N, kernel_size=5, stride=2)
        self.g_a_low_3 = ResBottleneck(N)
        self.g_a_low_4 = conv(N, N, kernel_size=5, stride=2)
        self.g_a_low_5 = ResBottleneck(N)
        self.g_a_low_6 = conv(N, M, kernel_size=5, stride=2)
        
        # 高码率点g_a_higher
        # self.g_a_higher = nn.Sequential(
        #     conv(3, N, kernel_size=5, stride=2),
        #     ResBottleneck(N),
        #     conv(N, N, kernel_size=5, stride=2),
        #     last_STB(dim=N,
        #             input_resolution=(32,32),
        #             depth=depths[2],
        #             num_heads=num_heads[2],
        #             window_size=window_size,
        #             mlp_ratio=mlp_ratio,
        #             qkv_bias=qkv_bias, qk_scale=qk_scale,
        #             drop=drop_rate, attn_drop=attn_drop_rate,
        #             drop_path=dpr[sum(depths[:2]):sum(depths[:3])],
        #             norm_layer=norm_layer,
        #             use_checkpoint=use_checkpoint,
        #             ),
        #     conv(N, N, kernel_size=5, stride=2),
        #     ResBottleneck(N),
        #     conv(N, M, kernel_size=5, stride=2),
        # )
        
        self.g_a_high_0 = conv(3, N, kernel_size=5, stride=2)
        self.g_a_high_1 = ResBottleneck(N)
        self.g_a_high_2 = conv(N, N, kernel_size=5, stride=2)
        self.g_a_high_3 = last_STB(dim=N,
                        input_resolution=(64,64),
                        depth=depths[0],
                        num_heads=num_heads[0],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:0]):sum(depths[:1])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint,
                        )
        self.g_a_high_4 = conv(N, N, kernel_size=5, stride=2)
        self.g_a_high_5 = ResBottleneck(N)
        self.g_a_high_6 = conv(N, M, kernel_size=5, stride=2)
    
        

        # 低码率点g_s_lower
        # self.g_s_lower = nn.Sequential(
        #     deconv(M, N, kernel_size=5, stride=2),
        #     ResBottleneck(N),
        #     deconv(N, N, kernel_size=5, stride=2),
        #     ResBottleneck(N),
        #     deconv(N, N, kernel_size=5, stride=2),
        #     ResBottleneck(N),
        #     deconv(N, 3, kernel_size=5, stride=2),
        # )
        self.g_s_low_0 = deconv(M, N, kernel_size=5, stride=2)
        self.g_s_low_1 = ResBottleneck(N)
        self.g_s_low_2 = deconv(N, N, kernel_size=5, stride=2)
        self.g_s_low_3 = ResBottleneck(N)
        self.g_s_low_4 = deconv(N, N, kernel_size=5, stride=2)
        self.g_s_low_5 = ResBottleneck(N)
        self.g_s_low_6 = deconv(N, 3, kernel_size=5, stride=2)

        
        # self.g_s_higher = nn.Sequential(
        #     deconv(M, N, kernel_size=5, stride=2),
        #     ResBottleneck(N),
        #     deconv(N, N, kernel_size=5, stride=2),
        #     last_STB(dim=N,
        #             input_resolution=(32,32),
        #             depth=depths[2],
        #             num_heads=num_heads[2],
        #             window_size=window_size,
        #             mlp_ratio=mlp_ratio,
        #             qkv_bias=qkv_bias, qk_scale=qk_scale,
        #             drop=drop_rate, attn_drop=attn_drop_rate,
        #             drop_path=dpr[sum(depths[:2]):sum(depths[:3])],
        #             norm_layer=norm_layer,
        #             use_checkpoint=use_checkpoint,
        #             ),
        #     deconv(N, N, kernel_size=5, stride=2),
        #     ResBottleneck(N),
        #     deconv(N, 3, kernel_size=5, stride=2),
        # )
        
        self.g_s_high_0 = deconv(M, N, kernel_size=5, stride=2)
        self.g_s_high_1 = ResBottleneck(N)
        self.g_s_high_2 = deconv(N, N, kernel_size=5, stride=2)
        self.g_s_high_3 = last_STB(dim=N,
                        input_resolution=(64,64),
                        depth=depths[0],
                        num_heads=num_heads[0],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:0]):sum(depths[:1])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint,
                        )
        self.g_s_high_4 = deconv(N, N, kernel_size=5, stride=2)
        self.g_s_high_5 = ResBottleneck(N)
        self.g_s_high_6 = deconv(N, 3, kernel_size=5, stride=2)
        

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=5),
        )

        self.h_s = nn.Sequential(
            deconv(N, M, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )

        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(M * 12 // 3, M * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 10 // 3, M * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 8 // 3, M * 6 // 3, 1),
        )

        self.context_prediction = MaskedConv2d(
            M, 2 * M, kernel_size=5, padding=2, stride=1
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)
        
        self.lmbda = [0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.001, 0.0003]  # mxh add from HUAWEI CVPR2021 Gained...

        # Condition on Latent y, so the gain vector length M
        # e.g.: self.levels = 6 means we have 6 pairs gain vectors corresponding to 6 level RD performance
        # treat all channels the same in initialization
        # 创建了一个形状为 [self.levels, M] 的张量，并将其包装在 torch.nn.Parameter 类中，表示它是一个模型参数，需要进行训练。
        # 参数 requires_grad=True 表示需要计算该参数的梯度，并在反向传播时更新该参数的值。
        # M:通道数
        self.levels = len(self.lmbda) # 8
        self.Gain = torch.nn.Parameter(torch.ones(size=[self.levels, M]), requires_grad=True)
        self.InverseGain = torch.nn.Parameter(torch.ones(size=[self.levels, M]), requires_grad=True)
        self.HyperGain = torch.nn.Parameter(torch.ones(size=[self.levels, N]), requires_grad=True)
        self.InverseHyperGain = torch.nn.Parameter(torch.ones(size=[self.levels, N]), requires_grad=True)

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

        
    def g_a_higher(self, x, x_size=None):
            # print(x.shape, 'x.shape in g_a_higher')
            # 输入x的大小
            if x_size is None:
                x_size = x.shape[2:4]
            
            x = self.g_a_high_0(x)
            # print(x.shape, 'x.shape in g_a_high_0')
            x = self.g_a_high_1(x)
            # print(x.shape, 'x.shape in g_a_high_1')
            x = self.g_a_high_2(x)
            # print(x.shape, 'x.shape in g_a_high_2')
            x = self.g_a_high_3(x, (x_size[0]//4, x_size[1]//4))
            # print(x.shape, 'x.shape in g_a_high_3')
            x = self.g_a_high_4(x)
            # print(x.shape, 'x.shape in g_a_high_4')
            x = self.g_a_high_5(x)
            # print(x.shape, 'x.shape in g_a_high_5')
            x = self.g_a_high_6(x)
            # print(x.shape, 'x.shape in g_a_high_6')
            return x
        
    def g_a_lower(self, x, x_size=None):
            x = self.g_a_low_0(x)
            x = self.g_a_low_1(x)
            x = self.g_a_low_2(x)
            x = self.g_a_low_3(x)
            x = self.g_a_low_4(x)
            x = self.g_a_low_5(x)
            x = self.g_a_low_6(x)
            return x
        
    def g_s_higher(self, x, x_size=None):
            # print(x.shape, 'x.shape in g_s_higher')
            if x_size is None:
                # x_size = (x.shape[2]*16, x.shape[3]*16)
                x_size = (x.shape[2]*16, x.shape[3]*16)
                
            x = self.g_s_high_0(x)
            # print(x.shape, 'x.shape in g_s_high_0')
            x = self.g_s_high_1(x)
            # print(x.shape, 'x.shape in g_s_high_1')
            x = self.g_s_high_2(x)
            # print(x.shape, 'x.shape in g_s_high_2')
            x = self.g_s_high_3(x, (x_size[0]//4, x_size[1]//4))
            # print(x.shape, 'x.shape in g_s_high_3')
            x = self.g_s_high_4(x)
            # print(x.shape, 'x.shape in g_s_high_4')
            x = self.g_s_high_5(x)
            # print(x.shape, 'x.shape in g_s_high_5')
            x = self.g_s_high_6(x)
            # print(x.shape, 'x.shape in g_s_high_6')
            return x
    
    def g_s_lower(self, x, x_size=None):
            x = self.g_s_low_0(x)
            x = self.g_s_low_1(x)
            x = self.g_s_low_2(x)
            x = self.g_s_low_3(x)
            x = self.g_s_low_4(x)
            x = self.g_s_low_5(x)
            x = self.g_s_low_6(x)
            return x
    
    def forward(self, x, s):
        '''
            x: input image
            s: random num to choose gain vector
        '''
        # print(x.shape, 'x.shape in network forward')
        x_size = (x.shape[2], x.shape[3])
        if s >= 4: # 低码率点
            g_a_method = self.g_a_lower
            g_s_method = self.g_s_lower
        else: # 高码率点
            g_a_method = self.g_a_higher
            g_s_method = self.g_s_higher
            
        # ga->Gain->ha->HyperGain->InverseHyperGain->hs->gaussian_params->InverseGain->g_s
        y = g_a_method(x, x_size)
        
        # Gain
        y = y * torch.abs(self.Gain[s]).unsqueeze(0).unsqueeze(2).unsqueeze(3) # Gain[s]: [M]  -->  [1,M,1,1]
        z = self.h_a(y)
        # HyperGain
        z = z * torch.abs(self.HyperGain[s]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        # InverseHyperGain
        z_hat = z_hat * torch.abs(self.InverseHyperGain[s]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        params = self.h_s(z_hat)
        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )
        ctx_params = self.context_prediction(y_hat) # φ
        # 混合高斯模型
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        # 方差、均值
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        # InverseGain
        y_hat = y_hat * torch.abs(self.InverseGain[s]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        x_hat = g_s_method(y_hat)
        
        
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }
        
    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x, s, l):
        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )

        assert s in range(0,self.levels-1), f"s should in range(0, {self.levels-1}), but get s:{s}"
        assert l >=0 and l <=1, "l should in [0,1]"
        
        # l = 0, Interpolated = s+1; l = 1, Interpolated = s
        # 计算l实现连续码率可变
        InterpolatedGain = torch.abs(self.Gain[s]).pow(1-l) * torch.abs(self.Gain[s+1]).pow(l)
        InterpolatedHyperGain = torch.abs(self.HyperGain[s]).pow(1-l) * torch.abs(self.HyperGain[s+1]).pow(l)
        InterpolatedInverseHyperGain = torch.abs(self.InverseHyperGain[s]).pow(1-l) * torch.abs(self.InverseHyperGain[s+1]).pow(l)

        # InterpolatedGain = torch.abs(self.Gain[s]) * (1 - l) + torch.abs(self.Gain[s + 1]) * l
        # InterpolatedHyperGain = torch.abs(self.HyperGain[s]) * (1 - l) + torch.abs(self.HyperGain[s + 1]) * l
        # InterpolatedInverseHyperGain = torch.abs(self.InverseHyperGain[s]) * (1 - l) + torch.abs(self.InverseHyperGain[s + 1]) * (l)
        
        
        if s >= 4: # 低码率点
            g_a_method = self.g_a_lower
            g_s_method = self.g_s_lower
        else: # 高码率点
            g_a_method = self.g_a_higher
            g_s_method = self.g_s_higher
        
        y = g_a_method(x)
        # 记录没有改变的y
        ungained_y = y
        # InterpolatedGain
        y = y * InterpolatedGain.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        
        z = self.h_a(y)
        # InterpolatedHyperGain
        z = z * InterpolatedHyperGain.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        # InterpolatedInverseHyperGain
        z_hat = z_hat * InterpolatedInverseHyperGain.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        y_hat = F.pad(y, (padding, padding, padding, padding))

        y_strings = []
        for i in range(y.size(0)):
            string = self._compress_ar(
                y_hat[i : i + 1],
                params[i : i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )
            y_strings.append(string)

        gained_y_hat = self.gaussian_conditional.quantize(y, "symbols")
        return {"strings": [y_strings, z_strings], 
                "shape": z.size()[-2:],
                "gained_y_hat": gained_y_hat}

    def _compress_ar(self, y_hat, params, height, width, kernel_size, padding):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []

        # Warning, this is slow...
        # TODO: profile the calls to the bindings...
        masked_weight = self.context_prediction.weight * self.context_prediction.mask
        for h in range(height):
            for w in range(width):
                y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    masked_weight,
                    bias=self.context_prediction.bias,
                )

                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h : h + 1, w : w + 1]
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                gaussian_params = gaussian_params.squeeze(3).squeeze(2)
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.gaussian_conditional.build_indexes(scales_hat)

                y_crop = y_crop[:, :, padding, padding]
                y_q = self.gaussian_conditional.quantize(y_crop, "symbols", means_hat)
                y_hat[:, :, h + padding, w + padding] = y_q + means_hat

                symbols_list.extend(y_q.squeeze().tolist())
                indexes_list.extend(indexes.squeeze().tolist())

        encoder.encode_with_indexes(
            symbols_list, indexes_list, cdf, cdf_lengths, offsets
        )

        string = encoder.flush()
        return string

    def decompress(self, strings, shape, s, l):
        assert isinstance(strings, list) and len(strings) == 2

        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )

        # FIXME: we don't respect the default entropy coder and directly call the
        # range ANS decoder

        assert s in range(0, self.levels - 1), f"s should in range(0,{self.levels - 1})"
        assert l >= 0 and l <= 1, "l should in [0,1]"
        InterpolatedInverseGain = torch.abs(self.InverseGain[s]).pow(1-l) * torch.abs(self.InverseGain[s+1]).pow(l)
        InterpolatedInverseHyperGain = torch.abs(self.InverseHyperGain[s]).pow(1-l) * torch.abs(self.InverseHyperGain[s+1]).pow(l)
        # # Linear Interpolation can achieve the same result
        # InterpolatedInverseGain = torch.abs(self.InverseGain[s]) * (1 - l) + torch.abs(self.InverseGain[s + 1]) * (l)
        # InterpolatedInverseHyperGain = torch.abs(self.InverseHyperGain[s])* (1 - l) + torch.abs(self.InverseHyperGain[s + 1]) * (l)
        
        if s >= 4: # 低码率点
            g_a_method = self.g_a_lower
            g_s_method = self.g_s_lower
        else: # 高码率点
            g_a_method = self.g_a_higher
            g_s_method = self.g_s_higher
        
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        # InterpolatedInverseHyperGain
        z_hat = z_hat * InterpolatedInverseHyperGain.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        # initialize y_hat to zeros, and pad it so we can directly work with
        # sub-tensors of size (N, C, kernel size, kernel_size)
        y_hat = torch.zeros(
            (z_hat.size(0), self.M, y_height + 2 * padding, y_width + 2 * padding),
            device=z_hat.device,
        )
        

        for i, y_string in enumerate(strings[0]):
            self._decompress_ar(
                y_string,
                y_hat[i : i + 1],
                params[i : i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )

        y_hat = F.pad(y_hat, (-padding, -padding, -padding, -padding))
        gained_y_hat = y_hat
        # InterpolatedInverseGain
        y_hat = y_hat * InterpolatedInverseGain.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        x_hat = g_s_method(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat, "gained_y_hat": gained_y_hat}

    def _decompress_ar(
        self, y_string, y_hat, params, height, width, kernel_size, padding
    ):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        # Warning: this is slow due to the auto-regressive nature of the
        # decoding... See more recent publication where they use an
        # auto-regressive module on chunks of channels for faster decoding...
        for h in range(height):
            for w in range(width):
                # only perform the 5x5 convolution on a cropped tensor
                # centered in (h, w)
                y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    self.context_prediction.weight,
                    bias=self.context_prediction.bias,
                )
                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h : h + 1, w : w + 1]
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.gaussian_conditional.build_indexes(scales_hat)
                rv = decoder.decode_stream(
                    indexes.squeeze().tolist(), cdf, cdf_lengths, offsets
                )
                rv = torch.Tensor(rv).reshape(1, -1, 1, 1)
                rv = self.gaussian_conditional.dequantize(rv, means_hat)

                hp = h + padding
                wp = w + padding
                y_hat[:, :, hp : hp + 1, wp : wp + 1] = rv
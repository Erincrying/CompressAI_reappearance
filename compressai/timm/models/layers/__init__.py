# NOTE timm.models.layers is DEPRECATED, please use timm.layers, this is here to reduce breakages in transition
from compressai.timm.layers.activations import *
from compressai.timm.layers.adaptive_avgmax_pool import \
    adaptive_avgmax_pool2d, select_adaptive_pool2d, AdaptiveAvgMaxPool2d, SelectAdaptivePool2d
from compressai.timm.layers.attention_pool2d import AttentionPool2d, RotAttentionPool2d, RotaryEmbedding
from compressai.timm.layers.blur_pool import BlurPool2d
from compressai.timm.layers.classifier import ClassifierHead, create_classifier
from compressai.timm.layers.cond_conv2d import CondConv2d, get_condconv_initializer
from compressai.timm.layers.config import is_exportable, is_scriptable, is_no_jit, set_exportable, set_scriptable, set_no_jit,\
    set_layer_config
from compressai.timm.layers.conv2d_same import Conv2dSame, conv2d_same
from compressai.timm.layers.conv_bn_act import ConvNormAct, ConvNormActAa, ConvBnAct
from compressai.timm.layers.create_act import create_act_layer, get_act_layer, get_act_fn
from compressai.timm.layers.create_attn import get_attn, create_attn
from compressai.timm.layers.create_conv2d import create_conv2d
from compressai.timm.layers.create_norm import get_norm_layer, create_norm_layer
from compressai.timm.layers.create_norm_act import get_norm_act_layer, create_norm_act_layer, get_norm_act_layer
from compressai.timm.layers.drop import DropBlock2d, DropPath, drop_block_2d, drop_path
from compressai.timm.layers.eca import EcaModule, CecaModule, EfficientChannelAttn, CircularEfficientChannelAttn
from compressai.timm.layers.evo_norm import EvoNorm2dB0, EvoNorm2dB1, EvoNorm2dB2,\
    EvoNorm2dS0, EvoNorm2dS0a, EvoNorm2dS1, EvoNorm2dS1a, EvoNorm2dS2, EvoNorm2dS2a
from compressai.timm.layers.fast_norm import is_fast_norm, set_fast_norm, fast_group_norm, fast_layer_norm
from compressai.timm.layers.filter_response_norm import FilterResponseNormTlu2d, FilterResponseNormAct2d
from compressai.timm.layers.gather_excite import GatherExcite
from compressai.timm.layers.global_context import GlobalContext
from compressai.timm.layers.helpers import to_ntuple, to_2tuple, to_3tuple, to_4tuple, make_divisible, extend_tuple
from compressai.timm.layers.inplace_abn import InplaceAbn
from compressai.timm.layers.linear import Linear
from compressai.timm.layers.mixed_conv2d import MixedConv2d
from compressai.timm.layers.mlp import Mlp, GluMlp, GatedMlp, ConvMlp
from compressai.timm.layers.non_local_attn import NonLocalAttn, BatNonLocalAttn
from compressai.timm.layers.norm import GroupNorm, GroupNorm1, LayerNorm, LayerNorm2d
from compressai.timm.layers.norm_act import BatchNormAct2d, GroupNormAct, convert_sync_batchnorm
from compressai.timm.layers.padding import get_padding, get_same_padding, pad_same
from compressai.timm.layers.patch_embed import PatchEmbed
from compressai.timm.layers.pool2d_same import AvgPool2dSame, create_pool2d
from compressai.timm.layers.squeeze_excite import SEModule, SqueezeExcite, EffectiveSEModule, EffectiveSqueezeExcite
from compressai.timm.layers.selective_kernel import SelectiveKernel
from compressai.timm.layers.separable_conv import SeparableConv2d, SeparableConvNormAct
from compressai.timm.layers.space_to_depth import SpaceToDepthModule
from compressai.timm.layers.split_attn import SplitAttn
from compressai.timm.layers.split_batchnorm import SplitBatchNorm2d, convert_splitbn_model
from compressai.timm.layers.std_conv import StdConv2d, StdConv2dSame, ScaledStdConv2d, ScaledStdConv2dSame
from compressai.timm.layers.test_time_pool import TestTimePoolHead, apply_test_time_pool
from compressai.timm.layers.trace_utils import _assert, _float_to_int
from compressai.timm.layers.weight_init import trunc_normal_, trunc_normal_tf_, variance_scaling_, lecun_normal_

import warnings
warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.layers", DeprecationWarning)

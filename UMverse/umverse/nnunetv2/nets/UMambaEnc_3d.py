import numpy as np
import math
import torch
from torch import nn
from torch.nn import functional as F
from typing import Union, Type, List, Tuple

from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp

from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim

from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from nnunetv2.utilities.network_initialization import InitWeights_He
from .mamba_utils import MambaLayer
from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list, get_matching_pool_op
from torch.cuda.amp import autocast
from dynamic_network_architectures.building_blocks.residual import BasicBlockD
from nnunetv2.DWT_IDWT.DWT_IDWT_Functions import *
from nnunetv2.DWT_IDWT.DWT_IDWT_layer import DWT_3D,IDWT_3D

def up_(x, scale=2):
    return F.interpolate(x, scale_factor=scale, mode='nearest', align_corners=True)

class UpsampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride, wavelet='haar'):
        super().__init__()
        self.conv = nn.Conv3d(in_channels=in_channels+out_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        self.iDWT = IDWT_3D(wavename=wavelet)
        self.DWT = DWT_3D(wavename=wavelet)
        if min(stride) == 2 and max(stride) == 2:
            self.if_idwt = True
        else:
            self.if_idwt = False

    def forward(self, y, x):
        if self.if_idwt:
            # y是cat后输出的tensor
            LLL_, LLH_, LHL_, LHH_, HLL_, HLH_, HHL_, HHH_ = self.DWT(y)

            x = self.conv(torch.cat((x, LLL_),dim=1))
            x = self.iDWT(x, LLH_, LHL_, LHH_, HLL_, HLH_, HHL_, HHH_)
        else:
            x = self.conv(torch.cat((x, LLL_),dim=1))
        return x


class Norm_UpsampleLayer(nn.Module):
    def __init__(
            self,
            conv_op,
            input_channels,
            output_channels,
            pool_op_kernel_size,
            mode='nearest'
    ):
        super().__init__()
        self.conv = conv_op(input_channels, output_channels, kernel_size=1)
        self.pool_op_kernel_size = pool_op_kernel_size
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.pool_op_kernel_size, mode=self.mode)
        x = self.conv(x)
        return x

class DownsampleLayer(nn.Module):
    def __init__(self, stride,input_channels, output_channels, wavelet='haar'):
        super().__init__()
        if min(stride) == 2 and max(stride) == 2:
            self.if_dwt = True
        else:
            self.if_dwt = False
        self.DWT = DWT_3D(wavename=wavelet)

    def forward(self, x):
        if self.if_dwt:
            #x = self.conv(x)
            LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = self.DWT(x)
            HF = torch.cat((LLH, LHL, LHH, HLL, HLH, HHL, HHH),dim=1)
            LF = LLL
            return HF, LF
        else:
            return None, x

class EnhanceLayer(nn.Module):
    def __init__(self, channel, stage):
        super().__init__()
        self.stage = stage
        self.group = 2 ** self.stage
        # 3D Convolution Layers
        self.conv_left = nn.Conv3d(7*self.group , channel, kernel_size=1, stride=1, padding=0)
        self.conv_right = nn.Conv3d(7*self.group , channel, kernel_size=1, stride=1, padding=0)
        self.conv_combine = nn.Conv3d(2 * channel, channel, kernel_size=1, stride=1, padding=0)

        # Intermediate Convolution Layers
        self.conv1_left = nn.Conv3d(7*self.group, 7*self.group, kernel_size=1, stride=1, padding=0)
        self.conv1_right = nn.Conv3d(7*self.group, 7*self.group, kernel_size=1, stride=1, padding=0)

        # Activation functions
        self.sigmoid_ = nn.Sigmoid()
        self.softmax_ = nn.Softmax(dim=1)

    def forward(self, HF):
        b, c, h, w, d = HF.shape
        # Global Average Pooling (GAP) and Global Max Pooling (GMP)
        LLH, LHL, LHH, HLL, HLH, HHL, HHH = torch.split(HF, c//7,1)
        x_gap =torch.cat((self.gap(LLH), self.gap(LHL), self.gap(LHH), self.gap(HLL), self.gap(HLH), self.gap(HHL), self.gap(HHH)),dim=1)
        x_gmp = torch.cat((self.gmp(LLH), self.gmp(LHL), self.gmp(LHH), self.gmp(HLL), self.gmp(HLH), self.gmp(HHL), self.gmp(HHH)),dim=1)

        # Left Path
        left_path = self.conv1_left(x_gap)  # Conv 7 * H * W * D
        left_path_left = self.softmax_(left_path)  # Softmax
        left_path = left_path * left_path_left  # Element-wise multiplication
        left_path = self.conv_left(left_path)  # Conv to C * H * W * D
        left_path = self.sigmoid_(left_path)
        # Right Path
        right_path = self.conv1_right(x_gmp)  # Conv 7 * H * W * D
        right_path_right = self.softmax_(right_path)  # Softmax
        right_path = right_path * right_path_right  # Element-wise multiplication
        right_path = self.conv_right(right_path)  # Conv to C * H * W * D
        right_path = self.sigmoid_(right_path)

        # Combine Paths
        middel = LLH + LHL + LHH + HLL + HLH + HHL + HHH  # 7C * H * W * D to C * H * W * D
        combined = torch.cat((middel*left_path, middel*right_path), dim=1)  # Concatenate to 2C * H * W * D
        combined = self.conv_combine(combined)  # Conv to C * H * W * D

        return combined


    def gap(self, x):

        x_reshaped = x.view(x.shape[0], self.group, x.shape[1] // self.group, *x.shape[2:])
        x_pooled = torch.mean(x_reshaped, dim=2, keepdim=True)
        x_out = x_pooled.view(x.shape[0], -1, *x_pooled.shape[3:])
        return x_out

    def gmp(self, x):

        x_reshaped = x.view(x.shape[0], self.group, x.shape[1] // self.group, *x.shape[2:])
        x_pooled = torch.max(x_reshaped, dim=2, keepdim=True)[0]
        x_out = x_pooled.view(x.shape[0], -1, *x_pooled.shape[3:])
        return x_out

class BasicResBlock(nn.Module):
    def __init__(
            self,
            conv_op,
            input_channels,
            output_channels,
            norm_op,
            norm_op_kwargs,
            kernel_size=3,
            padding=1,
            stride=1,
            use_1x1conv=False,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={'inplace': True}
    ):
        super().__init__()

        self.conv1 = conv_op(input_channels, output_channels, kernel_size, stride=stride, padding=padding)
        self.norm1 = norm_op(output_channels, **norm_op_kwargs)
        self.act1 = nonlin(**nonlin_kwargs)

        self.conv2 = conv_op(output_channels, output_channels, kernel_size, padding=padding)
        self.norm2 = norm_op(output_channels, **norm_op_kwargs)
        self.act2 = nonlin(**nonlin_kwargs)

        if use_1x1conv:
            self.conv3 = conv_op(input_channels, output_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None

    def forward(self, x):
        y = self.conv1(x)
        y = self.act1(self.norm1(y))
        y = self.norm2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        return self.act2(y)


class ResidualMambaEncoder(nn.Module):
    def __init__(self,
                 input_size: Tuple[int, ...],
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...], Tuple[Tuple[int, ...], ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 return_skips: bool = False,
                 stem_channels: int = None,
                 pool_type: str = 'conv',
                 ):
        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
        assert len(
            kernel_sizes) == n_stages, "kernel_sizes must have as many entries as we have resolution stages (n_stages)"
        assert len(
            n_blocks_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(
            features_per_stage) == n_stages, "features_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(strides) == n_stages, "strides must have as many entries as we have resolution stages (n_stages). " \
                                         "Important: first entry is recommended to be 1, else we run strided conv drectly on the input"

        pool_op = get_matching_pool_op(conv_op, pool_type=pool_type) if pool_type != 'conv' else None

        do_channel_token = [False] * n_stages
        feature_map_sizes = []
        feature_map_size = input_size
        for s in range(n_stages):
            feature_map_sizes.append([i // j for i, j in zip(feature_map_size, strides[s])])
            feature_map_size = feature_map_sizes[-1]
            if np.prod(feature_map_size) <= features_per_stage[s]:
                do_channel_token[s] = True

        print(f"feature_map_sizes: {feature_map_sizes}")
        print(f"do_channel_token: {do_channel_token}")
        print(f"features_per_stage: {features_per_stage}")

        self.conv_pad_sizes = []
        for krnl in kernel_sizes:
            self.conv_pad_sizes.append([i // 2 for i in krnl])

        stem_channels = features_per_stage[0]
        self.stem = nn.Sequential(
            BasicResBlock(
                conv_op=conv_op,
                input_channels=input_channels,
                output_channels=stem_channels,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                kernel_size=kernel_sizes[0],
                padding=self.conv_pad_sizes[0],
                stride=1,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
                use_1x1conv=True
            ),
            *[
                BasicBlockD(
                    conv_op=conv_op,
                    input_channels=stem_channels,
                    output_channels=stem_channels,
                    kernel_size=kernel_sizes[0],
                    stride=1,
                    conv_bias=conv_bias,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                ) for _ in range(n_blocks_per_stage[0] - 1)
            ]
        )

        input_channels = stem_channels
        output_channels = []
        stages = []
        mamba_layers = []
        downsamples = []
        enhancers = []
        for s in range(n_stages):
            stage = nn.Sequential(
                BasicResBlock(
                    conv_op=conv_op,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    input_channels=input_channels,
                    output_channels=features_per_stage[s],
                    kernel_size=kernel_sizes[s],
                    padding=self.conv_pad_sizes[s],
                    stride=strides[s] if do_channel_token[s] else 1,
                    use_1x1conv=True,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs
                ),
                *[
                    BasicBlockD(
                        conv_op=conv_op,
                        input_channels=features_per_stage[s],
                        output_channels=features_per_stage[s],
                        kernel_size=kernel_sizes[s],
                        stride=1,
                        conv_bias=conv_bias,
                        norm_op=norm_op,
                        norm_op_kwargs=norm_op_kwargs,
                        nonlin=nonlin,
                        nonlin_kwargs=nonlin_kwargs,
                    ) for _ in range(n_blocks_per_stage[s] - 1)
                ]
            )
            downsamples.append(DownsampleLayer(stride=strides[s],input_channels=input_channels,
                    output_channels=features_per_stage[s]))

            enhancers.append(EnhanceLayer(channel=features_per_stage[s], stage=s))
            mamba_layers.append(
                MambaLayer(
                    channel=features_per_stage[s],
                    dim=np.prod(feature_map_sizes[s]) if do_channel_token[s] else features_per_stage[s],
                    channel_token=do_channel_token[s]
                )
            )

            stages.append(stage)
            input_channels = features_per_stage[s] if max(strides[s]) == 1 or do_channel_token[s] else features_per_stage[s]*2
            output_channels.append(input_channels)


        self.mamba_layers = nn.ModuleList(mamba_layers)
        self.enhancers = nn.ModuleList(enhancers)
        self.stages = nn.ModuleList(stages)
        self.downsamples = nn.ModuleList(downsamples)
        self.output_channels = output_channels
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips

        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes
        self.do_norm_downsample = do_channel_token

    def forward(self, x):
        if self.stem is not None:
            x = self.stem(x)
        ret = []
        for s in range(len(self.stages)):
            if self.do_norm_downsample[s]:
                x = self.stages[s](x)
                x = self.mamba_layers[s](x)
            else:
                x = self.stages[s](x)
                HF, LF = self.downsamples[s](x)
                if HF is not None:
                    HF = self.enhancers[s](HF)
                    LF = self.mamba_layers[s](LF)
                    x = torch.cat((HF, LF), dim=1)
                else:
                    x = self.mamba_layers[s](x)

            ret.append(x)
        if self.return_skips:
            return ret
        else:
            return ret[-1]

    def compute_conv_feature_map_size(self, input_size):
        if self.stem is not None:
            output = self.stem.compute_conv_feature_map_size(input_size)
        else:
            output = np.int64(0)

        for s in range(len(self.stages)):
            output += self.stages[s].compute_conv_feature_map_size(input_size)
            input_size = [i // j for i, j in zip(input_size, self.strides[s])]

        return output


class UNetResDecoder(nn.Module):
    def __init__(self,
                 encoder,
                 num_classes,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision, nonlin_first: bool = False):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.norm_upsample = encoder.do_norm_downsample
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1, "n_conv_per_stage must have as many entries as we have " \
                                                          "resolution stages - 1 (n_stages in encoder - 1), " \
                                                          "here: %d" % n_stages_encoder

        stages = []

        upsample_layers = []

        norm_upsample_layers = []

        seg_layers = []
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_upsampling = encoder.strides[-s]

            upsample_layers.append(UpsampleLayer(
                in_channels = input_features_below,
                out_channels = input_features_skip,
                stride = stride_for_upsampling
            ))

            norm_upsample_layers.append(Norm_UpsampleLayer(
                conv_op = encoder.conv_op,
                input_channels = input_features_below,
                output_channels = input_features_skip,
                pool_op_kernel_size = stride_for_upsampling,
                mode='nearest'
            ))

            stages.append(nn.Sequential(
                BasicResBlock(
                    conv_op = encoder.conv_op,
                    norm_op = encoder.norm_op,
                    norm_op_kwargs = encoder.norm_op_kwargs,
                    nonlin = encoder.nonlin,
                    nonlin_kwargs = encoder.nonlin_kwargs,
                    input_channels = input_features_skip*2 if self.norm_upsample[-(s)] else input_features_skip,  ########################$$$$$$$$$$$$$$$$$$$$$$$$
                    output_channels = input_features_skip,
                    kernel_size = encoder.kernel_sizes[-(s + 1)],
                    padding=encoder.conv_pad_sizes[-(s + 1)],
                    stride=1,
                    use_1x1conv=True
                ),
                *[
                    BasicBlockD(
                        conv_op = encoder.conv_op,
                        input_channels = input_features_skip,
                        output_channels = input_features_skip,
                        kernel_size = encoder.kernel_sizes[-(s + 1)],
                        stride = 1,
                        conv_bias = encoder.conv_bias,
                        norm_op = encoder.norm_op,
                        norm_op_kwargs = encoder.norm_op_kwargs,
                        nonlin = encoder.nonlin,
                        nonlin_kwargs = encoder.nonlin_kwargs,
                    ) for _ in range(n_conv_per_stage[s-1] - 1)
                ]
            ))
            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.upsample_layers = nn.ModuleList(upsample_layers)
        self.norm_upsample_layers = nn.ModuleList(norm_upsample_layers)
        self.seg_layers = nn.ModuleList(seg_layers)


    def forward(self, skips):
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            if self.norm_upsample[-(s + 1)]:
                x = self.norm_upsample_layers[s](lres_input)
                x = torch.cat((x, skips[-(s + 2)]), 1)
                x = self.stages[s](x)
            else:
                x = self.upsample_layers[s](skips[-(s + 2)], lres_input)
                x = self.stages[s](x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x


        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r

    def compute_conv_feature_map_size(self, input_size):
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]

        assert len(skip_sizes) == len(self.stages)

        output = np.int64(0)
        for s in range(len(self.stages)):
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s+1)])
            output += np.prod([self.encoder.output_channels[-(s+2)], *skip_sizes[-(s+1)]], dtype=np.int64)
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += np.prod([self.num_classes, *skip_sizes[-(s+1)]], dtype=np.int64)
        return output

class UMambaEnc(nn.Module):
    def __init__(self,
                 input_size: Tuple[int, ...],
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 stem_channels: int = None
                 ):
        super().__init__()
        n_blocks_per_stage = n_conv_per_stage
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)

        for s in range(math.ceil(n_stages / 2), n_stages):
            n_blocks_per_stage[s] = 1

        for s in range(math.ceil((n_stages - 1) / 2 + 0.5), n_stages - 1):
            n_conv_per_stage_decoder[s] = 1

        assert len(n_blocks_per_stage) == n_stages, "n_blocks_per_stage must have as many entries as we have " \
                                                    f"resolution stages. here: {n_stages}. " \
                                                    f"n_blocks_per_stage: {n_blocks_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        self.encoder = ResidualMambaEncoder(
            input_size,
            input_channels,
            n_stages,
            features_per_stage,
            conv_op,
            kernel_sizes,
            strides,
            n_blocks_per_stage,
            conv_bias,
            norm_op,
            norm_op_kwargs,
            nonlin,
            nonlin_kwargs,
            return_skips=True,
            stem_channels=stem_channels
        )

        self.decoder = UNetResDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision)

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(
            self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                   "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                   "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(
            input_size)


def get_umamba_enc_3d_from_plans(
        plans_manager: PlansManager,
        dataset_json: dict,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        deep_supervision: bool = True
):
    """
    we may have to change this in the future to accommodate other plans -> network mappings

    num_input_channels can differ depending on whether we do cascade. Its best to make this info available in the
    trainer rather than inferring it again from the plans here.
    """
    num_stages = len(configuration_manager.conv_kernel_sizes)

    dim = len(configuration_manager.conv_kernel_sizes[0])
    conv_op = convert_dim_to_conv_op(dim)

    label_manager = plans_manager.get_label_manager(dataset_json)

    segmentation_network_class_name = 'UMambaEnc'
    network_class = UMambaEnc
    kwargs = {
        'UMambaEnc': {
            'input_size': configuration_manager.patch_size,
            'conv_bias': True,
            'norm_op': get_matching_instancenorm(conv_op),
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'dropout_op': None, 'dropout_op_kwargs': None,
            'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
        }
    }

    conv_or_blocks_per_stage = {
        'n_conv_per_stage': configuration_manager.n_conv_per_stage_encoder,
        'n_conv_per_stage_decoder': configuration_manager.n_conv_per_stage_decoder
    }

    model = network_class(
        input_channels=num_input_channels,
        n_stages=num_stages,
        features_per_stage=[min(configuration_manager.UNet_base_num_features * 2 ** i,
                                configuration_manager.unet_max_num_features) for i in range(num_stages)],
        conv_op=conv_op,
        kernel_sizes=configuration_manager.conv_kernel_sizes,
        strides=configuration_manager.pool_op_kernel_sizes,
        num_classes=label_manager.num_segmentation_heads,
        deep_supervision=deep_supervision,
        **conv_or_blocks_per_stage,
        **kwargs[segmentation_network_class_name]
    )
    model.apply(InitWeights_He(1e-2))

    return model

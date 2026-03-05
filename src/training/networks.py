# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import copy
from einops import rearrange
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from omegaconf import OmegaConf

from src.torch_utils import misc
from src.torch_utils import persistence
from src.torch_utils.ops import conv2d_resample, upfirdn2d, bias_act, fma
from src.torch_utils3.ops import filtered_lrelu

from training.motion import MotionMappingNetwork, BSplineMotionMappingNetwork
from training.layers import (
    FullyConnectedLayer,
    GenInput,
    EqLRConv1d,
    TemporalDifferenceEncoder,
    Conv2dLayer,
    MappingNetwork,
)
from training.networks_stylegan3 import MappingNetwork as MappingNetwork3
from training.networks_stylegan3 import FullyConnectedLayer as FullyConnectedLayer3
from training.networks_stylegan3 import (
    SynthesisNetwork,
    SynthesisInput,
    modulated_conv2d,
    SynthesisLayer
)

# from training.networks_deform import MappingNetwork3 as MappingNetwork3Deform
# from training.networks_deform import SynthesisNetwork3v as SynthesisNetwork3vDeform

from src.training.networks_stylegan_v import MappingNetwork as StyleGAN2MappingNetwork
from src.training.networks_stylegan_v import StyleGAN2SynthesisNetwork


# import src.midas.utils as midas_utils

# from imutils.video import VideoStream
# from src.midas.model_loader import default_models, load_model


def generate_coords(batch_size: int, img_size: int, device='cuda', align_corners: bool = False) -> Tensor:
    """
    Generates the coordinates in [-1, 1] range for a square image
    if size (img_size x img_size) in such a way that
    - upper left corner: coords[0, 0] = (-1, -1)
    - upper right corner: coords[img_size - 1, img_size - 1] = (1, 1)
    """
    if align_corners:
        row = torch.linspace(-1, 1, img_size, device=device).float()  # [img_size]
    else:
        row = (torch.arange(0, img_size, device=device).float() / img_size) * 2 - 1  # [img_size]
    x_coords = row.view(1, -1).repeat(img_size, 1)  # [img_size, img_size]
    y_coords = x_coords.t().flip(dims=(0,))  # [img_size, img_size]

    coords = torch.stack([x_coords, y_coords], dim=2)  # [img_size, img_size, 2]
    coords = coords.view(-1, 2)  # [img_size ** 2, 2]
    coords = coords.t().view(1, 2, img_size, img_size).repeat(batch_size, 1, 1, 1)  # [batch_size, 2, img_size, img_size]

    return coords


#----------------------------------------------------------------------------

# @persistence.persistent_class
# class ToRGBLayer(torch.nn.Module):
#     def __init__(self, in_channels, out_channels, w_dim, kernel_size=1, conv_clamp=None, channels_last=False):
#         super().__init__()
#         self.conv_clamp = conv_clamp
#         self.affine = FullyConnectedLayer3(w_dim, in_channels, bias_init=1)
#         memory_format = torch.channels_last if channels_last else torch.contiguous_format
#         self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format))
#         self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
#         self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))

#     def forward(self, x, w, fused_modconv=True):
#         styles = self.affine(w) * self.weight_gain
#         x = modulated_conv2d(x=x, weight=self.weight, styles=styles, demodulate=False, fused_modconv=fused_modconv)
#         x = bias_act.bias_act(x, self.bias.to(x.dtype), clamp=self.conv_clamp)
#         return x

# #----------------------------------------------------------------------------

# @persistence.persistent_class
# class SynthesisBlock(torch.nn.Module):
#     def __init__(self,
#         in_channels,                        # Number of input channels, 0 = first block.
#         out_channels,                       # Number of output channels.
#         w_dim,                              # Intermediate latent (W) dimensionality.
#         motion_v_dim,                       # Motion code size
#         resolution,                         # Resolution of this block.
#         img_channels,                       # Number of output color channels.
#         is_last,                            # Is this the last block?
#         architecture        = 'skip',       # Architecture: 'orig', 'skip', 'resnet'.
#         resample_filter     = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
#         conv_clamp          = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
#         use_fp16            = False,        # Use FP16 for this block?
#         fp16_channels_last  = False,        # Use channels-last memory format with FP16?
#         cfg                 = {},           # Additional config
#         **layer_kwargs,                     # Arguments for SynthesisLayer.
#     ):
#         assert architecture in ['orig', 'skip', 'resnet']
#         super().__init__()

#         self.cfg = cfg
#         self.in_channels = in_channels
#         self.w_dim = w_dim
#         self.resolution = resolution
#         self.img_channels = img_channels
#         self.is_last = is_last
#         self.architecture = architecture
#         self.use_fp16 = use_fp16
#         self.channels_last = (use_fp16 and fp16_channels_last)
#         self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
#         self.num_conv = 0
#         self.num_torgb = 0

#         if in_channels == 0:
#             self.input = GenInput(self.cfg, out_channels, motion_v_dim=motion_v_dim)
#             conv1_in_channels = self.input.total_dim
#         else:
#             self.conv0 = SynthesisLayer(in_channels, out_channels, w_dim=w_dim, resolution=self.resolution, up=2,
#                 resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=self.channels_last,
#                 kernel_size=3, cfg=cfg, **layer_kwargs)
#             self.num_conv += 1
#             conv1_in_channels = out_channels

#         self.conv1 = SynthesisLayer(conv1_in_channels, out_channels, w_dim=w_dim, resolution=self.resolution,
#             conv_clamp=conv_clamp, channels_last=self.channels_last, kernel_size=3, cfg=cfg, **layer_kwargs)
#         self.num_conv += 1

#         if is_last or architecture == 'skip':
#             self.torgb = ToRGBLayer(out_channels, img_channels, w_dim=w_dim,
#                 conv_clamp=conv_clamp, channels_last=self.channels_last)
#             self.num_torgb += 1

#         if in_channels != 0 and architecture == 'resnet':
#             self.skip = Conv2dLayer(in_channels, out_channels, kernel_size=1, bias=False, up=2,
#                 resample_filter=resample_filter, channels_last=self.channels_last)

#     def forward(self, x, img, ws, motion_v=None, force_fp32=False, fused_modconv=None, **layer_kwargs):
#         misc.assert_shape(ws, [None, self.num_conv + self.num_torgb, self.w_dim])
#         w_iter = iter(ws.unbind(dim=1))
#         dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
#         memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format

#         if fused_modconv is None:
#             with misc.suppress_tracer_warnings(): # this value will be treated as a constant
#                 fused_modconv = (not self.training) and (dtype == torch.float32 or (isinstance(x, Tensor) and int(x.shape[0]) == 1))

#         # Input.
#         if self.in_channels == 0:
#             x = self.input(ws.shape[0], motion_v=motion_v, dtype=dtype, memory_format=memory_format)
#         else:
#             misc.assert_shape(x, [None, self.in_channels, self.resolution // 2, self.resolution // 2])
#             x = x.to(dtype=dtype, memory_format=memory_format)

#         # Main layers.
#         if self.in_channels == 0:
#             x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
#         elif self.architecture == 'resnet':
#             y = self.skip(x, gain=np.sqrt(0.5))
#             x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
#             x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, gain=np.sqrt(0.5), **layer_kwargs)
#             x = y.add_(x)
#         else:
#             conv0_w = next(w_iter)
#             x = self.conv0(x, conv0_w, fused_modconv=fused_modconv, **layer_kwargs)
#             x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)

#         # ToRGB.
#         if img is not None:
#             misc.assert_shape(img, [None, self.img_channels, self.resolution // 2, self.resolution // 2])
#             img = upfirdn2d.upsample2d(img, self.resample_filter)

#         if self.is_last or self.architecture == 'skip':
#             y = self.torgb(x, next(w_iter), fused_modconv=fused_modconv)
#             y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
#             img = img.add_(y) if img is not None else y

#         assert x.dtype == dtype
#         assert img is None or img.dtype == torch.float32
#         return x, img


@persistence.persistent_class
class SynthesisInput3v(SynthesisInput):
    def __init__(self,
        w_dim,          # Intermediate latent (W) dimensionality.
        channels,       # Number of output channels.
        size,           # Output spatial size: int or [width, height].
        sampling_rate,  # Output sampling rate.
        bandwidth,      # Output bandwidth.
        cfg,
    ):
        super().__init__(w_dim, channels, size, sampling_rate, bandwidth)

        # self.conv_kernel = 3
        self.cfg = cfg
        self.motion_affine = FullyConnectedLayer(512, 512, bias_init=1)

    def forward(self, w, motion_v, ti, motion_mask=None):
        # Introduce batch dimension.
        transforms = self.transform.unsqueeze(0) # [batch, row, col]
        freqs = self.freqs.unsqueeze(0) # [batch, channel, xy]
        phases = self.phases.unsqueeze(0) # [batch, channel]

        # Apply learned transformation.
        t = self.affine(w) # t = (r_c, r_s, t_x, t_y)
        t = t / t[:, :2].norm(dim=1, keepdim=True) # t' = (r'_c, r'_s, t'_x, t'_y)
        m_r = torch.eye(3, device=w.device).unsqueeze(0).repeat([w.shape[0], 1, 1]) # Inverse rotation wrt. resulting image.
        m_r[:, 0, 0] = t[:, 0]  # r'_c
        m_r[:, 0, 1] = -t[:, 1] # r'_s
        m_r[:, 1, 0] = t[:, 1]  # r'_s
        m_r[:, 1, 1] = t[:, 0]  # r'_c
        m_t = torch.eye(3, device=w.device).unsqueeze(0).repeat([w.shape[0], 1, 1]) # Inverse translation wrt. resulting image.
        m_t[:, 0, 2] = -t[:, 2] # t'_x
        m_t[:, 1, 2] = -t[:, 3] # t'_y
        transforms = m_r @ m_t @ transforms # First rotate resulting image, then translate, and finally apply user-specified transform.
        phases = phases + (freqs @ transforms[:, :2, 2:]).squeeze(2)
        freqs = freqs @ transforms[:, :2, :2]

        # Dampen out-of-band frequencies that may occur due to the user-specified transform.
        amplitudes = (1 - (freqs.norm(dim=2) - self.bandwidth) / (self.sampling_rate / 2 - self.bandwidth)).clamp(0, 1)

        # Construct sampling grid.
        theta = torch.eye(2, 3, device=w.device)
        theta[0, 0] = 0.5 * self.size[0] / self.sampling_rate
        theta[1, 1] = 0.5 * self.size[1] / self.sampling_rate
        grids = torch.nn.functional.affine_grid(theta.unsqueeze(0), [1, 1, self.size[1], self.size[0]], align_corners=False)
        
        # Compute Fourier features.
        x = (grids.unsqueeze(3) @ freqs.permute(0, 2, 1).unsqueeze(1).unsqueeze(2)).squeeze(3) # [batch, height, width, channel]
        x = x + phases.unsqueeze(1).unsqueeze(2)
        x = torch.sin(x * (np.pi * 2))
        x = x * amplitudes.unsqueeze(1).unsqueeze(2)

        # Apply trainable mapping.
        weight = self.weight / np.sqrt(self.channels)
        x = x @ weight.t()

        # Ensure correct shape.
        x = x.permute(0, 3, 1, 2) # [batch, channel, height, width]
        # Add motion-related feature.
        if motion_mask is not None:
            x += (motion_mask * self.motion_affine(motion_v)).unsqueeze(-1).unsqueeze(-1)
        else:
            x += self.motion_affine(motion_v).unsqueeze(-1).unsqueeze(-1)

        misc.assert_shape(x, [w.shape[0], self.channels, int(self.size[1]), int(self.size[0])])
        return x


@persistence.persistent_class
class SynthesisNetwork3v(SynthesisNetwork):
    def __init__(self,
        w_dim,                          # Intermediate latent (W) dimensionality.
        img_resolution,                 # Output image resolution.
        img_channels,                   # Number of color channels.
        channel_base        = 32768,    # Overall multiplier for the number of channels.
        channel_max         = 512,      # Maximum number of channels in any layer.
        num_layers          = 14,       # Total number of layers, excluding Fourier features and ToRGB.
        num_critical        = 2,        # Number of critically sampled layers at the end.
        first_cutoff        = 2,        # Cutoff frequency of the first layer (f_{c,0}).
        first_stopband      = 2**2.1,   # Minimum stopband of the first layer (f_{t,0}).
        last_stopband_rel   = 2**0.3,   # Minimum stopband of the last layer, expressed relative to the cutoff.
        margin_size         = 10,       # Number of additional pixels outside the image.
        output_scale        = 0.25,     # Scale factor for the output image.
        num_fp16_res        = 4,        # Use FP16 for the N highest resolutions.
        cfg                 = {},       # Additional config
        **layer_kwargs,                 # Arguments for SynthesisLayer.
    ):
        torch.nn.Module.__init__(self)
        self.w_dim = w_dim
        self.num_ws = num_layers + 2
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.num_layers = num_layers
        self.num_critical = num_critical
        self.margin_size = margin_size
        self.output_scale = output_scale
        self.num_fp16_res = num_fp16_res

        # Geometric progression of layer cutoffs and min. stopbands.
        last_cutoff = self.img_resolution / 2 # f_{c,N}
        last_stopband = last_cutoff * last_stopband_rel # f_{t,N}
        exponents = np.minimum(np.arange(self.num_layers + 1) / (self.num_layers - self.num_critical), 1)
        cutoffs = first_cutoff * (last_cutoff / first_cutoff) ** exponents # f_c[i]
        stopbands = first_stopband * (last_stopband / first_stopband) ** exponents # f_t[i]

        # Compute remaining layer parameters.
        sampling_rates = np.exp2(np.ceil(np.log2(np.minimum(stopbands * 2, self.img_resolution)))) # s[i]
        half_widths = np.maximum(stopbands, sampling_rates / 2) - cutoffs # f_h[i]
        sizes = sampling_rates + self.margin_size * 2
        sizes[-2:] = self.img_resolution
        channels = np.rint(np.minimum((channel_base / 2) / cutoffs, channel_max))
        channels[-1] = self.img_channels
        self.channels = channels
        self.sizes = sizes
        self.sampling_rates = sampling_rates
        self.cutoffs = cutoffs

        self.layer_names = []
        for idx in range(self.num_layers + 1):
            prev = max(idx - 1, 0)
            is_torgb = (idx == self.num_layers)
            is_critically_sampled = (idx >= self.num_layers - self.num_critical)
            use_fp16 = (sampling_rates[idx] * (2 ** self.num_fp16_res) > self.img_resolution)
            in_dcn = idx in [int(each) for each in cfg.dcn_lidx.split(',')] if cfg.dcn_lidx != -1 else True
            dcn = cfg.dcn and not is_torgb and (int(sizes[idx]) >= cfg.dcn_min_res and int(sizes[idx]) <= cfg.dcn_max_res) and in_dcn
            layer = SynthesisLayer(
                w_dim=self.w_dim, is_torgb=is_torgb, is_critically_sampled=is_critically_sampled, use_fp16=use_fp16,
                in_channels=int(channels[prev]), out_channels= int(channels[idx]),
                in_size=int(sizes[prev]), out_size=int(sizes[idx]),
                in_sampling_rate=int(sampling_rates[prev]), out_sampling_rate=int(sampling_rates[idx]),
                in_cutoff=cutoffs[prev], out_cutoff=cutoffs[idx],
                in_half_width=half_widths[prev], out_half_width=half_widths[idx],
                low_rank=cfg.low_rank,
                trainable = idx < (self.num_layers + 1 - cfg.freezesyn),
                dcn=dcn,
                dcn_lr_mult=cfg.dcn_lr_mult,
                **layer_kwargs)
            name = f'L{idx}_{layer.out_size[0]}_{layer.out_channels}'
            setattr(self, name, layer)
            self.layer_names.append(name)
        print(f"layer_names: {self.layer_names}")
        self.cfg = cfg

        if self.cfg.bspline:
            self.motion_encoder = BSplineMotionMappingNetwork(self.cfg)
        else:
            self.motion_encoder = MotionMappingNetwork(self.cfg)
        self.motion_v_dim = self.motion_encoder.get_dim()

        self.input = SynthesisInput3v(
            w_dim=self.w_dim, channels=int(self.channels[0]), size=int(self.sizes[0]),
            sampling_rate=self.sampling_rates[0], bandwidth=self.cutoffs[0], cfg=cfg)
        if self.cfg.fuse_w == 'concat':
            self.affine_w = FullyConnectedLayer(1024, 512, bias_init=1)
        
        init_motion_mask = cfg.init_motion_mask
        if init_motion_mask == 'zeros':
            motion_mask = torch.zeros([1, 512])
        elif init_motion_mask == 'ones':
            motion_mask = torch.ones([1, 512])
        else:
            raise ValueError

        if self.cfg.learnable_motion_mask:
            self.motion_mask = torch.nn.Parameter(motion_mask) 
        else:
            self.register_buffer('motion_mask', motion_mask)

    def forward(self, ws, t, c, motion_z=None, return_layer=None, **layer_kwargs):
        misc.assert_shape(ws, [None, self.num_ws, self.w_dim])

        if ws.shape[0] == t.shape[0]:
            ws = ws.repeat_interleave(t.shape[1], dim=0)
        else:
            # already expand the time dimension! 
            assert ws.shape[0] == t.shape[0] * t.shape[1], f"Wrong shape, ws: {ws.shape}, batch size: {t.shape[0]}, time: {t.shape[1]}"

        ws = ws.to(torch.float32).unbind(dim=1)

        motion_info = self.motion_encoder(c, t, motion_z=motion_z) # [batch_size * num_frames, motion_v_dim]
        motion_v = motion_info['motion_v'] # [batch_size * num_frames, motion_v_dim]
        motion_v = motion_v * self.motion_mask

        x = self.input(ws[0], motion_v, t, motion_mask=self.motion_mask if self.cfg.fuse_w == 'add' else None)
        cnt = 0
        x_feat = {}
        for name, w in zip(self.layer_names, ws[1:]):
            if self.cfg.fuse_w == 'concat': 
                w = self.affine_w(torch.cat([w, motion_v], 1))
            elif self.cfg.fuse_w == 'add':
                w = w + motion_v
            else:
                raise ValueError
            if self.cfg.low_rank is not None and self.cfg.low_rank > 0:
                x = getattr(self, name)(x, w, t=motion_v, **layer_kwargs)
            else:
                x = getattr(self, name)(x, w, **layer_kwargs)

            if (return_layer is not None) and (name in return_layer):
                x_feat[name] = x

            cnt += 1
        if self.output_scale != 1:
            x = x * self.output_scale

        # Ensure correct shape and dtype.
        misc.assert_shape(x, [None, self.img_channels, self.img_resolution, self.img_resolution])
        x = x.to(torch.float32)

        if return_layer is None:
            return x
        return x, x_feat


# NOTE source: network
@persistence.persistent_class
class Generator(torch.nn.Module):
    def __init__(self,
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        synthesis_kwargs    = {},   # Arguments for SynthesisNetwork.
        cfg                 = {},   # Config
    ):
        super().__init__()

        self.cfg = cfg
        self.sampling_dict = OmegaConf.to_container(OmegaConf.create({**self.cfg.sampling}))
        self.z_dim = self.cfg.z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.synthesis = SynthesisNetwork3v(w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels, cfg=cfg, **synthesis_kwargs)
        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork3(z_dim=self.z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, trainable=not cfg.freezemap)

        if getattr(self.cfg, "with_canonical", False):
            print(f"cfg.deform_channel: {cfg.deform_channel}")
            # # NOTE stylegan3 still won't converge
            # self.synthesis_deform = SynthesisNetwork3vDeform(w_dim=w_dim, img_resolution=img_resolution, img_channels=cfg.deform_channel, cfg=cfg, **synthesis_kwargs)
            # self.num_ws_deform = self.synthesis_deform.num_ws
            # self.mapping_deform = MappingNetwork3Deform(z_dim=self.z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws_deform, trainable=not cfg.freezemap)

            # switch back to stylegan2 (with dcn support)
            self.synthesis_deform = StyleGAN2SynthesisNetwork(w_dim=w_dim, img_resolution=img_resolution, img_channels=cfg.deform_channel, cfg=cfg, **synthesis_kwargs)
            self.num_ws_deform = self.synthesis_deform.num_ws
            self.mapping_deform = StyleGAN2MappingNetwork(z_dim=self.z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws_deform, **mapping_kwargs)
        torch.cuda.empty_cache()

    def forward(self, z, c, t, truncation_psi=1, truncation_cutoff=None, update_emas=False, return_aux=False, **synthesis_kwargs):
        assert len(z) == len(c) == len(t), f"Wrong shape: {z.shape}, {c.shape}, {t.shape}"
        assert t.ndim == 2, f"Wrong shape: {t.shape}"
        batch_size, num_timesteps = t.shape
        batch_times_timestep = batch_size * num_timesteps

        if not getattr(self.cfg, "with_canonical", False):
            # ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, ) # [batch_size, num_ws, w_dim]
            ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas) # [batch_size, num_ws, w_dim]
            img = self.synthesis(ws, t=t, c=c, update_emas=update_emas, **synthesis_kwargs) # [batch_size * num_frames, c, h, w]
            return img

        # generate canonical
        t_canonical = copy.deepcopy(t)[:, 0:1]  # (bs, 1)
        ws_canonical = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas) # [batch_size, num_ws, w_dim]
        # ['L0_36_512', 'L1_36_512', 'L2_36_512', 'L3_52_512', 'L4_52_512', 'L5_84_512', 'L6_84_512', 'L7_148_362', 'L8_148_256', 'L9_148_181', 'L10_276_128', 'L11_276_91', 'L12_276_64', 'L13_256_64', 'L14_256_3'] (legacy)
        # ytb: ['L0_36_512', 'L1_36_512', 'L2_36_512', 'L3_52_512', 'L4_52_512', 'L5_84_512', 'L6_84_512', 'L7_148_512', 'L8_148_512', 'L9_148_362', 'L10_276_256', 'L11_276_181', 'L12_276_128', 'L13_256_128', 'L14_256_3'] (now)
        # sky: ['L0_36_512', 'L1_36_512', 'L2_36_512', 'L3_52_512', 'L4_52_512', 'L5_84_512', 'L6_84_512', 'L7_148_362', 'L8_148_256', 'L9_148_181', 'L10_276_128', 'L11_276_91', 'L12_276_64', 'L13_256_64', 'L14_256_3']
        # taichi: 

        img_canonical, feat_canonical_dict = self.synthesis(
            ws_canonical, t=t_canonical, c=c, update_emas=update_emas,
            return_layer=self.cfg.canonical_feat,  # TODO
            **synthesis_kwargs) # [batch_size * num_frames, c, h, w]

        # generate anchor
        _, _, h, w = img_canonical.shape
        assert h == w, f"Wrong shape: {img_canonical.shape}"
        img_size = h
        raw_coords = generate_coords(batch_times_timestep, img_size, img_canonical.device)
        raw_coords = raw_coords.flip(-2)  # bchw, vertical flip (flip checked)

        # generate warp field, and apply it to canonical image for final gen
        ws_deform = self.mapping_deform(
            z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, 
            # update_emas=update_emas,
        ) # [batch_size, num_ws, w_dim]

        deform_offset = self.synthesis_deform(
            ws_deform, t=t, c=c, 
            # update_emas=update_emas, 
            # coords=raw_coords,  # used for tri-plane
            feat_canonical_dict=feat_canonical_dict,
            **synthesis_kwargs) # [batch_size * num_frames, c, h, w]
        deform_offset = deform_offset[:, :2, ...]

        # apply warp
        sample_coords = raw_coords + deform_offset  # x init checked
        sample_coords = rearrange(sample_coords, 'b c h w -> b h w c')  # b here is b*t
        # note the order: b t
        img_canonical_video = img_canonical.unsqueeze(1).repeat(1, num_timesteps, 1, 1, 1)
        img_canonical_video = rearrange(img_canonical_video, 'b t c h w -> (b t) c h w')
        img = F.grid_sample(img_canonical_video, sample_coords, align_corners=True, mode='bilinear', padding_mode='zeros')

        if not return_aux:
            return img
        return img, img_canonical, deform_offset, sample_coords


#----------------------------------------------------------------------------

@persistence.persistent_class
class DiscriminatorBlock(torch.nn.Module):
    def __init__(self,
        in_channels,                        # Number of input channels, 0 = first block.
        tmp_channels,                       # Number of intermediate channels.
        out_channels,                       # Number of output channels.
        resolution,                         # Resolution of this block.
        img_channels,                       # Number of input color channels.
        first_layer_idx,                    # Index of the first layer.
        architecture        = 'resnet',     # Architecture: 'orig', 'skip', 'resnet'.
        activation          = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
        resample_filter     = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp          = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16            = False,        # Use FP16 for this block?
        fp16_channels_last  = False,        # Use channels-last memory format with FP16?
        freeze_layers       = 0,            # Freeze-D: Number of layers to freeze.
        cfg                 = {},           # Main config.
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()

        self.cfg = cfg
        self.in_channels = in_channels
        self.resolution = resolution
        self.img_channels = img_channels
        self.first_layer_idx = first_layer_idx
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))

        self.num_layers = 0
        def trainable_gen():
            while True:
                layer_idx = self.first_layer_idx + self.num_layers
                trainable = (layer_idx >= freeze_layers)
                self.num_layers += 1
                yield trainable
        trainable_iter = trainable_gen()
        conv0_in_channels = in_channels if in_channels > 0 else tmp_channels

        total_train = True
        if in_channels == 0 or architecture == 'skip':
            trainable = next(trainable_iter)
            total_train = total_train and trainable
            self.fromrgb = Conv2dLayer(img_channels, tmp_channels, kernel_size=1, activation=activation,
                trainable=cfg.always_update or trainable, conv_clamp=conv_clamp, channels_last=self.channels_last)

        trainable = next(trainable_iter)
        total_train = total_train and trainable
        self.conv0 = Conv2dLayer(conv0_in_channels, tmp_channels, kernel_size=3, activation=activation,
                trainable=cfg.always_update or trainable, conv_clamp=conv_clamp, channels_last=self.channels_last)

        trainable = next(trainable_iter)
        total_train = total_train and trainable
        self.conv1 = Conv2dLayer(tmp_channels, out_channels, kernel_size=3, activation=activation, down=2,
            trainable=cfg.always_update or trainable, resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=self.channels_last)

        if architecture == 'resnet':
            trainable = next(trainable_iter)
            total_train = total_train and trainable
            self.skip = Conv2dLayer(conv0_in_channels, out_channels, kernel_size=1, bias=False, down=2,
                trainable=cfg.always_update or trainable, resample_filter=resample_filter, channels_last=self.channels_last)
        self.total_train = total_train

    def forward(self, x, img, cat, force_fp32=False):
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format

        # Input.
        if x is not None:
            misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution])
            x = x.to(dtype=dtype, memory_format=memory_format)

        # FromRGB.
        if self.in_channels == 0 or self.architecture == 'skip':
            misc.assert_shape(img, [None, self.img_channels, self.resolution, self.resolution])
            img = img.to(dtype=dtype, memory_format=memory_format)
            y = self.fromrgb(img)
            x = x + y if x is not None else y
            img = upfirdn2d.downsample2d(img, self.resample_filter) if self.architecture == 'skip' else None

        # Main layers.
        if self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            if self.cfg.tsm and not cat and self.total_train:
                x = x.view(-1, self.cfg.sampling.num_frames_per_video, *x.shape[1:])
                fold = x.shape[2] // 8
                out = torch.zeros_like(x, device=x.device)
                out[:, :-1, :fold] = x[:, 1:, :fold]
                out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]
                out[:, :, 2 * fold:] = x[:, :, 2 * fold:]
                x = out.view(-1, *x.shape[2:])
            x = self.conv0(x)
            x = self.conv1(x, gain=np.sqrt(0.5))
            x = y.add_(x)
        else:
            x = self.conv0(x)
            x = self.conv1(x)

        assert x.dtype == dtype
        return x, img

#----------------------------------------------------------------------------

@persistence.persistent_class
class MinibatchStdLayer(torch.nn.Module):
    def __init__(self, group_size, num_channels=1):
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x):
        N, C, H, W = x.shape
        with misc.suppress_tracer_warnings(): # as_tensor results are registered as constants
            G = torch.min(torch.as_tensor(self.group_size), torch.as_tensor(N)) if self.group_size is not None else N
        F = self.num_channels
        c = C // F

        y = x.reshape(G, -1, F, c, H, W)    # [GnFcHW] Split minibatch N into n groups of size G, and channels C into F groups of size c.
        y = y - y.mean(dim=0)               # [GnFcHW] Subtract mean over group.
        y = y.square().mean(dim=0)          # [nFcHW]  Calc variance over group.
        y = (y + 1e-8).sqrt()               # [nFcHW]  Calc stddev over group.
        y = y.mean(dim=[2,3,4])             # [nF]     Take average over channels and pixels.
        y = y.reshape(-1, F, 1, 1)          # [nF11]   Add missing dimensions.
        y = y.repeat(G, 1, H, W)            # [NFHW]   Replicate over group and pixels.
        x = torch.cat([x, y], dim=1)        # [N(C+1)HW]   Append to input as new channels.
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class DiscriminatorEpilogue(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        cmap_dim,                       # Dimensionality of mapped conditioning label, 0 = no label.
        resolution,                     # Resolution of this block.
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        mbstd_group_size    = 4,        # Group size for the minibatch standard deviation layer, None = entire minibatch.
        mbstd_num_channels  = 1,        # Number of features for the minibatch standard deviation layer, 0 = disable.
        activation          = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        conv_clamp          = None,     # Clamp the output of convolution layers to +-X, None = disable clamping.
        cfg                 = {},       # Architecture config.
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()

        self.cfg = cfg
        self.in_channels = in_channels
        self.cmap_dim = cmap_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.architecture = architecture

        if architecture == 'skip':
            self.fromrgb = Conv2dLayer(img_channels, in_channels, kernel_size=1, activation=activation)
        self.mbstd = MinibatchStdLayer(group_size=mbstd_group_size, num_channels=mbstd_num_channels) if mbstd_num_channels > 0 else None
        self.conv = Conv2dLayer(in_channels + mbstd_num_channels, in_channels, kernel_size=3, activation=activation, conv_clamp=conv_clamp)
        self.fc = FullyConnectedLayer(in_channels * (resolution ** 2), in_channels, activation=activation)
        self.out = FullyConnectedLayer(in_channels, 1 if cmap_dim == 0 else cmap_dim)

    def forward(self, x, img, cmap, force_fp32=False):
        misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution]) # [NCHW]
        _ = force_fp32 # unused
        dtype = torch.float32
        memory_format = torch.contiguous_format

        # FromRGB.
        x = x.to(dtype=dtype, memory_format=memory_format)
        if self.architecture == 'skip':
            misc.assert_shape(img, [None, self.img_channels, self.resolution, self.resolution])
            img = img.to(dtype=dtype, memory_format=memory_format)
            x = x + self.fromrgb(img)

        # Main layers.
        if self.mbstd is not None:
            x = self.mbstd(x)

        x = self.conv(x)
        x = self.fc(x.flatten(1))
        x = self.out(x) # [batch_size, out_dim]

        # Conditioning.
        if self.cmap_dim > 0:
            misc.assert_shape(cmap, [None, self.cmap_dim])
            x = (x * cmap).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim)) # [batch_size, 1]

        assert x.dtype == dtype
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class Discriminator(torch.nn.Module):
    def __init__(self,
        c_dim,                          # Conditioning label (C) dimensionality.
        img_resolution,                 # Input resolution.
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        channel_base        = 32768,    # Overall multiplier for the number of channels.
        channel_max         = 512,      # Maximum number of channels in any layer.
        num_fp16_res        = 0,        # Use FP16 for the N highest resolutions.
        conv_clamp          = None,     # Clamp the output of convolution layers to +-X, None = disable clamping.
        cmap_dim            = None,     # Dimensionality of mapped conditioning label, None = default.
        block_kwargs        = {},       # Arguments for DiscriminatorBlock.
        mapping_kwargs      = {},       # Arguments for MappingNetwork.
        epilogue_kwargs     = {},       # Arguments for DiscriminatorEpilogue.
        cfg                 = {},       # Additional config.
    ):
        super().__init__()

        # # NOTE hack implementation
        # self.depth_model = None
        # # self.depth_transform = None
        # self.depth_net_w = None
        # self.depth_net_h = None

        self.cfg = cfg
        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [4]}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        if cmap_dim is None:
            cmap_dim = channels_dict[4]

        if self.cfg.sampling.num_frames_per_video > 1:
            self.time_encoder = TemporalDifferenceEncoder(self.cfg)
            assert self.time_encoder.get_dim() > 0
        else:
            self.time_encoder = None

        if self.c_dim == 0 and self.time_encoder is None:
            cmap_dim = 0

        common_kwargs = dict(img_channels=img_channels, architecture=architecture, conv_clamp=conv_clamp)
        total_c_dim = c_dim + (0 if self.time_encoder is None else self.time_encoder.get_dim())
        cur_layer_idx = 0

        for res in self.block_resolutions:
            in_channels = channels_dict[res] if res < img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]

            if not self.cfg.tmean:
                # using cat strategy.
                if res // 2 == self.cfg.concat_res:
                    out_channels = out_channels // self.cfg.num_frames_div_factor
                if res == self.cfg.concat_res:
                    in_channels = (in_channels // self.cfg.num_frames_div_factor) * self.cfg.sampling.num_frames_per_video

            use_fp16 = (res >= fp16_resolution)
            block = DiscriminatorBlock(in_channels, tmp_channels, out_channels, resolution=res,
                first_layer_idx=cur_layer_idx, use_fp16=use_fp16, cfg=self.cfg, **block_kwargs, **common_kwargs)
            setattr(self, f'b{res}', block)
            cur_layer_idx += block.num_layers

        if self.c_dim > 0 or not self.time_encoder is None:
            self.mapping = MappingNetwork(z_dim=0, c_dim=total_c_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None, **mapping_kwargs)
        self.b4 = DiscriminatorEpilogue(channels_dict[4], cmap_dim=cmap_dim, resolution=4, cfg=self.cfg, **epilogue_kwargs, **common_kwargs)

    def forward(self, img, c, t, **block_kwargs):
        assert len(img) == t.shape[0] * t.shape[1], f"Wrong shape: {img.shape}, {t.shape}"
        assert t.ndim == 2, f"Wrong shape: {t.shape}"

        if not self.time_encoder is None:
            # Encoding the time distances
            t_embs = self.time_encoder(t.view(-1, self.cfg.sampling.num_frames_per_video)) # [batch_size, t_dim]

            # Concatenate `c` and time embeddings
            c = torch.cat([c, t_embs], dim=1) # [batch_size, c_dim + t_dim]
            c = (c * 0.0) if self.cfg.dummy_c else c # [batch_size, c_dim + t_dim]

        x = None
        cat = False
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            if res == self.cfg.concat_res:
                # Concatenating the frames
                if self.cfg.tmean:
                    x = x.view(-1, self.cfg.sampling.num_frames_per_video, *x.shape[1:]) # [batch_size, num_frames, c, h, w]
                    x = x.mean(1)
                else:
                    x = x.view(-1, self.cfg.sampling.num_frames_per_video, *x.shape[1:]) # [batch_size, num_frames, c, h, w]
                    x = x.view(x.shape[0], -1, *x.shape[3:]) # [batch_size, num_frames * c, h, w]
                cat = True
            x, img = block(x, img, cat, **block_kwargs)

        cmap = None
        if self.c_dim > 0 or not self.time_encoder is None:
            assert c.shape[1] > 0
        if c.shape[1] > 0:
            cmap = self.mapping(None, c)
        x = self.b4(x, img, cmap)
        x = x.squeeze(1) # [batch_size]

        return {'image_logits': x}

#----------------------------------------------------------------------------


            # NOTE legacy depth processing, which does not support ddp, and is time consuming
            # # normalize to depth input format
            # img_for_depth = img.detach().cpu().numpy()
            # img_for_depth = (img_for_depth + 1) * 0.5  # [-1, 1] -> [0, 1]
            # img_for_depth = rearrange(img_for_depth, 'b c h w -> b h w c')

            # # transform preprocessing, TODO better way than for loop ?
            # img_list = []
            # for i in range(img_for_depth.shape[0]):
            #     sample = self.depth_transform({"image": img_for_depth[i]})["image"]
            #     sample = torch.from_numpy(sample).to(img.device)
            #     img_list.append(sample)
            # img_for_depth = torch.stack(img_list, dim=0)

            # for n, p in self.depth_model.named_parameters():
            #     p.requires_grad = False

            # # forward
            # depth = self.depth_model.forward(img_for_depth)  # (bt, h, w)

            # # interpolation to 256
            # depth = torch.nn.functional.interpolate(
            #     depth.unsqueeze(1),
            #     size=img.shape[-2:],
            #     mode="bicubic",
            #     align_corners=False,
            # ).squeeze().cpu().numpy()

            # # check inf
            # if not np.isfinite(depth).all():
            #     depth=np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
            #     print("WARNING: Non-finite depth values present")

            # # normalize to [-1, 1]
            # depth_min = depth.min()
            # depth_max = depth.max()
            # depth_normed =  (depth - depth_min) / (depth_max - depth_min)
            # depth_normed = 2 * depth_normed - 1

            # # concat
            # depth_normed = torch.from_numpy(depth_normed).to(img.device).unsqueeze(1)
            # img = torch.cat([img, depth_normed], dim=1)




            # if img.requires_grad:
            #     import pdb; pdb.set_trace()
            # if self.depth_model is not None:
            #     # normalize to depth input format
            #     # [-1, 1] -> [0, 1] -> resize -> normalize to [-1, 1]
            #     # equivalent to [-1, 1] -> resize
            #     img_for_depth = torch.nn.functional.interpolate(
            #         img.detach(),
            #         size=(self.depth_net_h, self.depth_net_w),
            #         mode="bicubic",
            #         align_corners=False,
            #     )

            #     # forward
            #     depth = self.depth_model.forward(img_for_depth)  # (bt, h, w)
            #     # depth = self.depth_model(img_for_depth)  # (bt, h, w)

            #     # interpolation to 256
            #     depth = torch.nn.functional.interpolate(
            #         depth.unsqueeze(1),
            #         size=img.shape[-2:],
            #         mode="bicubic",
            #         align_corners=False,
            #     )

            #     if depth.requires_grad:
            #         import pdb; pdb.set_trace()

            #     # check inf
            #     if not torch.isfinite(depth).all():
            #         depth=torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
            #         print("WARNING: Non-finite depth values present")

            #     # normalize to [-1, 1]
            #     depth_min = depth.min()
            #     depth_max = depth.max()
            #     depth_normed =  (depth - depth_min) / (depth_max - depth_min)
            #     depth_normed = 2 * depth_normed - 1

            #     # concat
            #     # img = torch.cat([img, depth_normed], dim=1)
            #     img = img[:, 0:1, ...].repeat(1, 4, 1, 1)

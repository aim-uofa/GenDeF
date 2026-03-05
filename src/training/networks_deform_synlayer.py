# import copy
from einops import rearrange
import numpy as np
# import scipy
import scipy.signal
import scipy.optimize
import torch
# from torch import Tensor
import torch.nn.functional as F
# from omegaconf import OmegaConf

from src.torch_utils import misc
from src.torch_utils import persistence
# from src.torch_utils.ops import conv2d_resample, upfirdn2d, bias_act, fma
from src.torch_utils3.ops import filtered_lrelu

# from training.motion import MotionMappingNetwork, BSplineMotionMappingNetwork
# from training.layers import (
#     FullyConnectedLayer,
    # GenInput,
    # EqLRConv1d,
    # TemporalDifferenceEncoder,
    # Conv2dLayer,
    # MappingNetwork,
# )
# from training.networks_stylegan3 import MappingNetwork as MappingNetwork3
# from training.networks_stylegan3 import FullyConnectedLayer as FullyConnectedLayer3
from training.networks_stylegan3 import (
    # SynthesisNetwork,
    # SynthesisInput,
    modulated_conv2d,
    FullyConnectedLayer,
)



@persistence.persistent_class
class SynthesisLayer(torch.nn.Module):
    def __init__(self,
        w_dim,                          # Intermediate latent (W) dimensionality.
        is_torgb,                       # Is this the final ToRGB layer?
        is_critically_sampled,          # Does this layer use critical sampling?
        use_fp16,                       # Does this layer use FP16?

        # Input & output specifications.
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        in_size,                        # Input spatial size: int or [width, height].
        out_size,                       # Output spatial size: int or [width, height].
        in_sampling_rate,               # Input sampling rate (s).
        out_sampling_rate,              # Output sampling rate (s).
        in_cutoff,                      # Input cutoff frequency (f_c).
        out_cutoff,                     # Output cutoff frequency (f_c).
        in_half_width,                  # Input transition band half-width (f_h).
        out_half_width,                 # Output Transition band half-width (f_h).

        # Hyperparameters.
        low_rank            = None,
        always_lr            = False,
        conv_kernel         = 3,        # Convolution kernel size. Ignored for final the ToRGB layer.
        filter_size         = 6,        # Low-pass filter size relative to the lower resolution when up/downsampling.
        lrelu_upsampling    = 2,        # Relative sampling rate for leaky ReLU. Ignored for final the ToRGB layer.
        use_radial_filters  = False,    # Use radially symmetric downsampling filter? Ignored for critically sampled layers.
        conv_clamp          = 256,      # Clamp the output to [-X, +X], None = disable clamping.
        magnitude_ema_beta  = 0.999,    # Decay rate for the moving average of input magnitudes.
        trainable           = True,
        # dcn
        dcn                 = False,
        dcn_lr_mult         = False,
        # all parameters
        cfg                 = {},           # Additional config
    ):
        super().__init__()

        self.cfg = cfg
        assert self.cfg.canonical_cond in ['none', "concat"]
        # assert self.cfg.canonical_cond in ['none', "concat", "spade", "res_spade"]
        if self.cfg.canonical_cond != 'none':
            assert self.cfg.canonical_cond_dim > 0
        # print(self.cfg.canonical_cond, self.cfg.spade_pos)
        # assert self.cfg.spade_pos in ['before_conv', 'after_conv', 'after_act']

        self.w_dim = w_dim
        self.is_torgb = is_torgb
        self.is_critically_sampled = is_critically_sampled
        self.use_fp16 = use_fp16
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_size = np.broadcast_to(np.asarray(in_size), [2])
        self.out_size = np.broadcast_to(np.asarray(out_size), [2])
        self.in_sampling_rate = in_sampling_rate
        self.out_sampling_rate = out_sampling_rate
        self.tmp_sampling_rate = max(in_sampling_rate, out_sampling_rate) * (1 if is_torgb else lrelu_upsampling)
        self.in_cutoff = in_cutoff
        self.out_cutoff = out_cutoff
        self.in_half_width = in_half_width
        self.out_half_width = out_half_width
        self.conv_kernel = 1 if is_torgb else conv_kernel
        self.conv_clamp = conv_clamp
        self.magnitude_ema_beta = magnitude_ema_beta
        self.low_rank = low_rank
        self.always_lr = always_lr

        affine_out_channels = in_channels
        if self.cfg.canonical_cond == "concat":
            affine_out_channels += self.cfg.canonical_cond_dim
        # for dcn
        self.dcn = dcn
        self.dcn_lr_mult = dcn_lr_mult
        if self.dcn:
            dcn_out_channel = 3 * self.conv_kernel ** 2
            self.dcn_affine = FullyConnectedLayer(w_dim, affine_out_channels, bias_init=1)
            self.dcn_weight = torch.nn.Parameter(torch.randn([dcn_out_channel, affine_out_channels, self.conv_kernel, self.conv_kernel]))
            self.dcn_scales = torch.nn.Parameter(torch.zeros([1, dcn_out_channel, 1, 1]))
            self.dcn_bias = torch.nn.Parameter(torch.zeros([1, dcn_out_channel, 1, 1]))

        # Setup parameters and buffers.
        self.trainable = trainable
        self.affine = FullyConnectedLayer(self.w_dim, affine_out_channels, bias_init=1, trainable = trainable)
        weight = torch.randn([self.out_channels, affine_out_channels, self.conv_kernel, self.conv_kernel])
        if self.is_torgb:
            assert self.cfg.deform_init in ['zero', 'xavier', 'random']
            if self.cfg.deform_init == 'zero':
                weight = torch.zeros([self.out_channels, affine_out_channels, self.conv_kernel, self.conv_kernel])
            elif self.cfg.deform_init == 'xavier':
                torch.nn.init.xavier_uniform_(weight, gain=1.0)
        # bias are already initialized to 0
        bias = torch.zeros([self.out_channels])
        if trainable or always_lr:
            if self.low_rank is not None and self.low_rank > 0:
                self.affine_t = FullyConnectedLayer(self.w_dim, self.in_channels, bias_init=1, trainable = True)
                self.w1 = torch.nn.Parameter(torch.zeros([self.out_channels, low_rank, self.conv_kernel, self.conv_kernel]))
                self.w2 = torch.nn.Parameter(torch.zeros([low_rank, self.in_channels, self.conv_kernel, self.conv_kernel]))
        if trainable:
            self.weight = torch.nn.Parameter(weight)
            self.bias = torch.nn.Parameter(bias)
        else:
            self.register_buffer('weight', weight)
            self.register_buffer('bias', bias)
        self.register_buffer('magnitude_ema', torch.ones([]))

        # Design upsampling filter.
        self.up_factor = int(np.rint(self.tmp_sampling_rate / self.in_sampling_rate))
        assert self.in_sampling_rate * self.up_factor == self.tmp_sampling_rate
        self.up_taps = filter_size * self.up_factor if self.up_factor > 1 and not self.is_torgb else 1
        self.register_buffer('up_filter', self.design_lowpass_filter(
            numtaps=self.up_taps, cutoff=self.in_cutoff, width=self.in_half_width*2, fs=self.tmp_sampling_rate))

        # Design downsampling filter.
        self.down_factor = int(np.rint(self.tmp_sampling_rate / self.out_sampling_rate))
        assert self.out_sampling_rate * self.down_factor == self.tmp_sampling_rate
        self.down_taps = filter_size * self.down_factor if self.down_factor > 1 and not self.is_torgb else 1
        self.down_radial = use_radial_filters and not self.is_critically_sampled
        self.register_buffer('down_filter', self.design_lowpass_filter(
            numtaps=self.down_taps, cutoff=self.out_cutoff, width=self.out_half_width*2, fs=self.tmp_sampling_rate, radial=self.down_radial))

        # Compute padding.
        pad_total = (self.out_size - 1) * self.down_factor + 1 # Desired output size before downsampling.
        pad_total -= (self.in_size + self.conv_kernel - 1) * self.up_factor # Input size after upsampling.
        pad_total += self.up_taps + self.down_taps - 2 # Size reduction caused by the filters.
        pad_lo = (pad_total + self.up_factor) // 2 # Shift sample locations according to the symmetric interpretation (Appendix C.3).
        pad_hi = pad_total - pad_lo
        self.padding = [int(pad_lo[0]), int(pad_hi[0]), int(pad_lo[1]), int(pad_hi[1])]

    def forward(self, x, w, t=None, noise_mode='random', force_fp32=False, update_emas=False, canonical_feat=None):
        assert noise_mode in ['random', 'const', 'none'] # unused
        misc.assert_shape(x, [None, self.in_channels, int(self.in_size[1]), int(self.in_size[0])])
        misc.assert_shape(w, [x.shape[0], self.w_dim])

        # TODO: may concat after obtaining dcn offsets
        if self.cfg.canonical_cond == "concat":
            # resize canonical_feat to x size
            _, _, h_, w_ = x.shape
            canonical_feat = F.interpolate(canonical_feat, size=(h_, w_), mode='bilinear', align_corners=False)
            x = torch.cat([x, canonical_feat], dim=1)

        # Track input magnitude.
        # False
        if update_emas:
            with torch.autograd.profiler.record_function('update_magnitude_ema'):
                magnitude_cur = x.detach().to(torch.float32).square().mean()
                self.magnitude_ema.copy_(magnitude_cur.lerp(self.magnitude_ema, self.magnitude_ema_beta))
        input_gain = self.magnitude_ema.rsqrt()

        #dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32 
        #x = x.to(dtype)
        # for dcn
        # True
        if self.dcn:
            # as w is fused with t before
            dcn_styles = self.dcn_affine(w)
            dcn_weight = self.dcn_weight * self.dcn_lr_mult
            dcn_bias = self.dcn_bias * self.dcn_lr_mult
            dcn_scales = self.dcn_scales * self.dcn_lr_mult
            offset = modulated_conv2d(x=x, w=dcn_weight.to(x.dtype), s=dcn_styles, padding=self.conv_kernel-1)
            offset = offset * dcn_scales.to(x.dtype) + dcn_bias.to(x.dtype)
            offset_x, offset_y, offsets_mask = torch.chunk(offset, 3, dim=1)
            offsets = torch.cat([offset_x, offset_y], 1)
            offsets_mask = offsets_mask.sigmoid() * 2 # range in [0, 2]

        use_low_rank = self.low_rank is not None and self.low_rank > 0 and (self.always_lr or self.trainable)

        if use_low_rank and (self.cfg.canonical_cond != 'none'):
            raise NotImplementedError

        if use_low_rank:
            style_t = self.affine_t(t)

        # Execute affine layer.
        styles = self.affine(w)
        if self.is_torgb:
            weight_gain = 1 / np.sqrt(self.in_channels * (self.conv_kernel ** 2))
            styles = styles * weight_gain
            if use_low_rank:
                style_t = style_t * weight_gain

        # Execute modulated conv2d.
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32
       
        if use_low_rank:
            style_t = self.affine_t(t)
            x = modulated_conv2d(x=x.to(dtype), w=self.weight, s=styles, w1=self.w1, w2=self.w2, t=style_t,
                padding=self.conv_kernel-1, demodulate=(not self.is_torgb), input_gain=input_gain)
        else:
            if self.dcn:
                x = modulated_conv2d(x=x.to(dtype), w=self.weight, s=styles,
                    padding=self.conv_kernel-1, demodulate=(not self.is_torgb), input_gain=input_gain, offsets=offsets, offsets_mask=offsets_mask)
            else:
                x = modulated_conv2d(x=x.to(dtype), w=self.weight, s=styles, 
                    padding=self.conv_kernel-1, demodulate=(not self.is_torgb), input_gain=input_gain)

        # Execute bias, filtered leaky ReLU, and clamping.
        gain = 1 if self.is_torgb else np.sqrt(2)
        # NOTE leaky relu slope=1 for output layer, which means relu is not applied
        slope = 1 if self.is_torgb else 0.2
        if self.is_torgb:
            # to deformation, we make sure properly designed
            assert self.up_filter is None
            assert self.down_filter is None
            assert self.up_factor == 1
            assert self.down_factor == 1
        x = filtered_lrelu.filtered_lrelu(x=x, fu=self.up_filter, fd=self.down_filter, b=self.bias.to(x.dtype),
            up=self.up_factor, down=self.down_factor, padding=self.padding, gain=gain, slope=slope, clamp=self.conv_clamp)

        # Ensure correct shape and dtype.
        misc.assert_shape(x, [None, self.out_channels, int(self.out_size[1]), int(self.out_size[0])])
        assert x.dtype == dtype
        return x

    @staticmethod
    def design_lowpass_filter(numtaps, cutoff, width, fs, radial=False):
        assert numtaps >= 1

        # Identity filter.
        if numtaps == 1:
            return None

        # Separable Kaiser low-pass filter.
        if not radial:
            f = scipy.signal.firwin(numtaps=numtaps, cutoff=cutoff, width=width, fs=fs)
            return torch.as_tensor(f, dtype=torch.float32)

        # Radially symmetric jinc-based filter.
        x = (np.arange(numtaps) - (numtaps - 1) / 2) / fs
        r = np.hypot(*np.meshgrid(x, x))
        f = scipy.special.j1(2 * cutoff * (np.pi * r)) / (np.pi * r)
        beta = scipy.signal.kaiser_beta(scipy.signal.kaiser_atten(numtaps, width / (fs / 2)))
        w = np.kaiser(numtaps, beta)
        f *= np.outer(w, w)
        f /= np.sum(f)
        return torch.as_tensor(f, dtype=torch.float32)

    def extra_repr(self):
        return '\n'.join([
            f'w_dim={self.w_dim:d}, is_torgb={self.is_torgb},',
            f'is_critically_sampled={self.is_critically_sampled}, use_fp16={self.use_fp16},',
            f'in_sampling_rate={self.in_sampling_rate:g}, out_sampling_rate={self.out_sampling_rate:g},',
            f'in_cutoff={self.in_cutoff:g}, out_cutoff={self.out_cutoff:g},',
            f'in_half_width={self.in_half_width:g}, out_half_width={self.out_half_width:g},',
            f'in_size={list(self.in_size)}, out_size={list(self.out_size)},',
            f'in_channels={self.in_channels:d}, out_channels={self.out_channels:d}'])


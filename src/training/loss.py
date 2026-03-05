# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import copy
import random
import numpy as np
import torch
import torch.nn.functional as F
from src.torch_utils import training_stats
from src.torch_utils import misc
from src.torch_utils.ops import conv2d_gradfix
from einops import rearrange
from src.training.networks import generate_coords
import PIL


# def save_image_grid(img, fname, drange, grid_size):
#     lo, hi = drange
#     img = np.asarray(img, dtype=np.float32)
#     img = (img - lo) * (255 / (hi - lo))
#     img = np.rint(img).clip(0, 255).astype(np.uint8)

#     gw, gh = grid_size
#     _N, C, H, W = img.shape
#     img = img.reshape(gh, gw, C, H, W)
#     img = img.transpose(0, 3, 1, 4, 2)
#     img = img.reshape(gh * H, gw * W, C)

#     assert C in [1, 3]
#     if C == 1:
#         PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
#     if C == 3:
#         PIL.Image.fromarray(img, 'RGB').save(fname, quality=95)


def apply_norm(x):
    # keep dim
    assert len(x.shape) == 4
    mean = x.abs().mean(2, True).mean(3, True)
    norm = x / (mean + 1e-7)
    return norm


# borrowed from https://github.com/nianticlabs/monodepth2
def get_smooth_loss(disp, img, smooth_loss_norm=False):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    # TODO: should img in image space or feature space?
    # in monodepth: feature space, range [0, 1]
    # ours in feature space, range [-1, 1]
    #   out = (out * 0.5 + 0.5).clamp(0, 1).cpu()

    # img: (bs, 3, h, w)
    # disp: (bs, 2, h, w)

    if smooth_loss_norm:
        # NOTE apply norm on disp before computing grad
        disp = apply_norm(disp)
        # NOTE apply norm only on deformation
        # img = apply_norm(img)  # out = (out * 0.5 + 0.5).clamp(0, 1).cpu()

    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    # grad = grad_disp_x.mean(dim=[1,2,3]) + grad_disp_y.mean(dim=[1,2,3])
    grad = grad_disp_x.mean() + grad_disp_y.mean()

    return grad


#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(
            self, 
            cfg, 
            device, G_mapping, G_synthesis, D, augment_pipe=None, G_motion_encoder=None,
            style_mixing_prob=0.9, r1_gamma=10, pl_batch_shrink=2, pl_decay=0.01, pl_weight=2,
            # ours (many in cfg)
            G_mapping_deform=None, G_synthesis_deform=None,
            depth_model=None, depth_net_h=None, depth_net_w=None,
        ):
        super().__init__()

        self.cfg = cfg
        self.device = device
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.D = D
        self.augment_pipe = augment_pipe
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)
        self.G_motion_encoder = G_motion_encoder
        self.G_mapping_deform = G_mapping_deform
        self.G_synthesis_deform = G_synthesis_deform

        self.depth_model = depth_model
        self.depth_net_h = depth_net_h
        self.depth_net_w = depth_net_w

    def run_G(self, z, c, t, z_addition, sync, return_aux=False):
        assert (not self.cfg.training.interp_p > 0) and (not self.style_mixing_prob > 0)
        if not self.cfg.model.generator.with_canonical:
            with misc.ddp_sync(self.G_mapping, sync):
                ws = self.G_mapping(z, c)
                # interp_p = False
                # if self.cfg.training.interp_p > 0:
                #     if torch.rand([]) < self.cfg.training.interp_p:
                #         interp_p = True
                #         ws_addition = self.G_mapping(z_addition, c)
                #         k = torch.rand([t.shape[0], 1]).to(t.device) - (t - t.min(1).values.unsqueeze(1)) / self.cfg.training.interp_len

                # if self.style_mixing_prob > 0:
                #     with torch.autograd.profiler.record_function('style_mixing'):
                #         cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                #         cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                #         ws[:, cutoff:] = self.G_mapping(torch.randn_like(z), c, skip_w_avg_update=True)[:, cutoff:]
                #         if interp_p:
                #             ws_addition[:, cutoff:] = self.G_mapping(torch.randn_like(z), c, skip_w_avg_update=True)[:, cutoff:]

                # if interp_p:
                #     # ws, ws_addition: B, n, c
                #     ws = ws.unsqueeze(1).repeat(1, t.shape[1], 1, 1)
                #     ws_addition = ws_addition.unsqueeze(1).repeat(1, t.shape[1], 1, 1)
                #     k = k.view(k.shape[0], k.shape[1], 1, 1)
                #     ws = k * ws + (1-k) * ws_addition
                #     ws = ws.view(ws.shape[0] * ws.shape[1], ws.shape[2], ws.shape[3]) # B*t, n, c

            with misc.ddp_sync(self.G_synthesis, sync):
                out = self.G_synthesis(ws, t=t, c=c)
            return out, ws

        # ours
        batch_size, num_timesteps = t.shape
        batch_times_timestep = batch_size * num_timesteps

        # generate canonical
        with misc.ddp_sync(self.G_mapping, sync):
            t_canonical = copy.deepcopy(t)[:, 0:1]  # (bs, 1)
            ws_canonical = self.G_mapping(z, c) # [batch_size, num_ws, w_dim]
        
        with misc.ddp_sync(self.G_synthesis, sync):
            # ['L0_36_512', 'L1_36_512', 'L2_36_512', 'L3_52_512', 'L4_52_512', 'L5_84_512', 'L6_84_512', 'L7_148_362', 'L8_148_256', 'L9_148_181', 'L10_276_128', 'L11_276_91', 'L12_276_64', 'L13_256_64', 'L14_256_3'] (legacy)
            # ['L0_36_512', 'L1_36_512', 'L2_36_512', 'L3_52_512', 'L4_52_512', 'L5_84_512', 'L6_84_512', 'L7_148_512', 'L8_148_512', 'L9_148_362', 'L10_276_256', 'L11_276_181', 'L12_276_128', 'L13_256_128', 'L14_256_3'] (now)
            img_canonical, feat_canonical_dict = self.G_synthesis(
                ws_canonical, t=t_canonical, c=c,
                return_layer=self.cfg.model.generator.canonical_feat,  # TODO
            ) # [batch_size * num_frames, c, h, w]

        # generate anchor
        _, _, h, w = img_canonical.shape
        assert h == w, f"Wrong shape: {img_canonical.shape}"
        img_size = h
        raw_coords = generate_coords(batch_times_timestep, img_size, img_canonical.device)
        raw_coords = raw_coords.flip(-2)  # bchw, vertical flip (flip checked)

        # generate warp field, and apply it to canonical image for final gen
        with misc.ddp_sync(self.G_mapping_deform, sync):
            ws_deform = self.G_mapping_deform(z, c) # [batch_size, num_ws, w_dim]
        with misc.ddp_sync(self.G_synthesis_deform, sync):
            deform_offset = self.G_synthesis_deform(
                ws_deform, t=t, c=c,
                # coords=raw_coords,  # used for tri-plane
                feat_canonical_dict=feat_canonical_dict,
            ) # [batch_size * num_frames, c, h, w]
        deform_offset = deform_offset[:, :2, ...]

        # apply warp
        sample_coords = raw_coords + deform_offset  # x init checked
        sample_coords = rearrange(sample_coords, 'b c h w -> b h w c')  # b here is b*t
        # note the order: b t
        canonical_size = self.cfg.model.generator.get('canonical_size', 256)
        if canonical_size != sample_coords.shape[1]:
            img_canonical = F.interpolate(img_canonical, size=(canonical_size, canonical_size), mode='bilinear')
        img_canonical_video = img_canonical.unsqueeze(1).repeat(1, num_timesteps, 1, 1, 1)
        img_canonical_video = rearrange(img_canonical_video, 'b t c h w -> (b t) c h w')
        img = F.grid_sample(img_canonical_video, sample_coords, align_corners=True, mode='bilinear', padding_mode='zeros')

        if not return_aux:
            return img, ws_deform
        return img, img_canonical, deform_offset, sample_coords

    def run_D(self, img, c, t, sync):
        if self.augment_pipe is not None:
            if self.cfg.model.loss_kwargs.get('video_consistent_aug', False):
                nf, ch, h, w = img.shape
                f = self.cfg.sampling.num_frames_per_video
                n = nf // f
                img = img.view(n, f * ch, h, w) # [n, f * ch, h, w]

            img = self.augment_pipe(img) # [n, f * ch, h, w]

            if self.cfg.model.loss_kwargs.get('video_consistent_aug', False):
                img = img.view(n * f, ch, h, w) # [n * f, ch, h, w]

        if self.depth_model is not None:
            # normalize to depth input format
            # [-1, 1] -> [0, 1] -> resize -> normalize to [-1, 1]
            # equivalent to [-1, 1] -> resize
            img_for_depth = torch.nn.functional.interpolate(
                img.detach(),
                size=(self.depth_net_h, self.depth_net_w),
                mode="bicubic",
                align_corners=False,
            )

            # forward
            depth = self.depth_model.forward(img_for_depth)  # (bt, h, w)
            assert depth.requires_grad is False

            # interpolation to 256
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(1),
                size=img.shape[-2:],
                mode="bicubic",
                align_corners=False,
            )

            # check inf
            if not torch.isfinite(depth).all():
                depth=torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
                print("WARNING: Non-finite depth values present")

            # normalize to [-1, 1]
            depth_min = depth.min()
            depth_max = depth.max()
            depth_normed =  (depth - depth_min) / (depth_max - depth_min)
            depth_normed = 2 * depth_normed - 1

            # # checked
            # if is_real:
            #     # save img and corresponding depth for check
            #     batch_size = img.shape[0] // t.shape[1]
            #     img_out = (img * 0.5 + 0.5).clamp(0, 1).detach().cpu().numpy()
            #     depth_out = (depth_normed * 0.5 + 0.5).clamp(0, 1).detach().cpu().numpy()
            #     save_image_grid(img_out, f"img_{t[0, 0].item()}.png", drange=[0, 1], grid_size=(batch_size, t.shape[1]))
            #     save_image_grid(depth_out, f"depth_{t[0, 0].item()}.png", drange=[0, 1], grid_size=(batch_size, t.shape[1]))

            # concat
            img = torch.cat([img, depth_normed], dim=1)

        with misc.ddp_sync(self.D, sync):
            outputs = self.D(img, c, t)

        return outputs

    def accumulate_gradients(self, phase, real_img, real_c, real_t, gen_z, gen_c, gen_t, gen_z_addition, sync, gain):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain        = (phase in ['Gmain', 'Gboth'])
        do_Dmain        = (phase in ['Dmain', 'Dboth'])
        do_Gpl          = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)
        do_Dr1          = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)

        real_img = real_img.view(-1, *real_img.shape[2:]) # [batch_size * num_frames, c, h, w]

        # Gmain: Maximize logits for generated images.
        if do_Gmain:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, gen_t, gen_z_addition, sync=(sync and not do_Gpl)) # [batch_size * num_frames, c, h, w]
                D_out_gen = self.run_D(gen_img, gen_c, gen_t, sync=False) # [batch_size]
                training_stats.report('Loss/scores/fake', D_out_gen['image_logits'])
                training_stats.report('Loss/signs/fake', D_out_gen['image_logits'].sign())
                loss_Gmain = F.softplus(-D_out_gen['image_logits']) # -log(sigmoid(y))
                if 'video_logits' in D_out_gen:
                    loss_Gmain_video = F.softplus(-D_out_gen['video_logits']).mean() # -log(sigmoid(y)) # [1]
                    training_stats.report('Loss/scores/fake_video', D_out_gen['video_logits'])
                    training_stats.report('Loss/G/loss_video', loss_Gmain_video)
                else:
                    loss_Gmain_video = 0.0 # [1]
                training_stats.report('Loss/G/loss', loss_Gmain)

                # NOTE apply edge smoothness loss on deformation field of nearby frames
                if self.cfg.model.loss_kwargs.get('near_frame_smooth_loss', False):
                    # obtain nearby frames
                    near_t = []
                    starting_t = random.choice(gen_t.permute(1, 0))
                    near_t.append(starting_t)
                    for i in range(1, self.cfg.sampling.num_frames_per_video):
                        near_t.append(starting_t + i)
                    near_t = torch.stack(near_t, dim=1)
                    batch_size, num_frames = near_t.shape

                    # generate nearby frame deformation field
                    # img, img_canonical, deform_offset, sample_coords
                    gen_img_near_frame, img_canonical_near_frame, _, sample_coords_near_frame = self.run_G(gen_z, gen_c, near_t, gen_z_addition, sync=(sync and not do_Gpl), return_aux=True)

                    # apply smooth loss on deformation field
                    near_frame_smooth_loss_weight = self.cfg.model.loss_kwargs.get('near_frame_smooth_loss', False)
                    if self.cfg.model.loss_kwargs.get('near_frame_smooth_loss_norm', False):
                        sample_coords_near_frame = apply_norm(sample_coords_near_frame)
                    sample_coords_near_frame = rearrange(sample_coords_near_frame, '(b t) h w c -> b t c h w', t=num_frames)
                    gen_img_near_frame = rearrange(gen_img_near_frame, '(b t) c h w -> b t c h w', t=num_frames)

                    near_frame_smooth_loss_type = self.cfg.model.loss_kwargs.get('near_frame_smooth_loss_type')
                    assert near_frame_smooth_loss_type in ['deform2img', 'deform2deform', 'flow2img', '2to3_deform2deform', '2to3_img2img', '2to3_deform2img']
                    # NOTE: consider 2to3_deform2deform & 2to3_deform2img
                    if near_frame_smooth_loss_type == 'deform2deform':
                        near_frame_loss_smooth = get_smooth_loss(sample_coords_near_frame[:, 1, ...], sample_coords_near_frame[:, 0, ...].detach()) + \
                            get_smooth_loss(sample_coords_near_frame[:, 2, ...], sample_coords_near_frame[:, 1, ...].detach())
                    elif near_frame_smooth_loss_type == 'deform2img':
                        near_frame_loss_smooth = get_smooth_loss(sample_coords_near_frame[:, 1, ...], gen_img_near_frame[:, 0, ...].detach()) + \
                            get_smooth_loss(sample_coords_near_frame[:, 2, ...], gen_img_near_frame[:, 1, ...].detach())
                    elif near_frame_smooth_loss_type == 'flow2img':
                        coords = (sample_coords_near_frame + 1) * 0.5
                        flow = coords[:, 1:, ...] - coords[:, :-1, ...]
                        flow = apply_norm(rearrange(flow, 'b t c h w -> (b t) c h w'))
                        flow = rearrange(flow, '(b t) c h w -> b t c h w', b=batch_size)
                        near_frame_loss_smooth = get_smooth_loss(flow[:, 0, ...], gen_img_near_frame[:, 0, ...].detach()) + \
                            get_smooth_loss(flow[:, 1, ...], gen_img_near_frame[:, 1, ...].detach())

                    elif near_frame_smooth_loss_type in ['2to3_deform2deform', '2to3_img2img', '2to3_deform2img']:
                        sample_coords_real = sample_coords_near_frame[:, 2, ...]
                        img_real = gen_img_near_frame[:, 2, ...]
                        sample_coords_fake = 2 * sample_coords_near_frame[:, 1, ...] - sample_coords_near_frame[:, 0, ...]  # (bs, 2, h, w)
                        if near_frame_smooth_loss_type == '2to3_deform2deform':
                            near_frame_loss_smooth = get_smooth_loss(sample_coords_fake, sample_coords_real.detach())
                        elif near_frame_smooth_loss_type == '2to3_deform2img':
                            near_frame_loss_smooth = get_smooth_loss(sample_coords_fake, img_real.detach())
                        elif near_frame_smooth_loss_type == '2to3_img2img':
                            sample_coords_fake = rearrange(sample_coords_fake, 'b c h w -> b h w c')  # b here is just b
                            img_canonical_near_frame = rearrange(img_canonical_near_frame, '(b t) c h w -> b t c h w', b=batch_size)[:, 0, ...]  # b here is just b
                            img_fake = F.grid_sample(img_canonical_near_frame, sample_coords_fake, align_corners=True, mode='bilinear', padding_mode='zeros')
                            near_frame_loss_smooth = get_smooth_loss(img_fake, img_real.detach())

                    near_frame_loss_smooth = near_frame_loss_smooth * near_frame_smooth_loss_weight
                else:
                    near_frame_loss_smooth = 0.0

            with torch.autograd.profiler.record_function('Gmain_backward'):
                (loss_Gmain + loss_Gmain_video + near_frame_loss_smooth).mean().mul(gain).backward()

        # Gpl: Apply path length regularization.
        if do_Gpl:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_img, gen_ws = self.run_G(gen_z[:batch_size], gen_c[:batch_size], gen_t[:batch_size], gen_z_addition[:batch_size], sync=sync) # [batch_size * num_frames, c, h, w]
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
                    pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                (gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if do_Dmain:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                with torch.no_grad():
                    gen_img, _gen_ws = self.run_G(gen_z, gen_c, gen_t, gen_z_addition, sync=False) # [batch_size * num_frames, c, h, w]
                D_out_gen = self.run_D(gen_img, gen_c, gen_t, sync=False) # Gets synced by loss_Dreal.
                training_stats.report('Loss/scores/fake', D_out_gen['image_logits'])
                training_stats.report('Loss/signs/fake', D_out_gen['image_logits'].sign())
                loss_Dgen = F.softplus(D_out_gen['image_logits']) # -log(1 - sigmoid(y))

                if 'video_logits' in D_out_gen:
                    loss_Dgen_video = F.softplus(D_out_gen['video_logits']).mean() # [1]
                    training_stats.report('Loss/scores/fake_video', D_out_gen['video_logits'])
                else:
                    loss_Dgen_video = 0.0 # [1]

            with torch.autograd.profiler.record_function('Dgen_backward'):
                (loss_Dgen + loss_Dgen_video).mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain or do_Dr1:
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
                D_out_real = self.run_D(real_img_tmp, real_c, real_t, sync=sync)
                training_stats.report('Loss/scores/real', D_out_real['image_logits'])
                training_stats.report('Loss/signs/real', D_out_real['image_logits'].sign())

                loss_Dreal = 0
                loss_Dreal_dist_preds = 0
                loss_Dreal_video = 0.0 # [1]
                if do_Dmain:
                    loss_Dreal = F.softplus(-D_out_real['image_logits']) # -log(sigmoid(y))
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                    if 'video_logits' in D_out_gen:
                        loss_Dreal_video = F.softplus(-D_out_real['video_logits']).mean() # [1]
                        training_stats.report('Loss/scores/real_video', D_out_real['video_logits'])
                        training_stats.report('Loss/D/loss_video', loss_Dgen_video + loss_Dreal_video)

                loss_Dr1 = 0
                if do_Dr1:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[D_out_real['image_logits'].sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2) # [batch_size * num_frames_per_video]
                    loss_Dr1 = loss_Dr1.view(-1, len(real_img_tmp) // len(D_out_real['image_logits'])).mean(dim=1) # [batch_size]
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            dummy_video_logits = (D_out_real["video_logits"].sum() * 0.0) if "video_logits" in D_out_real else 0.0
            with torch.autograd.profiler.record_function(name + '_backward'):
                (D_out_real["image_logits"] * 0 + dummy_video_logits + loss_Dreal + loss_Dreal_video + loss_Dr1 + loss_Dreal_dist_preds).mean().mul(gain).backward()

#----------------------------------------------------------------------------

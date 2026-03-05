import os
from typing import List, Callable, Optional, Dict
from multiprocessing.pool import ThreadPool

from PIL import Image
import torch
from torch import Tensor
import numpy as np
import cv2
from tqdm import tqdm
from torchvision import utils
import torchvision.transforms.functional as TVF
from training.networks_stylegan3 import modulated_conv2d
import torch.nn.functional as F
from einops import rearrange

#----------------------------------------------------------------------------


@torch.no_grad()
def generate_videos(
    G: Callable, z: Tensor, c: Tensor, ts: Tensor, z_addition: Tensor=None, motion_z: Optional[Tensor]=None,
    noise_mode='const', truncation_psi=1.0, verbose: bool=False, as_grids: bool=False, batch_size_num_frames: int=100,
    return_aux=False, 
    ) -> Tensor:

    assert len(ts) == len(z) == len(c), f"Wrong shape: {ts.shape}, {z.shape}, {c.shape}"
    assert ts.ndim == 2, f"Wrong shape: {ts.shape}"

    G.eval()
    videos = []
    videos_canonical = []
    videos_deformation = []
    videos_deformation_offset = []

    if c.shape[1] > 0 and truncation_psi < 1:
        num_ws_to_average = 1000
        c_for_avg = c.repeat_interleave(num_ws_to_average, dim=0) # [num_classes * num_ws_to_average, num_classes]
        z_for_avg = torch.randn(c_for_avg.shape[0], G.z_dim, device=z.device) # [num_classes * num_ws_to_average, z_dim]
        w = G.mapping(z_for_avg, c=c_for_avg)[:, 0] # [num_classes * num_ws_to_average, w_dim]
        w_avg = w.view(-1, num_ws_to_average, G.w_dim).mean(dim=1) # [num_classes, w_dim]

    iters = range(len(z))
    iters = tqdm(iters, desc='Generating videos') if verbose else iters

    if motion_z is None and not G.synthesis.motion_encoder is None:
        motion_z = G.synthesis.motion_encoder(c=c, t=ts)['motion_z'] # [...any...]

    for video_idx in iters:
        curr_video = []
        curr_canonical = []
        curr_deform_field = []
        curr_deform_field_offset = []

        for curr_ts in ts[[video_idx]].split(batch_size_num_frames, dim=1):
            curr_z = z[[video_idx]] # [1, z_dim]
            if z_addition is not None:
                curr_z_addition = z_addition[[video_idx]] # [1, z_dim]
            else:
                curr_z_addition = None
            curr_c = c[[video_idx]] # [1, c_dim]
            try:
                curr_motion_z = motion_z[[video_idx]]
            except:
                curr_motion_z = None
                

            if curr_c.shape[1] > 0 and truncation_psi < 1:
                raise NotImplementedError
                curr_w = G.mapping(curr_z, c=curr_c, truncation_psi=1) # [1, num_ws, w_dim]
                curr_w = truncation_psi * curr_w + (1 - truncation_psi) * w_avg.unsqueeze(1) # [1, num_ws, w_dim]
                out = G.synthesis(
                    ws=curr_w,
                    c=curr_c,
                    t=curr_ts,
                    motion_z=curr_motion_z,
                    noise_mode=noise_mode) # [1 * curr_num_frames, 3, h, w]
            else:
                if curr_z_addition is not None:
                    raise NotImplementedError
                    curr_w = G.mapping(curr_z, c=curr_c, truncation_psi=truncation_psi).unsqueeze(1).repeat(1, curr_ts.shape[1], 1, 1)
                    curr_w_addition = G.mapping(curr_z_addition, c=curr_c, truncation_psi=truncation_psi).unsqueeze(1).repeat(1, curr_ts.shape[1], 1, 1)
                    k = 1 - (curr_ts) / 4096
                    k = k.view(k.shape[0], k.shape[1], 1, 1)
                    curr_w = k * curr_w + (1-k) * curr_w_addition
                    curr_w = curr_w.view(-1, curr_w.shape[2], curr_w.shape[3])
                    out = G.synthesis(ws=curr_w,c=curr_c,t=curr_ts,motion_z=curr_motion_z,noise_mode=noise_mode)
                else:
                    assert return_aux
                    out, img_canonical, deform_offset, sample_coords = G(
                        z=curr_z,
                        c=curr_c,
                        t=curr_ts,
                        motion_z=curr_motion_z,
                        truncation_psi=truncation_psi,
                        noise_mode=noise_mode,
                        return_aux=return_aux,
                    ) # [1 * curr_num_frames, 3, h, w]

                # ws = G.mapping(curr_z, curr_c, truncation_psi=truncation_psi)
                # curr_w = ws.repeat_interleave(curr_ts.shape[1], dim=0)
                # curr_w = curr_w.to(torch.float32).unbind(dim=1)
                # if G.cfg.flow or G.cfg.fuse_w:
                #     motion_info = G.synthesis.motion_encoder(curr_c, curr_ts, motion_z=curr_motion_z)
                #     motion_v = motion_info['motion_v']
                # else:
                #     motion_v = None

                # # x = G.synthesis.input(curr_w[0], motion_v)
                # w = curr_w[0]
                # transforms = G.synthesis.input.transform.unsqueeze(0)
                # freqs = G.synthesis.input.freqs.unsqueeze(0)
                # phases = G.synthesis.input.phases.unsqueeze(0)
                # t = G.synthesis.input.affine(w)
                # t = t / t[:, :2].norm(dim=1, keepdim=True)
                # m_r = torch.eye(3, device=w.device).unsqueeze(0).repeat([w.shape[0], 1, 1])
                # m_r[:, 0, 0] = t[:, 0]  # r'_c   
                # m_r[:, 0, 1] = -t[:, 1] # r'_s  
                # m_r[:, 1, 0] = t[:, 1]  # r'_s   
                # m_r[:, 1, 1] = t[:, 0]  # r'_c 
                # m_t = torch.eye(3, device=w.device).unsqueeze(0).repeat([w.shape[0], 1, 1]) 
                # m_t[:, 0, 2] = -t[:, 2] # t'_x            
                # m_t[:, 1, 2] = -t[:, 3] # t'_y     
                # transforms = m_r @ m_t @ transforms 
                # phases = phases + (freqs @ transforms[:, :2, 2:]).squeeze(2)   
                # freqs = freqs @ transforms[:, :2, :2]             
                # amplitudes = (1 - (freqs.norm(dim=2) - G.synthesis.input.bandwidth) / (G.synthesis.input.sampling_rate / 2 - G.synthesis.input.bandwidth)).clamp(0, 1) 

                # theta = torch.eye(2, 3, device=w.device)  
                # theta[0, 0] = 0.5 * G.synthesis.input.size[0] / G.synthesis.input.sampling_rate
                # theta[1, 1] = 0.5 * G.synthesis.input.size[1] / G.synthesis.input.sampling_rate
                # grids = torch.nn.functional.affine_grid(theta.unsqueeze(0), [1, 1, G.synthesis.input.size[1], G.synthesis.input.size[0]], align_corners=False)
                # x = (grids.unsqueeze(3) @ freqs.permute(0, 2, 1).unsqueeze(1).unsqueeze(2)).squeeze(3) 
                # x = x + phases.unsqueeze(1).unsqueeze(2)  
                # x = torch.sin(x * (np.pi * 2)) 
                # x = x * amplitudes.unsqueeze(1).unsqueeze(2)           
                # weight = G.synthesis.input.weight / np.sqrt(G.synthesis.input.channels)
                # x = x @ weight.t()         

                # x = x.permute(0, 3, 1, 2) 
                # if G.synthesis.input.cfg.flow:
                #     flow = modulated_conv2d(x, w=G.synthesis.input.grid_weight, s=G.synthesis.input.motion_affine(motion_v), padding=G.synthesis.input.conv_kernel-2, demodulate=True, input_gain=None)
                #     grids = grids + flow.permute(0,2,3,1)
                #     x = (grids.unsqueeze(3) @ freqs.permute(0, 2, 1).unsqueeze(1).unsqueeze(2)).squeeze(3) 
                #     x = x + phases.unsqueeze(1).unsqueeze(2)  
                #     x = torch.sin(x * (np.pi * 2)) 
                #     x = x * amplitudes.unsqueeze(1).unsqueeze(2)           
                #     weight = G.synthesis.input.weight / np.sqrt(G.synthesis.input.channels)
                #     x = x @ weight.t()         

                #     x = x.permute(0, 3, 1, 2)

                #     # grids, 128, 36, 36, 2 -> grids_video, 128, 3, 36, 36
                #     def norma(x):
                #         L,H = x.quantile(0.05), x.quantile(0.95)
                #         return (x-L) / (H-L)
                #     video_x = norma(grids[...,0])[...,None]
                #     video_y = norma(grids[...,1])[...,None]
                #     video_z = torch.zeros_like(video_x)                    
                #     
                #     grids_video = torch.cat([video_x, video_y, video_z], -1).permute(0, 3, 1, 2)
                #     grids_video = F.interpolate(grids_video, size=(256, 256))

                # for name, w in zip(G.synthesis.layer_names, curr_w[1:]):
                #     if G.synthesis.cfg.fuse_w:
                #         w = G.synthesis.affine_w(torch.cat([w, motion_v], 1))
                #     x = getattr(G.synthesis, name)(x, w, noise_mode=noise_mode)
                # if G.synthesis.output_scale != 1:
                #     x = x * G.synthesis.output_scale
                # out = x.to(torch.float32)

            out = (out * 0.5 + 0.5).clamp(0, 1).cpu() # [1 * curr_num_frames, 3, h, w]
            img_canonical = (img_canonical * 0.5 + 0.5).clamp(0, 1).cpu() # [1 * curr_num_frames, 3, h, w]
            sample_coords = rearrange(sample_coords, 'b h w c -> b c h w')
            # if G.synthesis.input.cfg.flow:
            #     out = torch.cat([out, grids_video.cpu()], -1)

            curr_video.append(out)
            curr_canonical.append(img_canonical)
            curr_deform_field.append(sample_coords)
            curr_deform_field_offset.append(deform_offset)

        videos.append(torch.cat(curr_video, dim=0))
        videos_canonical.append(torch.cat(curr_canonical, dim=0))
        videos_deformation.append(torch.cat(curr_deform_field, dim=0))
        videos_deformation_offset.append(torch.cat(curr_deform_field_offset, dim=0))

    videos = torch.stack(videos) # [len(z), video_len, c, h, w]
    videos_canonical = torch.stack(videos_canonical) # [len(z), video_len, c, h, w]
    videos_deformation = torch.stack(videos_deformation) # [len(z), video_len, c, h, w]
    videos_deformation_offset = torch.stack(videos_deformation_offset) # [len(z), video_len, c, h, w]

    if as_grids:
        raise NotImplementedError
        frame_grids = videos.permute(1, 0, 2, 3, 4) # [video_len, len(z), c, h, w]
        frame_grids = [utils.make_grid(fs, nrow=int(np.sqrt(len(z)))) for fs in frame_grids] # [video_len, 3, grid_h, grid_w]

        return torch.stack(frame_grids)
    else:
        return videos, videos_canonical, videos_deformation, videos_deformation_offset

#----------------------------------------------------------------------------

def run_batchwise(fn: Callable, data_kwargs: Dict[str, Tensor], batch_size: int, **kwargs) -> Tensor:
    data_kwargs = {k: v for k, v in data_kwargs.items() if not v is None}
    seq_len = len(data_kwargs[list(data_kwargs.keys())[0]])
    result = []

    for i in range((seq_len + batch_size - 1) // batch_size):
        curr_data_kwargs = {k: d[i * batch_size: (i+1) * batch_size] for k, d in data_kwargs.items()}
        result.append(fn(**curr_data_kwargs, **kwargs))

    return torch.cat(result, dim=0)

#----------------------------------------------------------------------------

def save_video_frames_as_mp4(frames: List[Tensor], fps: int, save_path: os.PathLike, verbose: bool=False):
    # Load data
    frame_h, frame_w = frames[0].shape[1:]
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # fourcc = cv2.VideoWriter_fourcc('F', 'L', 'V', '1')
    # save_path = save_path.replace('.mp4', '.flv')
    # fourcc = cv2.VideoWriter_fourcc('P', 'I', 'M', 'I')
    # fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    # save_path = save_path.replace('.mp4', '.avi')
    # save_path = save_path.replace('.mp4', '.gif')
    video = cv2.VideoWriter(save_path, fourcc, fps, (frame_w, frame_h))
    frames = tqdm(frames, desc='Saving videos') if verbose else frames
    for frame in frames:
        assert frame.shape[0] == 3, "RGBA/grayscale images are not supported"
        frame = np.array(TVF.to_pil_image(frame))
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    # Uncomment this line to release the memory.
    # It didn't work for me on centos and complained about installing additional libraries (which requires root access)
    # cv2.destroyAllWindows()
    video.release()

#----------------------------------------------------------------------------

def save_video_frames_as_frames(frames: List[Tensor], save_dir: os.PathLike, time_offset: int=0):
    os.makedirs(save_dir, exist_ok=True)

    for i, frame in enumerate(frames):
        save_path = os.path.join(save_dir, f'{i + time_offset:06d}.jpg')
        TVF.to_pil_image(frame).save(save_path, q=95)

#----------------------------------------------------------------------------

def save_video_frames_as_frames_parallel(frames: List[np.ndarray], save_dir: os.PathLike, time_offset: int=0, num_processes: int=1):
    assert num_processes > 1, "Use `save_video_frames_as_frames` if you do not plan to use num_processes > 1."
    os.makedirs(save_dir, exist_ok=True)
    # We are fine with the ThreadPool instead of Pool since most of the work is I/O
    pool = ThreadPool(processes=num_processes)
    save_paths = [os.path.join(save_dir, f'{i + time_offset:06d}.jpg') for i in range(len(frames))]
    pool.map(save_jpg_mp_proxy, [(f, p) for f, p in zip(frames, save_paths)])

#----------------------------------------------------------------------------

def save_jpg_mp_proxy(args):
    return save_jpg(*args)

#----------------------------------------------------------------------------

def save_jpg(x: np.ndarray, save_path: os.PathLike):
    Image.fromarray(x).save(save_path, q=95)

#----------------------------------------------------------------------------

from transformers import logging
from omegaconf import OmegaConf
# suppress partial model loading warning
logging.set_verbosity_error()
from mvdream.ldm.util import instantiate_from_config
import pkg_resources
import os

import torch

from utilities.camera import normalize_camera, convert_opengl_to_blender

import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd
from loguru import logger


def get_config_file(config_path):
    cfg_file = pkg_resources.resource_filename(
        "mvdream", os.path.join("configs", config_path)
    )
    if not os.path.exists(cfg_file):
        raise RuntimeError(f"Config {config_path} not available!")
    return cfg_file


class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None

class MVDiffusion(nn.Module):
    def __init__(self, device, ckpt_path='F:/models/sd-v2.1-base-4view.pt', latent_mode=False):
        super().__init__()
        config_file = get_config_file("sd-v2-base.yaml")
        config = OmegaConf.load(config_file)
        logger.info(f'loading mv stable diffusion at {ckpt_path}...')
        self.model = instantiate_from_config(config.model)
        self.model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        for p in self.model.parameters():
            p.requires_grad_(False)
        # self.sampler = DDIMSampler(self.model)
        self.dtype = torch.float16
        self.device = device

        self.latent_mode = latent_mode
        self.num_train_timesteps = 1000
        self.min_step = int(self.num_train_timesteps * 0.02)
        self.max_step = int(self.num_train_timesteps * 0.98)
        self.to(self.device)

        logger.info(f'\t successfully loaded stable diffusion!')

    def encode_images(self, imgs):
        imgs = imgs * 2.0 - 1.0
        latents = self.model.get_first_stage_encoding(self.model.encode_first_stage(imgs))
        return latents

    def get_camera_cond(self, camera):
        camera = convert_opengl_to_blender(camera)
        camera = normalize_camera(camera)
        camera = camera.flatten(start_dim=1)
        return camera

    def get_text_embeds(self, prompt):
        # self.model.get_learned_conditioning(prompt)
        # Tokenize text and get embeddings
        text_embeddings= self.model.get_learned_conditioning(prompt)
        # Do the same for unconditional embeddings
        uncond_embeddings = self.model.get_learned_conditioning([''])

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    def train_step(self, inputs, c2w, text_embeddings, guidance_scale=100, grad_scale=1., grad_clamp=False):
        batch_size = inputs.shape[0]
        camera = c2w
        # interp to 512x512 to be fed into vae.
        # uncond, cond = text_embeddings.chunk(2)
        # # _t = time.time()
        if not self.latent_mode:
            # latents = F.interpolate(latents, (64, 64), mode='bilinear', align_corners=False)
            pred_rgb_512 = F.interpolate(inputs, (512, 512), mode='bilinear', align_corners=False)
            latents = self.encode_images(pred_rgb_512)
        else:
            latents = inputs

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(self.min_step, self.max_step + 1, (latents.shape[0],), dtype=torch.long, device=self.device)
        t_expand = t.repeat(text_embeddings.shape[0])
        # encode image into latents with vae, requires grad!
        # _t = time.time()

        # torch.cuda.synchronize(); print(f'[TIME] guiding: vae enc {time.time() - _t:.4f}s')

        # predict the noise residual with unet, NO grad!
        # _t = time.time()
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=self.dtype):
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.model.q_sample(latents, t, noise)
            latent_model_input = torch.cat([latents_noisy] * 2)
            # pred noise
            # latent_model_input = torch.cat([latents_noisy] * 2)
            if camera is not None:
                camera = self.get_camera_cond(camera)
                camera = camera.repeat(2, 1).to(text_embeddings)
                context = {"context": text_embeddings, "camera": camera, "num_frames": 1}
            else:
                context = {"context": text_embeddings}

            noise_pred = self.model.apply_model(latent_model_input, t_expand, context)

        # torch.cuda.synchronize(); print(f'[TIME] guiding: unet {time.time() - _t:.4f}s')

        # perform guidance (high scale from paper!)
        # noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # w(t), alpha_t * sigma_t^2
        w = (1 - self.model.alphas_cumprod[t])
        # w = self.alphas[t] ** 0.5 * (1 - self.alphas[t])
        # grad = w * (noise_pred - noise)
        grad = grad_scale * w[:, None, None, None] * (noise_pred - noise)
        grad = torch.nan_to_num(grad)
        # clip grad for stable training?
        if grad_clamp:
            grad = grad.clamp(-1, 1)

        # manually backward, since we omitted an item in grad and cannot simply autodiff.
        # _t = time.time()
        target = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents, target, reduction="sum") / latents.shape[0]
        # loss = SpecifyGradient.apply(latents, grad)
        # torch.cuda.synchronize(); print(f'[TIME] guiding: backward {time.time() - _t:.4f}s')

        return loss  # dummy loss value



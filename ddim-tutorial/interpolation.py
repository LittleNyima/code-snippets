# model

from diffusers import UNet2DModel

model = UNet2DModel.from_pretrained('ddpm-anime-faces-64').cuda()


# interpolation

import torch

def slerp(
    x0: torch.Tensor,
    x1: torch.Tensor,
    alpha: float,
):
    # todo: separate each batch?
    theta = torch.acos(torch.sum(x0 * x1) / (torch.norm(x0) * torch.norm(x1)))
    w0 = torch.sin((1.0 - alpha) * theta) / torch.sin(theta)
    w1 = torch.sin(alpha * theta) / torch.sin(theta)
    return w0 * x0 + w1 * x1

def interpolation_grid(
    rows: int,
    cols: int,
    in_channels: int,
    sample_size: int,
):
    images = torch.zeros((rows * cols, in_channels, sample_size, sample_size), dtype=torch.float32)
    images[0, ...] = torch.randn_like(images[0, ...]) # top left
    images[cols - 1, ...] = torch.randn_like(images[0, ...]) # top right
    images[(rows - 1) * cols, ...] = torch.randn_like(images[0, ...]) # bottom left
    images[-1] = torch.randn_like(images[0, ...]) # bottom right
    for row in range(1, rows - 1): # interpolate left most column and right most column
        alpha = row / (rows - 1)
        images[row * cols, ...] = slerp(images[0, ...], images[(rows - 1) * cols, ...], alpha)
        images[(row + 1) * cols - 1, ...] = slerp(images[cols - 1, ...], images[-1, ...], alpha)
    for col in range(1, cols - 1): # interpolate others
        alpha = col / (cols - 1)
        images[col::cols, ...] = slerp(images[0::cols, ...], images[cols - 1::cols, ...], alpha)
    return images


# core

import torch
import math
from tqdm import tqdm

class DDIM:
    def __init__(
        self,
        num_train_timesteps:int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        sample_steps: int = 20,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.timesteps = torch.linspace(num_train_timesteps - 1, 0, sample_steps).long()

    @torch.no_grad()
    def sample(
        self,
        unet: UNet2DModel,
        rows: int,
        cols: int,
        in_channels: int,
        sample_size: int,
        eta: float = 0.0,
    ):
        alphas = self.alphas.to(unet.device)
        alphas_cumprod = self.alphas_cumprod.to(unet.device)
        timesteps = self.timesteps.to(unet.device)
        images = interpolation_grid(rows, cols, in_channels, sample_size).to(unet.device)
        for t, tau in tqdm(list(zip(timesteps[:-1], timesteps[1:])), desc='Sampling'):
            pred_noise: torch.Tensor = unet(images, t).sample

            # sigma_t
            if not math.isclose(eta, 0.0):
                one_minus_alpha_prod_tau = 1.0 - alphas_cumprod[tau]
                one_minus_alpha_prod_t = 1.0 - alphas_cumprod[t]
                one_minus_alpha_t = 1.0 - alphas[t]
                sigma_t = eta * (one_minus_alpha_prod_tau * one_minus_alpha_t / one_minus_alpha_prod_t) ** 0.5
            else:
                sigma_t = torch.zeros_like(alphas[0])

            # first term of x_tau
            alphas_cumprod_tau = alphas_cumprod[tau]
            sqrt_alphas_cumprod_tau = alphas_cumprod_tau ** 0.5
            alphas_cumprod_t = alphas_cumprod[t]
            sqrt_alphas_cumprod_t = alphas_cumprod_t ** 0.5
            sqrt_one_minus_alphas_cumprod_t = (1.0 - alphas_cumprod_t) ** 0.5
            first_term = sqrt_alphas_cumprod_tau * (images - sqrt_one_minus_alphas_cumprod_t * pred_noise) / sqrt_alphas_cumprod_t

            # second term of x_tau
            coeff = (1.0 - alphas_cumprod_tau - sigma_t ** 2) ** 0.5
            second_term = coeff * pred_noise

            epsilon = torch.randn_like(images)
            images = first_term + second_term + sigma_t * epsilon
        images = (images / 2.0 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
        return images

ddim = DDIM()
images = ddim.sample(model, 4, 8, 3, 64)

from diffusers.utils import make_image_grid, numpy_to_pil
image_grid = make_image_grid(numpy_to_pil(images), rows=4, cols=8)
image_grid.save('ddim-interpolation-results.png')

import math
import torch
from functools import partial
from tqdm import tqdm
from diffusers import UNet2DModel

def make_betas_cosine_schedule(
    num_diffusion_timesteps: int = 1000,
    beta_max: float = 0.999,
    s: float = 8e-3,
):
    fn = lambda t: math.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(1.0 - fn(t2) / fn(t1))
    return torch.tensor(betas, dtype=torch.float32).clamp_max(beta_max)

def extract(arr: torch.Tensor, timesteps: torch.Tensor, broadcast_shape: torch.Size):
    arr = arr[timesteps]
    while len(arr.shape) < len(broadcast_shape):
        arr = arr.unsqueeze(-1)
    return arr.expand(broadcast_shape)

class IDDPM:

    def __init__(
        self,
        num_diffusion_timesteps: int = 1000,
        beta_max: float = 0.999,
    ):
        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.betas = make_betas_cosine_schedule(num_diffusion_timesteps, beta_max)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.concat((torch.ones(1).to(self.alphas_cumprod), self.alphas_cumprod[:-1]))
        self.alphas_cumprod_next = torch.concat((self.alphas_cumprod[1:], torch.zeros(1).to(self.alphas_cumprod)))
        self.timesteps = torch.arange(num_diffusion_timesteps - 1, -1, -1)

    @torch.no_grad()
    def sample(
        self,
        unet: UNet2DModel,
        batch_size: int,
        in_channels: int,
        sample_size: int,
    ):
        images = torch.randn((batch_size, in_channels, sample_size, sample_size), device=unet.device)

        betas = self.betas.to(unet.device)
        alphas = self.alphas.to(unet.device)
        alphas_cumprod = self.alphas_cumprod.to(unet.device)
        alphas_cumprod_prev = self.alphas_cumprod_prev.to(unet.device)
        timesteps = self.timesteps.to(unet.device)

        sqrt_recip_alphas_cumprod = (1.0 / alphas_cumprod) ** 0.5
        sqrt_recipm1_alphas_cumprod = (1.0 / alphas_cumprod - 1.0) ** 0.5

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_log_variance_clipped = torch.log(torch.concat((posterior_variance[1:2], posterior_variance[1:])))
        posterior_mean_coef1 = betas * alphas_cumprod_prev ** 0.5 / (1.0 - alphas_cumprod)
        posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * alphas ** 0.5 / (1.0 - alphas_cumprod)

        for timestep in tqdm(timesteps, desc='Sampling'):
            _extract = partial(extract, timesteps=timestep, broadcast_shape=images.shape)
            preds: torch.Tensor = unet(images, timestep).sample
            pred_noises, pred_vars = torch.split(preds, in_channels, dim=1)

            # mean of p(x_{t-1}|x_t), same to DDPM
            x_0 = _extract(sqrt_recip_alphas_cumprod) * images - _extract(sqrt_recipm1_alphas_cumprod) * pred_noises
            mean = _extract(posterior_mean_coef1) * x_0.clamp(-1, 1) + _extract(posterior_mean_coef2) * images

            # variance of p(x_{t-1}|x_t), learned
            if timestep > 0:
                min_log = _extract(posterior_log_variance_clipped)
                max_log = _extract(torch.log(betas))
                frac = (pred_vars + 1.0) / 2.0
                log_variance = frac * max_log + (1.0 - frac) * min_log
                stddev = torch.exp(0.5 * log_variance)
            else:
                stddev = torch.zeros_like(timestep)
            
            epsilon = torch.randn_like(images)
            images = mean + stddev * epsilon
        images = (images / 2.0 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
        return images

model = UNet2DModel.from_pretrained('iddpm-animefaces-64').cuda()
ddpm = IDDPM()
images = ddpm.sample(model, 32, 3, 64)

from diffusers.utils import make_image_grid, numpy_to_pil
image_grid = make_image_grid(numpy_to_pil(images), rows=4, cols=8)
image_grid.save('iddpm-sample-results.png')

# config

from dataclasses import dataclass

import torch.types

@dataclass
class TrainingConfig:
    image_size = 64
    train_batch_size = 16
    eval_batch_size = 16
    num_epochs = 50
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    mixed_precision = "fp16"
    output_dir = "iddpm-animefaces-64"
    overwrite_output_dir = True

config = TrainingConfig()

# data

from datasets import load_dataset

dataset = load_dataset("huggan/anime-faces", split="train")
dataset = dataset.select(range(21551))

## preprocess

from torchvision import transforms

def get_transform():
    preprocess = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    def transform(samples):
        images = [preprocess(img.convert("RGB")) for img in samples["image"]]
        return dict(images=images)
    return transform

dataset.set_transform(get_transform())

## dataloader

from torch.utils.data import DataLoader

dataloader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

# unet model

from diffusers import UNet2DModel

model = UNet2DModel(
    sample_size=config.image_size,
    in_channels=3,
    out_channels=6,
    layers_per_block=2,
    block_out_channels=(128, 128, 256, 256, 512, 512),
    down_block_types=(
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
)

# core

import math
import torch
from tqdm import tqdm
from functools import partial

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

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ):
        alphas_cumprod = self.alphas_cumprod.to(original_samples)
        noise = noise.to(original_samples.device)
        timesteps = timesteps.to(original_samples.device)

        # \sqrt{\bar\alpha_t}
        sqrt_alpha_prod = alphas_cumprod[timesteps].flatten() ** 0.5
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
        
        # \sqrt{1 - \bar\alpha_t}
        sqrt_one_minus_alpha_prod = (1.0 - alphas_cumprod[timesteps]).flatten() ** 0.5
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        return sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise

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
    
# training

from accelerate import Accelerator
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.utils import make_image_grid, numpy_to_pil
import os

## training losses

def pred_mean_logvar(
    iddpm: IDDPM,
    pred_noises: torch.Tensor,
    pred_vars: torch.Tensor,
    noisy_images: torch.Tensor,
    timesteps: torch.Tensor,
):
    betas = iddpm.betas.to(timesteps.device)
    alphas = iddpm.alphas.to(timesteps.device)
    alphas_cumprod = iddpm.alphas_cumprod.to(timesteps.device)
    alphas_cumprod_prev = iddpm.alphas_cumprod_prev.to(timesteps.device)

    sqrt_recip_alphas_cumprod = (1.0 / alphas_cumprod) ** 0.5
    sqrt_recipm1_alphas_cumprod = (1.0 / alphas_cumprod - 1.0) ** 0.5

    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    posterior_log_variance_clipped = torch.log(torch.concat((posterior_variance[1:2], posterior_variance[1:])))
    posterior_mean_coef1 = betas * alphas_cumprod_prev ** 0.5 / (1.0 - alphas_cumprod)
    posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * alphas ** 0.5 / (1.0 - alphas_cumprod)

    _extract = partial(extract, timesteps=timesteps, broadcast_shape=noisy_images.shape)
    # mean of p(x_{t-1}|x_t), same to DDPM
    x_0 = _extract(sqrt_recip_alphas_cumprod) * noisy_images - _extract(sqrt_recipm1_alphas_cumprod) * pred_noises
    mean = _extract(posterior_mean_coef1) * x_0.clamp(-1, 1) + _extract(posterior_mean_coef2) * noisy_images
    # variance of p(x_{t-1}|x_t), learned
    min_log = _extract(posterior_log_variance_clipped)
    max_log = _extract(torch.log(betas))
    frac = (pred_vars + 1.0) / 2.0
    log_variance = frac * max_log + (1.0 - frac) * min_log

    return mean, log_variance

def true_mean_logvar(
    iddpm: IDDPM,
    clean_images: torch.Tensor,
    noisy_images: torch.Tensor,
    timesteps: torch.Tensor,
):
    betas = iddpm.betas.to(timesteps.device)
    alphas = iddpm.alphas.to(timesteps.device)
    alphas_cumprod = iddpm.alphas_cumprod.to(timesteps.device)
    alphas_cumprod_prev = iddpm.alphas_cumprod_prev.to(timesteps.device)

    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    posterior_log_variance_clipped = torch.log(torch.concat((posterior_variance[1:2], posterior_variance[1:])))
    posterior_mean_coef1 = betas * alphas_cumprod_prev ** 0.5 / (1.0 - alphas_cumprod)
    posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * alphas ** 0.5 / (1.0 - alphas_cumprod)

    _extract = partial(extract, timesteps=timesteps, broadcast_shape=noisy_images.shape)
    posterior_mean = _extract(posterior_mean_coef1) * clean_images + _extract(posterior_mean_coef2) * noisy_images
    posterior_log_variance_clipped = _extract(posterior_log_variance_clipped)

    return posterior_mean, posterior_log_variance_clipped

def gaussian_kl_divergence(
    mean_1: torch.Tensor,
    logvar_1: torch.Tensor,
    mean_2: torch.Tensor,
    logvar_2: torch.Tensor,
):
    return 0.5 * (
        -1.0
        + logvar_2
        - logvar_1
        + torch.exp(logvar_1 - logvar_2)
        + ((mean_1 - mean_2) ** 2) * torch.exp(-logvar_2)
    )

def approx_standard_normal_cdf(x):
    return 0.5 * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

def gaussian_nll(
    clean_images: torch.Tensor,
    pred_mean: torch.Tensor,
    pred_logvar: torch.Tensor,
):
    # stdnorm = Normal(torch.Tensor([0.0]), torch.Tensor([1.0]))
    centered_x = clean_images - pred_mean
    inv_stdv = torch.exp(-pred_logvar)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in).clamp_min(1e-12)
    # cdf_plus = stdnorm.cdf(plus_in).clamp_min(1e-12)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in).clamp_min(1e-12)
    # cdf_min = stdnorm.cdf(min_in).clamp_min(1e-12)
    cdf_delta = (cdf_plus - cdf_min).clamp_min(1e-12)
    log_cdf_plus = torch.log(cdf_plus)
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp_min(1e-12))
    log_probs = torch.log(cdf_delta.clamp_min(1e-12))
    log_probs[clean_images < -0.999] = log_cdf_plus[clean_images < -0.999]
    log_probs[clean_images > 0.999] = log_one_minus_cdf_min[clean_images > 0.999]
    return log_probs

def vlb_loss(
    iddpm: IDDPM,
    pred_noises: torch.Tensor,
    pred_vars: torch.Tensor,
    clean_images: torch.Tensor, # x_0
    noisy_images: torch.Tensor, # x_t
    timesteps: torch.Tensor,    # t
):
    # 1. calculate predicted mean and log var, same to sampling
    pred_mean, pred_logvar = pred_mean_logvar(iddpm, pred_noises, pred_vars, noisy_images, timesteps)
    # 2. calculate the true mean and log var with q(x_{t-1}|x_t,x_0)
    true_mean, true_logvar = true_mean_logvar(iddpm, clean_images, noisy_images, timesteps)
    # 3. calculate the KL divergences
    kl = gaussian_kl_divergence(true_mean, true_logvar, pred_mean, pred_logvar)
    kl = kl.mean(dim=list(range(1, len(kl.shape)))) / math.log(2.0)
    # 4. calculate the NLL
    nll = gaussian_nll(clean_images, pred_mean, pred_logvar * 0.5)
    nll = nll.mean(dim=list(range(1, len(nll.shape)))) / math.log(2.0)
    # 5. gather results
    results = torch.where(timesteps == 0, nll, kl)
    return results

def training_losses(
    iddpm: IDDPM,
    model: UNet2DModel,
    clean_images: torch.Tensor,
    noise: torch.Tensor,
    noisy_images: torch.Tensor,
    timesteps: torch.Tensor,
    vlb_weight: float = 1e-3,
) -> torch.Tensor:
    _, channels, _, _ = noisy_images.shape
    pred: torch.Tensor = model(noisy_images, timesteps, return_dict=False)[0]
    pred_noises, pred_vars = torch.split(pred, channels, dim=1)
    # 1. L_simple
    l_simple = (pred_noises - noise) ** 2
    l_simple = l_simple.mean(dim=list(range(1, len(l_simple.shape))))
    # 2. L_vlb
    l_vlb = vlb_loss(iddpm, pred_noises.detach(), pred_vars, clean_images, noisy_images, timesteps)
    return l_simple + vlb_weight * l_vlb

## importance sampler

import numpy as np
from typing import List
from torch.distributed.nn import dist

class ImportanceSampler:

    def __init__(
        self,
        num_diffusion_timesteps: int = 1000,
        history_per_term: int = 10, 
    ):
        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.history_per_term = history_per_term
        self.uniform_prob = 1.0 / num_diffusion_timesteps
        self.loss_history = np.zeros([num_diffusion_timesteps, history_per_term], dtype=np.float64)
        self.loss_counts = np.zeros([num_diffusion_timesteps], dtype=int)

    def sample(self, batch_size: int):
        weights = self.weights
        prob = weights / np.sum(weights)
        timesteps = np.random.choice(self.num_diffusion_timesteps, size=(batch_size,), p=prob)
        weights = 1.0 / (self.num_diffusion_timesteps * prob[timesteps])
        return torch.from_numpy(timesteps).long(), torch.from_numpy(weights).float()
    
    @property
    def weights(self):
        if not np.all(self.loss_counts == self.history_per_term):
            return np.ones([self.num_diffusion_timesteps], dtype=np.float64)
        weights = np.sqrt(np.mean(self.loss_history ** 2, axis=-1))
        weights /= np.sum(weights)
        weights *= 1.0 - self.uniform_prob
        weights += self.uniform_prob / len(weights)
        return weights
    
    def update(self, timesteps: torch.Tensor, losses: torch.Tensor):
        # collect
        if dist.is_initialized():
            world_size = dist.get_world_size()
            # get batch sizes for padding to the maximum bs
            batch_sizes = [torch.tensor([0], dtype=torch.int32, device=timesteps.device) for _ in range(world_size)]
            dist.all_gather(batch_sizes, torch.full_like(batch_sizes[0], timesteps.size(0)))
            max_batch_size = max([bs.item() for bs in batch_sizes])
            # gather all timesteps and losses
            timestep_batches: List[torch.Tensor] = [torch.zeros(max_batch_size).to(timesteps) for _ in range(world_size)]
            loss_batches: List[torch.Tensor] = [torch.zeros(max_batch_size).to(losses) for _ in range(world_size)]
            dist.all_gather(timestep_batches, timesteps)
            dist.all_gather(loss_batches, losses)
            all_timesteps = [ts.item() for ts_batch, bs in zip(timestep_batches, batch_sizes) for ts in ts_batch[:bs]]
            all_losses = [loss.item() for loss_batch, bs in zip(loss_batches, batch_sizes) for loss in loss_batch[:bs]]
        else:
            all_timesteps = timesteps.tolist()
            all_losses = losses.tolist()
        # update
        for timestep, loss in zip(all_timesteps, all_losses):
            if self.loss_counts[timestep] == self.history_per_term:
                self.loss_history[timestep, :-1] = self.loss_history[timestep, 1:]
                self.loss_history[timestep, -1] = loss
            else:
                self.loss_history[timestep, self.loss_counts[timestep]] = loss
                self.loss_counts[timestep] += 1

model = model.cuda()
iddpm = IDDPM()
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(dataloader) * config.num_epochs),
)
importance_sampler = ImportanceSampler()
accelerator = Accelerator(
    mixed_precision=config.mixed_precision,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    log_with="tensorboard",
    project_dir=os.path.join(config.output_dir, "logs"),
)
model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, dataloader, lr_scheduler
)
global_step = 0
for epoch in range(config.num_epochs):
    progress_bar = tqdm(total=len(dataloader), disable=not accelerator.is_local_main_process, desc=f'Epoch {epoch}')

    for step, batch in enumerate(dataloader):
        clean_images = batch["images"]
        # Sample noise to add to the images
        noise = torch.randn(clean_images.shape, device=clean_images.device)
        bs = clean_images.shape[0]
        # Sample a random timestep for each image
        timesteps, weights = importance_sampler.sample(bs)
        timesteps = timesteps.to(clean_images.device)
        weights = weights.to(clean_images.device)
        # Add noise to the clean images according to the noise magnitude at each timestep
        noisy_images = iddpm.add_noise(clean_images, noise, timesteps)
        with accelerator.accumulate(model):
            # Predict the noise residual
            losses = training_losses(iddpm, model, clean_images, noise, noisy_images, timesteps)
            importance_sampler.update(timesteps, losses)
            loss = (losses * weights).mean()
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        progress_bar.update(1)
        logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
        progress_bar.set_postfix(**logs)
        accelerator.log(logs, step=global_step)
        global_step += 1

    if accelerator.is_main_process:
        # evaluate
        images = iddpm.sample(model, config.eval_batch_size, 3, config.image_size)
        image_grid = make_image_grid(numpy_to_pil(images), rows=4, cols=4)
        samples_dir = os.path.join(config.output_dir, 'samples')
        os.makedirs(samples_dir, exist_ok=True)
        image_grid.save(os.path.join(samples_dir, f'{global_step}.png'))
        # save models
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model.module.save_pretrained(config.output_dir)
        else:
            model.save_pretrained(config.output_dir)

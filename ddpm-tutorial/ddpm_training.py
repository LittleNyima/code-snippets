# config

from dataclasses import dataclass

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
    output_dir = "ddpm-animefaces-64"
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

from diffusers import UNet2DModel

model = UNet2DModel(
    sample_size=config.image_size,
    in_channels=3,
    out_channels=3,
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

import torch
from tqdm import tqdm

class DDPM:
    def __init__(
        self,
        num_train_timesteps:int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.timesteps = torch.arange(num_train_timesteps - 1, -1, -1)
    
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ):
        alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device ,dtype=original_samples.dtype)
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
        betas = self.betas.to(unet.device)
        alphas = self.alphas.to(unet.device)
        alphas_cumprod = self.alphas_cumprod.to(unet.device)
        timesteps = self.timesteps.to(unet.device)
        images = torch.randn((batch_size, in_channels, sample_size, sample_size), device=unet.device)
        for timestep in tqdm(timesteps, desc='Sampling'):
            pred_noise: torch.Tensor = unet(images, timestep).sample

            # mean of q(x_{t-1}|x_t)
            alpha_t = alphas[timestep]
            alpha_cumprod_t = alphas_cumprod[timestep]
            sqrt_alpha_t = alpha_t ** 0.5
            one_minus_alpha_t = 1.0 - alpha_t
            sqrt_one_minus_alpha_cumprod_t = (1 - alpha_cumprod_t) ** 0.5
            mean = (images - one_minus_alpha_t / sqrt_one_minus_alpha_cumprod_t * pred_noise) / sqrt_alpha_t
            
            # variance of q(x_{t-1}|x_t)
            if timestep > 0:
                beta_t = betas[timestep]
                one_minus_alpha_cumprod_t_minus_one = 1.0 - alphas_cumprod[timestep - 1]
                one_divided_by_sigma_square = alpha_t / beta_t + 1.0 / one_minus_alpha_cumprod_t_minus_one
                variance = (1.0 / one_divided_by_sigma_square) ** 0.5
            else:
                variance = torch.zeros_like(timestep)
            
            epsilon = torch.randn_like(images)
            images = mean + variance * epsilon
        images = (images / 2.0 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
        return images
    
# training

from accelerate import Accelerator
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.utils import make_image_grid, numpy_to_pil
import torch.nn.functional as F
import os

model = model.cuda()
ddpm = DDPM()
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(dataloader) * config.num_epochs),
)
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
        timesteps = torch.randint(
            0, ddpm.num_train_timesteps, (bs,), device=clean_images.device,
            dtype=torch.int64
        )
        # Add noise to the clean images according to the noise magnitude at each timestep
        noisy_images = ddpm.add_noise(clean_images, noise, timesteps)
        with accelerator.accumulate(model):
            # Predict the noise residual
            noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
            loss = F.mse_loss(noise_pred, noise)
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
        images = ddpm.sample(model, config.eval_batch_size, 3, config.image_size)
        image_grid = make_image_grid(numpy_to_pil(images), rows=4, cols=4)
        samples_dir = os.path.join(config.output_dir, 'samples')
        os.makedirs(samples_dir, exist_ok=True)
        image_grid.save(os.path.join(samples_dir, f'{global_step}.png'))
        # save models
        model.save_pretrained(config.output_dir)

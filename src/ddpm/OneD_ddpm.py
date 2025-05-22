import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def cosine_beta_schedule(timesteps, s = 0.008, device="cpu"):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float32)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    
    return torch.clip(betas, 0, 0.999)

class DDPM:
    def __init__(self,
                 model: nn.Module,
                 timesteps: int = 1000,
                 beta_start: float = 0.0001,
                 beta_end: float = 0.02,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):

        self.model = model
        self.timesteps = timesteps
        self.device = device
        self.model.to(self.device)

        #self.betas = torch.linspace(beta_start, beta_end, timesteps, device=self.device)
        self.betas = cosine_beta_schedule(timesteps, device=self.device).to(self.device)
        
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Calculations for diffusion q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        # Clip variance to prevent it from being 0 (due to F.pad for alphas_cumprod_prev at t=0)
        self.posterior_variance = torch.clamp(self.posterior_variance, min=1e-20)


    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start, device=self.device)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        noisy_x = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        return noisy_x

    def p_losses(self, x_start, t, noise=None, loss_type="l2", c=None):
        if noise is None:
            noise = torch.randn_like(x_start, device=self.device)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = self.model(x_noisy, t, c) # predicts the noise

        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        else:
            raise NotImplementedError(f"Loss type '{loss_type}' not implemented.")
        return loss

    @torch.no_grad()
    def p_sample(self, x_t, t_tensor, t_index, c):
        betas_t = extract(self.betas, t_tensor, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t_tensor, x_t.shape)
        sqrt_recip_alphas_t = extract(torch.sqrt(1.0 / self.alphas), t_tensor, x_t.shape)

        # Equation 11 in the DDPM paper
        # Use model to predict the mean
        predicted_noise = self.model(x_t, t_tensor, c)
        model_mean = sqrt_recip_alphas_t * (x_t - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)

        if t_index == 0:
            return model_mean # No noise added at the last step
        else:
            posterior_variance_t = extract(self.posterior_variance, t_tensor, x_t.shape)
            noise = torch.randn_like(x_t)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, batch_size, seq_len, features=1, c=None):
        self.model.eval()

        # Start from pure noise (x_T)
        shape = (batch_size, features, seq_len)
        img = torch.randn(shape, device=self.device)
        #imgs = [] # To store intermediate images if desired

        for i in tqdm(reversed(range(0, self.timesteps)), desc='Sampling loop', total=self.timesteps):
            t_tensor = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            img = self.p_sample(img, t_tensor, i, c)
            # if i % 50 == 0: # save some intermediate steps
            #     imgs.append(img.cpu().numpy())
        self.model.train()
        return img #, imgs
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def linear_beta_schedule(timesteps, beta_start=1e-6, beta_end=2e-2):
    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(timesteps, s = 0.008):
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

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))



class DDPM(nn.Module):

    def __init__(self, denoising_network, timesteps=500, beta_start=1e-6, beta_end=2e-2, scheduler="cosine"):

        super().__init__()
        self.timesteps = timesteps
        self.denoising_net = denoising_network
        self.scheduler = scheduler

        
        if scheduler == "cosine":
            betas = cosine_beta_schedule(self.timesteps)
        else:
            betas = linear_beta_schedule(timesteps, beta_start, beta_end)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0) 
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)


        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)


        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))

        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        self.register_buffer('posterior_mean_coef2', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef1', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))


    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        # Get sqrt(alpha_bar_n) and sqrt(1 - alpha_bar_n) for the given timesteps t
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        # Calculate x_t = sqrt(alpha_bar_n)*x_0 + sqrt(1 - alpha_bar_n)*epsilon
        x_noisy = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        return x_noisy

    def predict_noise_from_start(self, x_t, t, x_start):
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return (x_t - sqrt_alphas_cumprod_t * x_start) / sqrt_one_minus_alphas_cumprod_t

    def predict_start_from_noise(self, x_t, t, noise):
        sqrt_recip_alphas_cumprod_t = extract(1. / self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1_alphas_cumprod_t = extract(torch.sqrt(1. / self.alphas_cumprod - 1), t, x_t.shape)

        pred_x_start = sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * noise
        return pred_x_start


    def p_mean_variance(self, x_t, t, y, clip_denoised=True):
        # Predict noise using the network: epsilon_theta(x_n, sqrt(alpha_bar_n), y)
        # Note: The paper passes sqrt(alpha_bar_n) as conditioning
        sqrt_alpha_bar_t = extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        pred_noise = self.denoising_net(x_t, sqrt_alpha_bar_t.squeeze(), y) # Pass sqrt_alpha_bar

        # Predict x_0 from the noise
        x_start_pred = self.predict_start_from_noise(x_t, t, pred_noise)

        if clip_denoised:
            # Clipping helps stabilize training/sampling
            x_start_pred.clamp_(-1., 1.)

        # Calculate the mean of p(x_{t-1} | x_t, x_0_pred)
        # This uses the formula derived from q(x_{t-1} | x_t, x_0)
        mean_coef1 = extract(self.posterior_mean_coef1, t, x_t.shape)
        mean_coef2 = extract(self.posterior_mean_coef2, t, x_t.shape)
        model_mean = mean_coef1 * x_start_pred + mean_coef2 * x_t

        # Get the variance and log variance
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)

        return model_mean, posterior_variance, posterior_log_variance_clipped

    @torch.no_grad()
    def p_sample(self, x_t, t, y, clip_denoised=True):
        # Calculate mean and variance of p(x_{t-1} | x_t)
        model_mean, _, model_log_variance = self.p_mean_variance(
            x_t=x_t, t=t, y=y, clip_denoised=clip_denoised
        )

        noise = torch.randn_like(x_t)
        # No noise added at timestep 0
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))

        # Calculate x_{t-1} = mean + sqrt(variance) * z * mask
        pred_img = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred_img

    @torch.no_grad()
    def sample(self, shape, cond_info):
        device = self.betas.device
        batch_size = shape[0]

        img = torch.randn(shape, device=device)

        for i in reversed(range(0, self.timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            img = self.p_sample(img, t, cond_info) 

        return img

    def p_losses(self, x_start, t, y, noise=None, loss_type="l2"):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

 
        sqrt_alpha_bar_t = extract(self.sqrt_alphas_cumprod, t, x_noisy.shape)
        predicted_noise = self.denoising_net(x_noisy, sqrt_alpha_bar_t.squeeze(), y)

        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        else:
            raise NotImplementedError(f"Loss type '{loss_type}' not implemented.")

        return loss

    def forward(self, x_start, y):
        batch_size = x_start.shape[0]
        device = x_start.device

        t = torch.randint(0, self.timesteps, (batch_size,), device=device).long()

        return self.p_losses(x_start, t, y, loss_type="l2")
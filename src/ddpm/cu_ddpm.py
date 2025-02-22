import numpy as np

import torch


def linear_schedule(beta1, beta2, timesteps, device):
    b_t = (beta2 - beta1) * torch.linspace(0, 1, timesteps + 1, device=device) + beta1
    a_t = 1 - b_t
    ab_t = torch.cumsum(a_t.log(), dim=0).exp()
    ab_t[0] = 1
    
    return a_t, b_t, ab_t
    
def perturb_input(x, t, noise, beta1, beta2, timesteps, device):
    a_t, b_t, ab_t = linear_schedule(beta1, beta2, timesteps, device)
    return ab_t.sqrt()[t, None, None, None] * x + (1 - ab_t[t, None, None, None]) * noise

def denoise_add_noise(x, t, pred_noise, beta1, beta2, timesteps, device, z=None):
    a_t, b_t, ab_t = linear_schedule(beta1, beta2, timesteps, device)
    
    if z is None:
        z = torch.randn_like(x)
    noise = b_t.sqrt()[t] * z
    mean = (x - pred_noise * ((1 - a_t[t]) / (1 - ab_t[t]).sqrt())) / a_t[t].sqrt()
    return mean + noise

@torch.no_grad()
def sample_ddpm_context(n_sample, context, channels, height, device, timesteps, nn_model,beta1, beta2, save_rate=20):
    # x_T ~ N(0, 1), sample initial noise
    samples = torch.randn(n_sample, channels, height, height).to(device)

    # array to keep track of generated steps for plotting
    intermediate = []
    for i in range(timesteps, 0, -1):
        print(f'sampling timestep {i:3d}', end='\r')

        # reshape time tensor
        t = torch.tensor([i / timesteps])[:, None, None, None].to(device)

        # sample some random noise to inject back in. For i = 1, don't add back in noise
        z = torch.randn_like(samples) if i > 1 else 0

        eps = nn_model(samples, t, c=context)    # predict noise e_(x_t,t, ctx)
        samples = denoise_add_noise(samples, i, eps, beta1, beta2, timesteps, device, z=z)
        if i % save_rate==0 or i==timesteps or i<8:
            intermediate.append(samples.detach().cpu().numpy())

    intermediate = np.stack(intermediate)
    return samples, intermediate
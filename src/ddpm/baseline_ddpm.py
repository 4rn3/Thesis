import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def linear_beta_schedule(timesteps, beta_start=1e-6, beta_end=2e-2):
    """
    Generates a linear beta schedule.

    Args:
        timesteps (int): Number of diffusion steps (N in the paper).
        beta_start (float): Starting beta value.
        beta_end (float): Ending beta value.

    Returns:
        torch.Tensor: Tensor of beta values.
    """
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
    """
    Extracts the t-th value from tensor a and reshapes it to match x_shape
    for broadcasting.

    Args:
        a (torch.Tensor): Tensor to extract from (e.g., alphas, betas).
        t (torch.Tensor): Timestep indices (batch size).
        x_shape (tuple): Shape of the target tensor (e.g., input data).

    Returns:
        torch.Tensor: Extracted values reshaped for broadcasting.
    """
    batch_size = t.shape[0]
    # Get the value corresponding to the timestep t for each item in the batch
    out = a.gather(-1, t)
    # Reshape to (batch_size, 1, 1, ...) to allow broadcasting
    # For data shape (B, F, S), we need (B, 1, 1)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))



class DDPM(nn.Module):
    """
    Denoising Diffusion Probabilistic Model (DDPM).
    """
    def __init__(self, denoising_network, timesteps=500, beta_start=1e-6, beta_end=2e-2, scheduler="cosine"):
        """
        Initializes the DDPM model.

        Args:
            denoising_network (nn.Module): The neural network used for denoising (predicting noise).
            timesteps (int): Number of diffusion steps (N).
            beta_start (float): Starting value for the beta schedule.
            beta_end (float): Ending value for the beta schedule.
        """
        super().__init__()
        self.timesteps = timesteps
        self.denoising_net = denoising_network
        self.scheduler = scheduler
        # --- Diffusion Schedule ---
        # Calculate betas
        
        if scheduler == "cosine":
            betas = cosine_beta_schedule(self.timesteps)
        else:
            betas = linear_beta_schedule(timesteps, beta_start, beta_end)

        # Calculate alphas related terms (using notation from paper and Ho et al.)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0) # alpha_bar in paper notation
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0) # alpha_bar_{n-1}

        # Register buffers are tensors part of the model state, but not parameters
        # They are saved with the model's state_dict
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # --- Calculations for diffusion q(x_t | x_0) ---
        # sqrt(alpha_bar_n) in paper notation
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        # sqrt(1 - alpha_bar_n) in paper notation
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))

        # --- Calculations for posterior q(x_{t-1} | x_t, x_0) ---
        # This is not strictly needed for the simplified training objective (Eq. 6)
        # but useful if implementing the full VLB objective or alternative sampling.

        # --- Calculations for sampling p(x_{t-1} | x_t) ---
        # sigma_n^2 * I in paper notation (Eq. 4 and below Eq. 7)
        # variance = beta_n * (1 - alpha_bar_{n-1}) / (1 - alpha_bar_n)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # We clamp the variance to avoid issues where beta is 0
        self.register_buffer('posterior_variance', posterior_variance)
        # log(variance) for numerical stability
        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))

        # Coefficient for mean calculation in sampling: 1 / sqrt(alpha_n)
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        # Coefficient for mean calculation: beta_n / sqrt(1 - alpha_bar_n)
        self.register_buffer('posterior_mean_coef2', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
         # Coefficient for mean calculation: (1 - alpha_bar_{n-1}) * sqrt(alpha_n) / (1 - alpha_bar_n)
        self.register_buffer('posterior_mean_coef1', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))


    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion process: q(x_t | x_0).
        Calculates the noisy version x_t for a given x_0 at timestep t.
        Corresponds to Equation (3) in the paper.

        Args:
            x_start (torch.Tensor): The initial clean data (x_0), shape (B, F, S).
            t (torch.Tensor): Timesteps for each batch item, shape (B,).
            noise (torch.Tensor, optional): Optional noise tensor (epsilon). If None, generated. Shape (B, F, S).

        Returns:
            torch.Tensor: Noisy data x_t, shape (B, F, S).
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        # Get sqrt(alpha_bar_n) and sqrt(1 - alpha_bar_n) for the given timesteps t
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        # Calculate x_t = sqrt(alpha_bar_n)*x_0 + sqrt(1 - alpha_bar_n)*epsilon
        x_noisy = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        return x_noisy

    def predict_noise_from_start(self, x_t, t, x_start):
        """
        Predicts the noise epsilon given x_t and the original x_0.
        Rearranges Equation (3): epsilon = (x_t - sqrt(alpha_bar_n)*x_0) / sqrt(1 - alpha_bar_n)

        Args:
            x_t (torch.Tensor): Noisy data at timestep t, shape (B, F, S).
            t (torch.Tensor): Timesteps for each batch item, shape (B,).
            x_start (torch.Tensor): The initial clean data (x_0), shape (B, F, S).

        Returns:
            torch.Tensor: Predicted noise epsilon, shape (B, F, S).
        """
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return (x_t - sqrt_alphas_cumprod_t * x_start) / sqrt_one_minus_alphas_cumprod_t

    def predict_start_from_noise(self, x_t, t, noise):
        """
        Predicts the original x_0 given the noisy version x_t and the noise epsilon.
        Rearranges Equation (3): x_0 = (x_t - sqrt(1 - alpha_bar_n)*epsilon) / sqrt(alpha_bar_n)

        Args:
            x_t (torch.Tensor): Noisy data at timestep t, shape (B, F, S).
            t (torch.Tensor): Timesteps for each batch item, shape (B,).
            noise (torch.Tensor): Predicted noise epsilon_theta, shape (B, F, S).

        Returns:
            torch.Tensor: Predicted clean data x_0, shape (B, F, S).
        """
        sqrt_recip_alphas_cumprod_t = extract(1. / self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1_alphas_cumprod_t = extract(torch.sqrt(1. / self.alphas_cumprod - 1), t, x_t.shape)

        # Simplified formula derived from Ho et al. and DDIM paper
        # Equivalent to the rearrangement of Eq (3)
        pred_x_start = sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * noise
        return pred_x_start


    def p_mean_variance(self, x_t, t, y, clip_denoised=True):
        """
        Calculates the mean and variance of the reverse process distribution p(x_{t-1} | x_t).
        Uses the denoising network to predict the noise.
        Corresponds to the components used in Equation (7) sampling step.

        Args:
            x_t (torch.Tensor): Noisy data at timestep t, shape (B, F, S).
            t (torch.Tensor): Timesteps for each batch item, shape (B,). Should not contain 0.
            y (torch.Tensor): Conditioning information, shape (B, C).
            clip_denoised (bool): Whether to clip the predicted x_0 to [-1, 1] (assuming data is normalized).

        Returns:
            tuple: (predicted_mean, posterior_variance, posterior_log_variance)
                   Shapes: (B, F, S), (B, 1, 1), (B, 1, 1)
        """
        # Predict noise using the network: epsilon_theta(x_n, sqrt(alpha_bar_n), y)
        # Note: The paper passes sqrt(alpha_bar_n) as conditioning
        sqrt_alpha_bar_t = extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        pred_noise = self.denoising_net(x_t, sqrt_alpha_bar_t.squeeze(), y) # Pass sqrt_alpha_bar

        # Predict x_0 from the noise
        x_start_pred = self.predict_start_from_noise(x_t, t, pred_noise)

        if clip_denoised:
            # Clipping helps stabilize training/sampling, assuming data is in [-1, 1]
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
        """
        Single sampling step: Sample x_{t-1} from p(x_{t-1} | x_t).
        Corresponds to Equation (7).

        Args:
            x_t (torch.Tensor): Noisy data at timestep t, shape (B, F, S).
            t (torch.Tensor): Timesteps for each batch item, shape (B,). Should not contain 0.
            y (torch.Tensor): Conditioning information, shape (B, C).
            clip_denoised (bool): Whether to clip the predicted x_0.

        Returns:
            torch.Tensor: Sampled data x_{t-1}, shape (B, F, S).
        """
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
        """
        Generate samples starting from noise (x_N) and iteratively denoising.
        Corresponds to Algorithm 2.

        Args:
            shape (tuple): The desired shape of the output samples, e.g., (batch_size, features, seq_len).
            cond_info (torch.Tensor): Conditioning information for the batch, shape (batch_size, cond_features).

        Returns:
            torch.Tensor: Generated samples x_0, shape (batch_size, features, seq_len).
        """
        device = self.betas.device
        batch_size = shape[0]

        # Start from pure noise (x_N)
        img = torch.randn(shape, device=device)

        # Iteratively denoise from N down to 1
        for i in reversed(range(0, self.timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            img = self.p_sample(img, t, cond_info) # Pass conditioning info here

        # Final result is x_0
        return img

    def p_losses(self, x_start, t, y, noise=None, loss_type="l2"):
        """
        Calculate the training loss.
        Corresponds to the objective in Equation (6) (using L2 norm).

        Args:
            x_start (torch.Tensor): The initial clean data (x_0), shape (B, F, S).
            t (torch.Tensor): Timesteps for each batch item, shape (B,).
            y (torch.Tensor): Conditioning information, shape (B, C).
            noise (torch.Tensor, optional): The noise epsilon used for q_sample. If None, generated. Shape (B, F, S).
            loss_type (str): Type of loss ('l1' or 'l2'). Paper uses L2.

        Returns:
            torch.Tensor: The calculated loss (scalar).
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        # 1. Get noisy data x_t using q_sample (forward process)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # 2. Predict the noise using the denoising network
        # Pass sqrt(alpha_bar_n) as conditioning, consistent with p_mean_variance
        sqrt_alpha_bar_t = extract(self.sqrt_alphas_cumprod, t, x_noisy.shape)
        predicted_noise = self.denoising_net(x_noisy, sqrt_alpha_bar_t.squeeze(), y)

        # 3. Calculate the loss between the predicted noise and the actual noise
        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise) # MSELoss is L2 norm squared / num_elements
        else:
            raise NotImplementedError(f"Loss type '{loss_type}' not implemented.")

        return loss

    def forward(self, x_start, y):
        """
        Forward pass for training. Samples timesteps and calculates loss.

        Args:
            x_start (torch.Tensor): Batch of clean data (x_0), shape (B, F, S).
            y (torch.Tensor): Conditioning information for the batch, shape (B, C).

        Returns:
            torch.Tensor: The calculated loss for the batch.
        """
        batch_size = x_start.shape[0]
        device = x_start.device

        # Sample timesteps uniformly for each item in the batch
        # Paper samples n from {2, ..., N}, then sqrt(alpha_bar) uniformly between sqrt(alpha_bar_{n-1}) and sqrt(alpha_bar_n)
        # Simpler approach (common in many implementations): Sample t uniformly from {0, ..., N-1}
        t = torch.randint(0, self.timesteps, (batch_size,), device=device).long()

        # Calculate loss
        return self.p_losses(x_start, t, y, loss_type="l2")
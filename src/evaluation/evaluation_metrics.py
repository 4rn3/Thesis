from scipy.stats import wasserstein_distance
from scipy.stats import kurtosis, skew
from statsmodels.tsa.stattools import acf
import torch
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np
import seaborn as sns
import imageio.v2 as imageio
import os


def make_gif_from_images(image_paths, output_path="kde_progression.gif", fps=2):
    frames = [imageio.imread(p) for p in image_paths]
    imageio.mimsave(output_path, frames, fps=fps)

def plot_kde_samples(generated_samples, real_samples, num_samples=100000, random_state=42, reduction="mean", show=True, fpath="", epoch=None):
    if isinstance(real_samples, torch.Tensor):
        real_samples = real_samples.cpu().numpy()
    if isinstance(generated_samples, torch.Tensor):
        generated_samples = generated_samples.cpu().numpy()

    np.random.seed(random_state)
    batch_size, num_customers, seq_len = real_samples.shape
    total_customers = batch_size * num_customers
    indices = np.random.choice(total_customers, size=min(num_samples, total_customers), replace=False)
    
    real_flat = real_samples.reshape(-1, seq_len)
    gen_flat = generated_samples.reshape(-1, seq_len)
    
    if reduction == "mean":
        real_1d = real_flat[indices].mean(axis=1)
        gen_1d = gen_flat[indices].mean(axis=1)
    elif reduction == "sum":
        real_1d = real_flat[indices].sum(axis=1)
        gen_1d = gen_flat[indices].sum(axis=1)
    elif reduction == "first":
        real_1d = real_flat[indices][:, 0]
        gen_1d = gen_flat[indices][:, 0]
    else:
        raise ValueError("Unsupported reduction method")

    real_kde = gaussian_kde(real_1d)
    gen_kde = gaussian_kde(gen_1d)
    x_vals = np.linspace(min(real_1d.min(), gen_1d.min()), max(real_1d.max(), gen_1d.max()), 1000)

    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, real_kde(x_vals), label='Real Samples', color='blue', lw=2)
    plt.plot(x_vals, gen_kde(x_vals), label='Generated Samples', color='orange', lw=2)
    plt.fill_between(x_vals, 0, real_kde(x_vals), color='blue', alpha=0.3)
    plt.fill_between(x_vals, 0, gen_kde(x_vals), color='orange', alpha=0.3)
    
    if epoch == None:
        plt.title(f'KDE Comparison (reduction: {reduction})')
    else:
        plt.title(f'KDE Comparison Epoch:{epoch} (reduction: {reduction})')
        
    plt.xlabel('Reduced Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fpath)
    if show:
        plt.show()
    
    plt.close()
    
def calculate_stat_metrics(data_array, current_batch_size):
    means, std_devs, kurtoses, skews_list, mins, maxs = [], [], [], [], [], []

    for i in range(current_batch_size):
        sample_data = data_array[i, :, :]

        means.append(sample_data.mean())
        std_devs.append(sample_data.std())
        kurtoses.append(kurtosis(sample_data, axis=None))
        skews_list.append(skew(sample_data, axis=None))
        mins.append(sample_data.min())
        maxs.append(sample_data.max())

    return means, std_devs, kurtoses, skews_list, mins, maxs

def plot_metric(real, gen, feature, bins=50, save_path=None):
    if not isinstance(real, np.ndarray):
        real = np.array(real)
    if not isinstance(gen, np.ndarray):
        gen = np.array(gen)
    
    distance = wasserstein_distance(real, gen)
    
    plt.figure(figsize=(10, 6))
    
    combined_min = min(np.min(real), np.min(gen))
    combined_max = max(np.max(real), np.max(gen))


    if combined_min == combined_max:
        plot_range = (combined_min - 0.5, combined_max + 0.5)
        effective_bins = max(1, int(bins / 10))
    else:
        plot_range = (combined_min, combined_max)
        effective_bins = bins
        
    sns.histplot(real, color="skyblue", label=f'Real Data - {feature}', kde=True, stat="density", element="step", fill=True, alpha=0.6)
    sns.histplot(gen, color="red", label=f'Generated Data - {feature}', kde=True, stat="density", element="step", fill=True, alpha=0.6)

    plt.title(f'Distribution of "{feature}"\nWasserstein Distance: {distance:.4f}')
    plt.xlabel(f'{feature} Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"Plot for '{feature}' saved to {save_path}")
        except Exception as e:
            print(f"Error saving plot for '{feature}' to {save_path}: {e}")
    
    plt.show()

    return distance

def calc_acf(data, lag):
    acfs = []
    for i in range(data.shape[0]):
        ts = data[i, :, :]
        ts = ts.squeeze()
        acf_values = acf(ts, nlags=lag)
        acfs.append(acf_values[lag])
    
    return acfs   

def plot_acfs(real, gen, lag, save_path=None):
    plt.figure(figsize=(10, 6))
    
    sns.histplot(real, color="skyblue", label=f'Real Data (Lag {lag})', kde=True, stat="density", element="step", fill=True, alpha=0.6)
    sns.histplot(gen, color="red", label=f'Generated Data (Lag {lag})', kde=True, stat="density", element="step", fill=True, alpha=0.6)
    
    plt.title(f'Distribution of ACF Values at Lag {lag}')
    plt.xlabel(f'ACF Value at Lag {lag}')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(axis='y', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
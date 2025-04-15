import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from scipy.spatial.distance import jensenshannon
import torch
from IPython.display import HTML
from matplotlib.animation import PillowWriter


class KDEProgressionAnimator:
    def __init__(self, real_samples, projected_snapshots):

        #shape (batch, customers, seq_len)

        self.real_flat = real_samples.reshape(-1)
        self.snapshots = [gen.reshape(-1) for gen in projected_snapshots]
        self.real_kde = gaussian_kde(self.real_flat)

        self.x_min = min(np.min(self.real_flat), *(np.min(s) for s in self.snapshots))
        self.x_max = max(np.max(self.real_flat), *(np.max(s) for s in self.snapshots))
        self.x_grid = np.linspace(self.x_min, self.x_max, 200)

    def save(self, filename="kde_progression.gif", fps=2):
        anim = self.animate()
        anim.save(filename, writer=PillowWriter(fps=fps))

    def animate(self, interval=300):
        fig, ax = plt.subplots(figsize=(8, 5))
        real_line, = ax.plot(self.x_grid, self.real_kde(self.x_grid), label="Real", color="blue")
        gen_line, = ax.plot([], [], label="Generated", color="orange")
        title = ax.set_title("KDE Evolution")

        ax.legend()
        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(0, None)
        ax.set_xlabel("Projected Value")
        ax.set_ylabel("Density")
        ax.grid(True)

        def update(i):
            kde = gaussian_kde(self.snapshots[i])
            gen_line.set_data(self.x_grid, kde(self.x_grid))
            title.set_text(f"KDE Evolution - Step {i+1}")
            return gen_line, title

        anim = FuncAnimation(fig, update, frames=len(self.snapshots), interval=interval, blit=False)
        plt.close(fig)
        return HTML(anim.to_jshtml())
    
def vizual_comparison(generated_samples, real_samples, fpath):
    #shape (batch_size, num_customers, seq_len)

    # Parameters
    num_batches_to_plot = 4
    max_customers_per_batch = 500  # optional limit, all customers takes 5+min
    umap_sample_size = 5000        # total points to fit UMAP on, this decreases time a lot

    batch_indices = np.random.choice(real_samples.shape[0], size=num_batches_to_plot, replace=False)

    real_flat = real_samples.reshape(-1, real_samples.shape[-1])
    gen_flat = generated_samples.reshape(-1, generated_samples.shape[-1])
    combined = np.concatenate([real_flat, gen_flat], axis=0)

    # ===== PCA =====
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(combined)
    real_pca_all = pca_result[:real_flat.shape[0]].reshape(real_samples.shape[0], real_samples.shape[1], 2)
    gen_pca_all = pca_result[real_flat.shape[0]:].reshape(generated_samples.shape[0], generated_samples.shape[1], 2)

    # ===== UMAP =====
    selected_real = real_samples[batch_indices].reshape(-1, real_samples.shape[-1])
    selected_gen = generated_samples[batch_indices].reshape(-1, generated_samples.shape[-1])
    selected_combined = np.concatenate([selected_real, selected_gen], axis=0)

    if selected_combined.shape[0] > umap_sample_size:
        idx = np.random.choice(selected_combined.shape[0], size=umap_sample_size, replace=False)
        umap_fit_data = selected_combined[idx]
    else:
        umap_fit_data = selected_combined

    umap_model = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.3, random_state=42)
    umap_model.fit(umap_fit_data)

    # ===== t-SNE =====
    tsne_model = TSNE(n_components=2, random_state=42)

    fig, axes = plt.subplots(3, num_batches_to_plot, figsize=(5 * num_batches_to_plot, 5 * num_batches_to_plot), sharex=False)

    for i, batch_idx in enumerate(batch_indices):
        # ===== PCA =====
        ax_pca = axes[0, i]
        real_pca = real_pca_all[batch_idx][:max_customers_per_batch]
        gen_pca = gen_pca_all[batch_idx][:max_customers_per_batch]
        ax_pca.scatter(real_pca[:, 0], real_pca[:, 1], alpha=0.5, label='Real', s=10)
        ax_pca.scatter(gen_pca[:, 0], gen_pca[:, 1], alpha=0.5, label='Generated', s=10)
        ax_pca.set_title(f'PCA - Batch {batch_idx}')
        ax_pca.set_xlabel('PCA 1')
        ax_pca.set_ylabel('PCA 2')
        ax_pca.grid(True)
        if i == 0:
            ax_pca.legend()

        # ===== UMAP =====
        ax_umap = axes[1, i]
        real_umap_input = real_samples[batch_idx][:max_customers_per_batch]
        gen_umap_input = generated_samples[batch_idx][:max_customers_per_batch]
        real_umap = umap_model.transform(real_umap_input)
        gen_umap = umap_model.transform(gen_umap_input)

        ax_umap.scatter(real_umap[:, 0], real_umap[:, 1], alpha=0.5, label='Real', s=10)
        ax_umap.scatter(gen_umap[:, 0], gen_umap[:, 1], alpha=0.5, label='Generated', s=10)
        ax_umap.set_title(f'UMAP - Batch {batch_idx}')
        ax_umap.set_xlabel('UMAP 1')
        ax_umap.set_ylabel('UMAP 2')
        ax_umap.grid(True)
        if i == 0:
            ax_umap.legend()

        # ===== t-SNE =====
        ax_tsne = axes[2, i]
        real_tsne_input = real_samples[batch_idx][:max_customers_per_batch].reshape(-1, real_samples.shape[-1])
        gen_tsne_input = generated_samples[batch_idx][:max_customers_per_batch].reshape(-1, generated_samples.shape[-1])

        all_tsne_input = np.vstack([real_tsne_input, gen_tsne_input])
        tsne_result = tsne_model.fit_transform(all_tsne_input)

        real_tsne = tsne_result[:real_tsne_input.shape[0]]
        gen_tsne = tsne_result[real_tsne_input.shape[0]:]

        ax_tsne.scatter(real_tsne[:, 0], real_tsne[:, 1], alpha=0.5, label='Real', s=10)
        ax_tsne.scatter(gen_tsne[:, 0], gen_tsne[:, 1], alpha=0.5, label='Generated', s=10)
        ax_tsne.set_title(f't-SNE - Batch {batch_idx}')
        ax_tsne.set_xlabel('t-SNE 1')
        ax_tsne.set_ylabel('t-SNE 2')
        ax_tsne.grid(True)
        if i == 0:
            ax_tsne.legend()

    plt.tight_layout()
    plt.savefig(fpath)
    plt.show();

def plot_jsd_per_customer(generated_samples, real_samples, fpath):

    #shape (batch_size, num_customers, seq_len)
    
    if isinstance(real_samples, torch.Tensor):
        real_samples = real_samples.numpy()
    if isinstance(generated_samples, torch.Tensor):
        generated_samples = generated_samples.numpy()

    batch_size, num_customers, seq_len = real_samples.shape
    
    jsd_values = []

    for i in range(num_customers):
        real_customer_data = real_samples[:, i, :]
        gen_customer_data = generated_samples[:, i, :]

        real_customer_data = real_customer_data / np.sum(real_customer_data, axis=-1, keepdims=True)
        gen_customer_data = gen_customer_data / np.sum(gen_customer_data, axis=-1, keepdims=True)

        jsd = jensenshannon(np.mean(real_customer_data, axis=0), np.mean(gen_customer_data, axis=0))
        jsd_values.append(jsd)

    plt.figure(figsize=(8, 6))
    plt.hist(jsd_values, bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.title('Histogram of Jensen-Shannon Divergence per Customer')
    plt.xlabel('JSD Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(fpath)
    plt.show();
    
def plot_kde_samples(generated_samples, real_samples, num_samples=100000, random_state=42, reduction="mean", show=True, fpath=""):
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
    plt.title(f'KDE Comparison (reduction: {reduction})')
    plt.xlabel('Reduced Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fpath)
    if show:
        plt.show();
    
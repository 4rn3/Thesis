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
import imageio.v2 as imageio
from sklearn.metrics.pairwise import rbf_kernel


def make_gif_from_images(image_paths, output_path="kde_progression.gif", fps=2):
    frames = [imageio.imread(p) for p in image_paths]
    imageio.mimsave(output_path, frames, fps=fps)
    
def vizual_comparison(generated_samples, real_samples, fpath, num_batch=4, use_all_data=False):

    #shape (batch, customers, seq_len)
    
    def combined_plot():
            real_flat = real_samples.reshape(-1, real_samples.shape[-1])
            gen_flat = generated_samples.reshape(-1, generated_samples.shape[-1])
            combined = np.concatenate([real_flat, gen_flat], axis=0)

            # PCA
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(combined)
            real_pca = pca_result[:real_flat.shape[0]]
            gen_pca = pca_result[real_flat.shape[0]:]

            # UMAP
            umap_model = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.3, random_state=42)
            umap_result = umap_model.fit_transform(combined)
            real_umap = umap_result[:real_flat.shape[0]]
            gen_umap = umap_result[real_flat.shape[0]:]

            # t-SNE
            max_perplexity = min(30, combined.shape[0] - 1)
            perplexity = max(5, max_perplexity // 3)
            tsne_model = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            tsne_result = tsne_model.fit_transform(combined)
            real_tsne = tsne_result[:real_flat.shape[0]]
            gen_tsne = tsne_result[real_flat.shape[0]:]

            # Plotting
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            axes[0].scatter(real_pca[:, 0], real_pca[:, 1], alpha=0.5, label='Real', s=10)
            axes[0].scatter(gen_pca[:, 0], gen_pca[:, 1], alpha=0.5, label='Generated', s=10)
            axes[0].set_title('PCA')
            axes[0].legend()
            axes[0].grid(True)

            axes[1].scatter(real_umap[:, 0], real_umap[:, 1], alpha=0.5, label='Real', s=10)
            axes[1].scatter(gen_umap[:, 0], gen_umap[:, 1], alpha=0.5, label='Generated', s=10)
            axes[1].set_title('UMAP')
            axes[1].legend()
            axes[1].grid(True)

            axes[2].scatter(real_tsne[:, 0], real_tsne[:, 1], alpha=0.5, label='Real', s=10)
            axes[2].scatter(gen_tsne[:, 0], gen_tsne[:, 1], alpha=0.5, label='Generated', s=10)
            axes[2].set_title(f't-SNE (Perplexity={perplexity})')
            axes[2].legend()
            axes[2].grid(True)

            plt.tight_layout()
            plt.savefig(fpath)
            plt.show()

    # === Combined View ===
    if use_all_data:
        combined_plot()
        return

    # === Batch-by-Batch View ===
    max_customers_per_batch = min(500, real_samples.shape[1])
    batch_indices = np.random.choice(real_samples.shape[0], size=num_batch, replace=False)

    real_flat = real_samples.reshape(-1, real_samples.shape[-1])
    gen_flat = generated_samples.reshape(-1, generated_samples.shape[-1])
    combined = np.concatenate([real_flat, gen_flat], axis=0)

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(combined)
    real_pca_all = pca_result[:real_flat.shape[0]].reshape(real_samples.shape[0], real_samples.shape[1], 2)
    gen_pca_all = pca_result[real_flat.shape[0]:].reshape(generated_samples.shape[0], generated_samples.shape[1], 2)

    selected_real = real_samples[batch_indices].reshape(-1, real_samples.shape[-1])
    selected_gen = generated_samples[batch_indices].reshape(-1, generated_samples.shape[-1])
    selected_combined = np.concatenate([selected_real, selected_gen], axis=0)

    if selected_combined.shape[0] > 5000:
        idx = np.random.choice(selected_combined.shape[0], size=5000, replace=False)
        umap_fit_data = selected_combined[idx]
    else:
        umap_fit_data = selected_combined

    umap_model = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.3, random_state=42)
    umap_model.fit(umap_fit_data)

    fig, axes = plt.subplots(3, num_batch, figsize=(5 * num_batch, 15))

    for i, batch_idx in enumerate(batch_indices):
        # PCA
        real_pca = real_pca_all[batch_idx][:max_customers_per_batch]
        gen_pca = gen_pca_all[batch_idx][:max_customers_per_batch]
        ax = axes[0, i]
        ax.scatter(real_pca[:, 0], real_pca[:, 1], alpha=0.5, label='Real', s=10)
        ax.scatter(gen_pca[:, 0], gen_pca[:, 1], alpha=0.5, label='Generated', s=10)
        ax.set_title(f'PCA - Batch {batch_idx}')
        if i == 0: ax.legend()
        ax.grid(True)

        # UMAP
        real_umap_input = real_samples[batch_idx][:max_customers_per_batch]
        gen_umap_input = generated_samples[batch_idx][:max_customers_per_batch]
        real_umap = umap_model.transform(real_umap_input)
        gen_umap = umap_model.transform(gen_umap_input)
        ax = axes[1, i]
        ax.scatter(real_umap[:, 0], real_umap[:, 1], alpha=0.5, label='Real', s=10)
        ax.scatter(gen_umap[:, 0], gen_umap[:, 1], alpha=0.5, label='Generated', s=10)
        ax.set_title(f'UMAP - Batch {batch_idx}')
        if i == 0: ax.legend()
        ax.grid(True)

        # t-SNE
        real_tsne_input = real_samples[batch_idx][:max_customers_per_batch].reshape(-1, real_samples.shape[-1])
        gen_tsne_input = generated_samples[batch_idx][:max_customers_per_batch].reshape(-1, generated_samples.shape[-1])
        all_tsne_input = np.vstack([real_tsne_input, gen_tsne_input])
        max_perplexity = min(30, all_tsne_input.shape[0] - 1)
        perplexity = max(5, max_perplexity // 3)
        tsne_model = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        tsne_result = tsne_model.fit_transform(all_tsne_input)
        real_tsne = tsne_result[:real_tsne_input.shape[0]]
        gen_tsne = tsne_result[real_tsne_input.shape[0]:]
        ax = axes[2, i]
        ax.scatter(real_tsne[:, 0], real_tsne[:, 1], alpha=0.5, label='Real', s=10)
        ax.scatter(gen_tsne[:, 0], gen_tsne[:, 1], alpha=0.5, label='Generated', s=10)
        ax.set_title(f't-SNE - Batch {batch_idx}')
        if i == 0: ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(fpath)
    plt.show()


def plot_jsd_per_customer(generated_samples, real_samples, fpath):
    # shape: (batch_size, num_customers, seq_len)

    if isinstance(real_samples, torch.Tensor):
        real_samples = real_samples.numpy()
    if isinstance(generated_samples, torch.Tensor):
        generated_samples = generated_samples.numpy()

    batch_size, num_customers, seq_len = real_samples.shape

    jsd_values = []
    for i in range(num_customers):
        real_customer_data = real_samples[:, i, :]
        gen_customer_data = generated_samples[:, i, :]

        # Normalize, add small epsilon to avoid division by zero
        real_customer_data = real_customer_data / (np.sum(real_customer_data, axis=-1, keepdims=True) + 1e-8)
        gen_customer_data = gen_customer_data / (np.sum(gen_customer_data, axis=-1, keepdims=True) + 1e-8)

        real_mean = np.mean(real_customer_data, axis=0)
        gen_mean = np.mean(gen_customer_data, axis=0)

        if not np.all(np.isfinite(real_mean)) or not np.all(np.isfinite(gen_mean)):
            continue  # skip this customer if data is not valid

        jsd = jensenshannon(real_mean, gen_mean)
        if np.isfinite(jsd):
            jsd_values.append(jsd)

    if len(jsd_values) == 0:
        print("No valid JSD values to plot.")
        return

    # Plot only finite, valid values
    plt.figure(figsize=(8, 6))
    plt.hist(jsd_values, bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.title('Histogram of Jensen-Shannon Divergence per Customer')
    plt.xlabel('JSD Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(fpath)
    plt.show()
    
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

def mmd_histogram_per_customer(real, synth, gamma=None, bins=30, show=True, title="Customer-wise MMD Histogram", fpath=""):
    batch, num_customers, seq_len = real.shape
    if gamma is None:
        gamma = 1.0 / seq_len

    mmd_scores = []
    for customer in range(num_customers):
        real_c = real[:, customer, :]  # shape: (batch, seq_len)
        synth_c = synth[:, customer, :]  # shape: (batch, seq_len)

        XX = rbf_kernel(real_c, real_c, gamma=gamma)
        YY = rbf_kernel(synth_c, synth_c, gamma=gamma)
        XY = rbf_kernel(real_c, synth_c, gamma=gamma)

        mmd = np.mean(XX) + np.mean(YY) - 2 * np.mean(XY)
        mmd_scores.append(mmd)

    mmd_scores = np.array(mmd_scores)

    plt.figure(figsize=(8, 4))
    plt.hist(mmd_scores, bins=bins, edgecolor='black', color='skyblue')
    plt.xlabel("MMD per customer")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fpath)
    plt.show()
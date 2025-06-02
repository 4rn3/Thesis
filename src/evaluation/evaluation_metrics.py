from scipy.stats import wasserstein_distance
from scipy.stats import kurtosis, skew
from statsmodels.tsa.stattools import acf
import torch
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np
import seaborn as sns
import imageio.v2 as imageio
from sklearn.neighbors import NearestNeighbors
import os

plt.rcParams['font.size'] = 16 #14  Default font size for text elements
plt.rcParams['axes.labelsize'] = 18 #16 # Font size for x and y labels
plt.rcParams['axes.titlesize'] = 18 #20 # Font size for the title
plt.rcParams['xtick.labelsize'] = 14 #12 # Font size for x-axis tick labels
plt.rcParams['ytick.labelsize'] = 14 #12 # Font size for y-axis tick labels
plt.rcParams['legend.fontsize'] = 16 #14 # Font size for the legend
plt.rcParams['figure.titlesize'] = 22 #22 # Font size for the figure's suptitle

sns.set_theme(style="whitegrid")

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
    
    sns.set_theme(style="whitegrid", font_scale=1.5)

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
    sns.set_theme(style="whitegrid", font_scale=1.5)
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
    
def plot_query_and_neighbors(query_idx_in_sample1, 
                             original_sample1, 
                             original_sample2, 
                             neighbor_indices_in_sample2, 
                             k_neighbors,
                             save_path):

    query_series = original_sample1[query_idx_in_sample1].squeeze()
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(query_series, label=f'Query: Generated data[{query_idx_in_sample1}]', color='black', linewidth=2)
    
    for i in range(k_neighbors):
        neighbor_idx = neighbor_indices_in_sample2[i]
        neighbor_series = original_sample2[neighbor_idx].squeeze()
        plt.plot(neighbor_series, label=f'Neighbor {i+1}: real data[{neighbor_idx}]', linestyle='--')
        
    plt.title(f'Query Time Series (generated data[{query_idx_in_sample1}]) and its {k_neighbors} Closest Neighbors from real data')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()


def find_and_plot_similar_timeseries(
    real_data_all,
    samples_all,
    real_series_idx_to_plot,
    n_neighbors_to_find,
    metric='euclidean',
    color_palette_name="tab10",
    plot_filename="knn_similar_timeseries_plot.png",
):

    if real_data_all.ndim != 3 or samples_all.ndim != 3:
        raise ValueError("Input real_data_all and samples_all must be 3D arrays.")
    if real_data_all.shape[1:] != samples_all.shape[1:]:
        if real_data_all.shape[2] != samples_all.shape[2] or real_data_all.shape[1] != samples_all.shape[1]:
             raise ValueError("Features and sequence length must match between real and generated data.")

    batch_size_real, num_features, seq_len = real_data_all.shape
    batch_size_samples = samples_all.shape[0]

    if not (0 <= real_series_idx_to_plot < batch_size_real):
        raise ValueError(f"real_series_idx_to_plot ({real_series_idx_to_plot}) "
                         f"is out of bounds for real_data_all with batch size {batch_size_real}.")
    if not (0 < n_neighbors_to_find <= batch_size_samples):
        raise ValueError(f"n_neighbors_to_find ({n_neighbors_to_find}) must be between 1 and "
                         f"the number of samples ({batch_size_samples}).")

    if num_features > 1:
        print(f"Warning: Data has {num_features} features. Flattening all features and sequence length together.")
        real_data_flat = real_data_all.reshape(batch_size_real, num_features * seq_len)
        samples_flat = samples_all.reshape(batch_size_samples, num_features * seq_len)
    else:
        real_data_flat = real_data_all.reshape(batch_size_real, seq_len)
        samples_flat = samples_all.reshape(batch_size_samples, seq_len)

    print(f"Shape of reshaped real_data_flat: {real_data_flat.shape}")
    print(f"Shape of reshaped samples_flat: {samples_flat.shape}")

    selected_real_series = real_data_flat[real_series_idx_to_plot]

    knn = NearestNeighbors(n_neighbors=n_neighbors_to_find, metric=metric)
    knn.fit(samples_flat)

    distances, indices = knn.kneighbors(selected_real_series.reshape(1, -1))

    print(f"\nSelected real series index: {real_series_idx_to_plot}")
    print(f"Indices of the {n_neighbors_to_find} most similar generated series: {indices[0]}")
    print(f"Distances to these series: {distances[0]}")

    top_n_generated_series = samples_flat[indices[0]]

    plt.figure(figsize=(15, 7))
    sns.set_theme(style="whitegrid")

    time_axis = np.arange(seq_len)

    plt.plot(time_axis, selected_real_series,
             label=f'Real Series (Index {real_series_idx_to_plot})',
             color='blue', linewidth=2.5, zorder=n_neighbors_to_find + 1)

    try:
        generated_colors = sns.color_palette(color_palette_name, n_neighbors_to_find)
    except Exception as e:
        print(f"Warning: Could not use palette '{color_palette_name}'. Defaulting to 'tab10'. Error: {e}")
        generated_colors = sns.color_palette("tab10", n_neighbors_to_find)


    for i, series_idx_in_samples in enumerate(indices[0]):
        plt.plot(time_axis, top_n_generated_series[i],
                 label=f'Generated Neighbor {i+1} (Sample Index {series_idx_in_samples})',
                 linewidth=1.5,
                 alpha=0.7,  
                 color=generated_colors[i % len(generated_colors)])

    plt.title(f'Real Series vs. Top {n_neighbors_to_find} Similar Generated Series (KNN)')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend(loc='upper right')
    plt.tight_layout()

    plt.savefig(plot_filename)
    print(f"\nPlot saved as '{plot_filename}'")

    plt.show()

    return distances, indices
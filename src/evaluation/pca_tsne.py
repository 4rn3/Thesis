import os
import random 

import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

def visual_evaluation(real_data, generated_data, filename, cond, train_test, n_components=2, alpha=0.7):

    plot_dir = f'./logging/plots/viz/{filename}'
    
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    f_name = f'./logging/plots/viz//{filename}/without_conditioning_{train_test}.png'
    
    if cond:
        f_name = f'./logging/plots/viz//{filename}/with_conditioning_{train_test}.png'
    
    real_reshaped = real_data.reshape(real_data.shape[0], -1)
    generated_reshaped = generated_data.reshape(generated_data.shape[0], -1)
    
    #generated_scaled = normalize_data(generated_reshaped)
    
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(3, 2, figure=fig)
    
    ax1 = fig.add_subplot(gs[0, 0])
    
    pca = PCA(n_components=n_components)
    real_pca = pca.fit_transform(real_reshaped)
    generated_pca = pca.transform(generated_reshaped)
    
    ax1.scatter(real_pca[:, 0], 
                real_pca[:, 1] if n_components > 1 else np.zeros_like(real_pca[:, 0]), 
                label='Real Data', alpha=alpha)
    ax1.scatter(generated_pca[:, 0], 
                generated_pca[:, 1] if n_components > 1 else np.zeros_like(generated_pca[:, 0]), 
                label='Generated Data', alpha=alpha)
    
    ax1.set_title('PCA Comparison')
    ax1.set_xlabel(f'First PC')
    ax1.set_ylabel(f'Second PC')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    ax2 = fig.add_subplot(gs[0, 1])
    
    combined_data = np.vstack([real_reshaped, generated_reshaped])
    
    tsne = TSNE(n_components=2, random_state=42)
    combined_tsne = tsne.fit_transform(combined_data)
    
    real_tsne = combined_tsne[:real_reshaped.shape[0]]
    generated_tsne = combined_tsne[real_reshaped.shape[0]:]
    
    ax2.scatter(real_tsne[:, 0], real_tsne[:, 1], 
                label='Real Data', alpha=alpha)
    ax2.scatter(generated_tsne[:, 0], generated_tsne[:, 1], 
                label='Generated Data', alpha=alpha)
    
    ax2.set_title('t-SNE Comparison')
    ax2.set_xlabel('t-SNE Dimension 1')
    ax2.set_ylabel('t-SNE Dimension 2')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    ax3 = fig.add_subplot(gs[1, :])
    
    umap_reducer = umap.UMAP(n_components=2, random_state=42)
    combined_umap = umap_reducer.fit_transform(combined_data)
    
    real_umap = combined_umap[:real_reshaped.shape[0]]
    generated_umap = combined_umap[real_reshaped.shape[0]:]
    
    ax3.scatter(real_umap[:, 0], real_umap[:, 1], 
                label='Real Data', alpha=alpha)
    ax3.scatter(generated_umap[:, 0], generated_umap[:, 1], 
                label='Generated Data', alpha=alpha)
    
    ax3.set_title('UMAP Comparison')
    ax3.set_xlabel('UMAP Dimension 1')
    ax3.set_ylabel('UMAP Dimension 2')
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    num_columns = real_data.shape[1]
    selected_column = random.randint(0, num_columns - 1)
    
    num_samples_real = real_data.shape[0]
    num_samples_generated = generated_data.shape[0]
    
    real_sample_index = random.randint(0, num_samples_real - 1)
    generated_sample_index = random.randint(0, num_samples_generated - 1)
    
    ax4 = fig.add_subplot(gs[2, :])
    
    real_sample = real_data[generated_sample_index, selected_column, :]
    generated_sample = generated_data[generated_sample_index, selected_column, :]
    
    time_range = np.arange(real_sample.shape[0])
    
    ax4.plot(time_range, real_sample, label=f'Real Data Sample {real_sample_index}')
    ax4.plot(time_range, generated_sample, label=f'Generated Data Sample {generated_sample_index}')
    
    ax4.set_title(f'Sample vs Real data line plot {selected_column}')
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Value')
    ax4.legend()
    ax4.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig(f_name)
    plt.tight_layout()
    plt.show()


def visual_evaluation_unet(ori_data, sample, filename, train_test="Train", cond=False):
    
    plot_dir = f'./logging/plots/viz/{filename}'
    
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    f_name = f'./logging/plots/viz/{filename}/without_conditioning_{train_test}.png'
    
    if cond:
        f_name = f'./logging/plots/viz/{filename}/with_conditioning_{train_test}.png'
    
    fake_data = np.asarray(sample)
    real_data = np.asarray(ori_data)
    
    customer = np.random.choice(real_data.shape[1], 1)[0]

    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(3, 2)
    
    ax_pca = fig.add_subplot(gs[0, 0])
    ax_tsne = fig.add_subplot(gs[0, 1])
    
    ax_umap = fig.add_subplot(gs[1, :])
    
    ax_orig = fig.add_subplot(gs[2, 0])
    ax_synth = fig.add_subplot(gs[2, 1])

    pca = PCA(n_components=2)
    pca.fit(real_data)
    pca_real = (pd.DataFrame(pca.transform(real_data))
                .assign(Data='Real'))
    pca_synthetic = (pd.DataFrame(pca.transform(fake_data))
                       .assign(Data='Synthetic'))
    pca_result = pd.concat([pca_real, pca_synthetic]).rename(
        columns={0: '1st Component', 1: '2nd Component'})

    sb.scatterplot(x='1st Component', y='2nd Component', data=pca_result,
                    hue='Data', style='Data', ax=ax_pca)
    sb.despine()
    ax_pca.set_title('PCA Result')

    tsne_data = np.concatenate((real_data, fake_data), axis=0)
        
    tsne = TSNE(n_components=2, verbose=0, perplexity=15)
    tsne_result = tsne.fit_transform(tsne_data)
    tsne_result = pd.DataFrame(tsne_result, columns=['X', 'Y']).assign(Data='Real')
    tsne_result.loc[len(real_data):, 'Data'] = 'Synthetic'

    sb.scatterplot(x='X', y='Y', data=tsne_result, hue='Data', style='Data', ax=ax_tsne)
    sb.despine()
    ax_tsne.set_title('t-SNE Result')
    
    umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    umap_data = np.concatenate((real_data, fake_data), axis=0)
    umap_result = umap_reducer.fit_transform(umap_data)
    umap_result_df = pd.DataFrame(umap_result, columns=['UMAP1', 'UMAP2']).assign(Data='Real')
    umap_result_df.loc[len(real_data):, 'Data'] = 'Synthetic'
    
    sb.scatterplot(x='UMAP1', y='UMAP2', data=umap_result_df, hue='Data', style='Data', ax=ax_umap)
    sb.despine()
    ax_umap.set_title('UMAP Result', fontsize=14)

    ax_orig.plot(real_data[:, customer].squeeze(), label='Original')
    ax_orig.set_title('Original Data')
        
    ax_synth.plot(fake_data[:, customer].squeeze(), label='Synthetic')
    ax_synth.set_title('Synthetic Data')
    
    fig.suptitle(f'Comparison of Real ({train_test}) and Synthetic Data Distributions', 
                 fontsize=14)
    fig.tight_layout()
    fig.subplots_adjust(top=.92)
    plt.savefig(f_name)
    plt.show()
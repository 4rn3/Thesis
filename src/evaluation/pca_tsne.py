import os
import random 

import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def visualize_pca_tsne(ori_data, fake_data, seq_len, filename, cond, train_test, feature):
    
    plot_dir = f'./logging/plots/{feature}/'
    
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    f_name = f'./logging/plots/{feature}/{filename}_pca_tsne_without_conditioning_{train_test}_{feature}.png'
    
    if cond:
        f_name = f'./logging/plots/{feature}/{filename}_pca_tsne_with_conditioning_{train_test}_{feature}.png'
    
    ori_data = np.asarray(ori_data)
    fake_data = np.asarray(fake_data)
    
    ori_data = ori_data[:fake_data.shape[0]]
    
    sample_size = 32
    idx = np.random.permutation(len(ori_data))[:sample_size]
    randn_num = np.random.permutation(sample_size)[:1]
    
    real_sample = ori_data[idx]
    fake_sample = fake_data[idx]
    
    real_sample_2d = real_sample.reshape(-1, seq_len)
    fake_sample_2d = fake_sample.reshape(-1, seq_len)
    
    ### PCA ###
    pca = PCA(n_components=2)
    pca.fit(real_sample_2d)
    pca_real = (pd.DataFrame(pca.transform(real_sample_2d))
                .assign(Data='Real'))
    pca_synthetic = (pd.DataFrame(pca.transform(fake_sample_2d))
                     .assign(Data='Synthetic'))
    pca_result = pd.concat([pca_real, pca_synthetic]).rename(
        columns={0: '1st Component', 1: '2nd Component'})
    
    ### TSNE ###
    tsne_data = np.concatenate((real_sample_2d, fake_sample_2d), axis=0)
    
    tsne = TSNE(n_components=2, verbose=0, perplexity=15)
    tsne_result = tsne.fit_transform(tsne_data)
    tsne_result = pd.DataFrame(tsne_result, columns=['X', 'Y']).assign(Data='Real')
    tsne_result.loc[len(real_sample_2d):, 'Data'] = 'Synthetic'
    
    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(12, 10))
    sb.scatterplot(x='1st Component', y='2nd Component', data=pca_result,
                   hue='Data', style='Data', ax=axs[0,0])
    sb.despine()
    axs[0,0].set_title('PCA Result')
    
    sb.scatterplot(x='X', y='Y', data=tsne_result, hue='Data', style='Data', ax=axs[0,1])
    sb.despine()
    axs[0,1].set_title('t-SNE Result')
    
    axs[1,0].plot(real_sample[randn_num[0], :, :].squeeze(), label='Original', color='blue')
    axs[1,0].set_title('Original Data')
    
    axs[1,1].plot(fake_sample[randn_num[0], :, :].squeeze(), label='Synthetic', color='red')
    axs[1,1].set_title('Synthetic Data')
    
    fig.suptitle(f'Qualitative Comparison of Real ({train_test}) and Synthetic Data Distributions: {feature}', 
                 fontsize=14)
    fig.tight_layout()
    fig.subplots_adjust(top=.88)
    plt.savefig(f_name)
    plt.show()
    
    
def visualize_all_customers(ori_data, sample, seq_len, filename, cond, train_test, batch_size):
    plot_dir = f'./logging/plots/viz/'
    
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    f_name = f'./logging/plots/viz/{filename}_pca_tsne_without_conditioning_{train_test}_{seq_len}.png'
    
    if cond:
        f_name = f'./logging/plots/viz/{filename}_pca_tsne_with_conditioning_{train_test}_{seq_len}.png'
    
    fake_data = np.asarray(sample)
    real_data = np.asarray(ori_data)

    sample_size = batch_size
    idx = np.random.permutation(len(real_data))[:sample_size]
    customer = np.random.choice(real_data.shape[1], 1)[0]
    randn_num = np.random.permutation(sample_size)[:1]

    fake_data_2d = fake_data[idx[0], :, :]
    real_data_2d = real_data[idx[0], :, :]

    pca = PCA(n_components=2)
    pca.fit(real_data_2d)
    pca_real = (pd.DataFrame(pca.transform(real_data_2d))
                .assign(Data='Real'))
    pca_synthetic = (pd.DataFrame(pca.transform(fake_data_2d))
                        .assign(Data='Synthetic'))
    pca_result = pd.concat([pca_real, pca_synthetic]).rename(
        columns={0: '1st Component', 1: '2nd Component'})

    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(12, 10))
    sb.scatterplot(x='1st Component', y='2nd Component', data=pca_result,
                    hue='Data', style='Data', ax=axs[0,0])
    sb.despine()
    axs[0,0].set_title('PCA Result')

    tsne_data = np.concatenate((real_data_2d, fake_data_2d), axis=0)
        
    tsne = TSNE(n_components=2, verbose=0, perplexity=15)
    tsne_result = tsne.fit_transform(tsne_data)
    tsne_result = pd.DataFrame(tsne_result, columns=['X', 'Y']).assign(Data='Real')
    tsne_result.loc[len(real_data_2d):, 'Data'] = 'Synthetic'

    sb.scatterplot(x='X', y='Y', data=tsne_result, hue='Data', style='Data', ax=axs[0,1])
    sb.despine()
    axs[0,1].set_title('t-SNE Result')

    axs[1,0].plot(real_data[randn_num[0], customer, :].squeeze(), label='Original', color='blue')
    axs[1,0].set_title('Original Data')
        
    axs[1,1].plot(fake_data[randn_num[0], customer, :].squeeze(), label='Synthetic', color='red')
    axs[1,1].set_title('Synthetic Data')
    
    fig.suptitle(f'Qualitative Comparison of Real ({train_test}) and Synthetic Data Distributions', 
                 fontsize=14)
    plt.savefig(f_name)
    fig.tight_layout()
    fig.subplots_adjust(top=.88)
    plt.show()
    
    
    
def visualize_all_customers_unet(ori_data, sample, filename, train_test = "Train", cond=False):
        
    f_name = f'./logging/plots/{filename}_pca_tsne_without_conditioning_{train_test}.png'
    
    if cond:
        f_name = f'./logging/plots/{filename}_pca_tsne_with_conditioning_{train_test}.png'
    
    
    fake_data = np.asarray(sample)
    real_data = np.asarray(ori_data)

    customer = np.random.choice(real_data.shape[1], 1)[0]

    pca = PCA(n_components=2)
    pca.fit(real_data)
    pca_real = (pd.DataFrame(pca.transform(real_data))
                    .assign(Data='Real'))
    pca_synthetic = (pd.DataFrame(pca.transform(fake_data))
                        .assign(Data='Synthetic'))
    pca_result = pd.concat([pca_real, pca_synthetic]).rename(
            columns={0: '1st Component', 1: '2nd Component'})

    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(12, 10))
    sb.scatterplot(x='1st Component', y='2nd Component', data=pca_result,
                    hue='Data', style='Data', ax=axs[0,0])
    sb.despine()
    axs[0,0].set_title('PCA Result')

    tsne_data = np.concatenate((real_data, fake_data), axis=0)
        
    tsne = TSNE(n_components=2, verbose=0, perplexity=15)
    tsne_result = tsne.fit_transform(tsne_data)
    tsne_result = pd.DataFrame(tsne_result, columns=['X', 'Y']).assign(Data='Real')
    tsne_result.loc[len(real_data):, 'Data'] = 'Synthetic'

    sb.scatterplot(x='X', y='Y', data=tsne_result, hue='Data', style='Data', ax=axs[0,1])
    sb.despine()
    axs[0,1].set_title('t-SNE Result')

    axs[1,0].plot(real_data[:, customer].squeeze(), label='Original', color='blue')
    axs[1,0].set_title('Original Data')
        
    axs[1,1].plot(fake_data[:, customer].squeeze(), label='Synthetic', color='red')
    axs[1,1].set_title('Synthetic Data')
    
    fig.tight_layout()
    plt.savefig(f_name)
    plt.show()
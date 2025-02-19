import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def visualize_pca_tsne(ori_data, fake_data, seq_len, filename, cond, train_test):
    
    f_name = f'./logging/plots/{filename}_pca_tsne_without_conditioning_{train_test}.png'
    
    if cond:
        f_name = f'./logging/plots/{filename}_pca_tsne_with_conditioning_{train_test}.png'
    
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
    
    tsne = TSNE(n_components=2, verbose=0, perplexity=40)
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
    
    fig.suptitle(f'Assessing Diversity: Qualitative Comparison of Real ({train_test}) and Synthetic Data Distributions', 
                 fontsize=14)
    fig.tight_layout()
    fig.subplots_adjust(top=.88)
    plt.savefig(f_name)
    plt.show()
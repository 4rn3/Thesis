import os

import numpy as np

import matplotlib.pyplot as plt

from scipy.spatial.distance import jensenshannon

def compute_jsd(real_data, generated_data, num_bins):
    real_data_flat = real_data.flatten()
    generated_data_flat = generated_data.flatten()
    
    real_hist, real_bins = np.histogram(real_data_flat, bins=num_bins, density=True)
    generated_hist, generated_bins = np.histogram(generated_data_flat, bins=num_bins, density=True)
    
    real_prob = real_hist / np.sum(real_hist)
    generated_prob = generated_hist / np.sum(generated_hist)
    
    jsd = jensenshannon(real_prob, generated_prob, base=2)
    return jsd

def plot_jsd(real_data_train, real_data_test, samples, filename, cond=False, num_bins=10):
    
    plot_dir = f'./logging/plots/JSD/{filename}'
    
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        
    f_name = f'./logging/plots/JSD/{filename}/without_conditioning.png'
    
    if cond:
        f_name = f'./logging/plots/JSD/{filename}/with_conditioning.png'
    
    train_jsd_no_con = []
    test_jsd_no_con = []
    
    for idx in range(0, real_data_train.shape[1]-1):
        customer_train = real_data_train[:, idx, :].squeeze()
        customer_test = real_data_test[:, idx, :].squeeze()

        customer_sample = samples[:, :, idx].squeeze()
            
        jsd_train = compute_jsd(customer_train, customer_sample, 100)
        jsd_test = compute_jsd(customer_test, customer_sample, 100)
            
        train_jsd_no_con.append(jsd_train)
        test_jsd_no_con.append(jsd_test)
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
    ax1.hist(train_jsd_no_con, bins=num_bins, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_title('Training JSD Counts')
    ax1.set_xlabel('Jensen-Shannon Divergence')
    ax1.set_ylabel('Frequency')
    ax1.grid(alpha=0.3)
        
    ax2.hist(test_jsd_no_con, bins=num_bins, alpha=0.7, color='blue', edgecolor='black')
    ax2.set_title('Test JSD Counts')
    ax2.set_xlabel('Jensen-Shannon Divergence')
    ax2.set_ylabel('Frequency')
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.suptitle('Jensen-Shannon Divergence Distributions', y=1.05)
    plt.savefig(f_name)
    plt.show()
import numpy as np
from scipy.spatial.distance import jensenshannon

def compute_jsd(real_data, generated_data, num_bins=100):
    real_data_flat = real_data.flatten()
    generated_data_flat = generated_data.flatten()
    
    real_hist, real_bins = np.histogram(real_data_flat, bins=num_bins, density=True)
    generated_hist, generated_bins = np.histogram(generated_data_flat, bins=num_bins, density=True)
    
    real_prob = real_hist / np.sum(real_hist)
    generated_prob = generated_hist / np.sum(generated_hist)
    
    jsd = jensenshannon(real_prob, generated_prob, base=2)
    return jsd
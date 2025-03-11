import os

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, TensorDataset

from preprocessing.write_to_disk import write_meter_data, write_cond_data

def serve_data(batch_size = 10, num_customers = 4096, overwrite=False):
    DATA_DIR = "./preprocessing/data/customer_led_network_revolution/preprocessed/"
            
    if not os.path.isfile(os.path.join(DATA_DIR, "img_train_data.npy")) or overwrite:
        print("writing data to disk")
        indices = write_meter_data(num_customers=num_customers, img=True)
        write_cond_data(indices)
        
        
    print("loading meter data")
    train_data= np.load(os.path.join(DATA_DIR, "img_train_data.npy"))
    test_data = np.load(os.path.join(DATA_DIR, "img_test_data.npy"))
    print(f"train_data shape: {train_data.shape}")
    print(f"test_data shape: {test_data.shape}")
    print("loading condition data")
    train_cond_data = np.load(os.path.join(DATA_DIR, "cond_train_data.npy"))
    test_cond_data = np.load(os.path.join(DATA_DIR, "cond_test_data.npy"))
    print(f"train_data shape: {train_cond_data.shape}")
    print(f"test_data shape: {test_cond_data.shape}")
    train_dataset = TensorDataset(torch.from_numpy(train_data), torch.from_numpy(train_cond_data))
    train_loader = DataLoader(train_dataset, batch_size)
    
    test_dataset = TensorDataset(torch.from_numpy(test_data), torch.from_numpy(test_cond_data))
    test_loader = DataLoader(test_dataset, batch_size)
    
    n_cfeat = train_cond_data.shape[1]
    print("loading customer ids")
    customer_ids = np.load(os.path.join(DATA_DIR,"customer_ids.npy")).tolist()
    
    # # Add these to your serve_data function at the end
    # print(f"Train data stats - Min: {train_data.min()}, Max: {train_data.max()}, Mean: {train_data.mean()}")
    # print(f"Test data stats - Min: {test_data.min()}, Max: {test_data.max()}, Mean: {test_data.mean()}")
    # print(f"Train cond stats - Min: {train_cond_data.min()}, Max: {train_cond_data.max()}, Mean: {train_cond_data.mean()}")

    # # Check for NaN values
    # print(f"NaN in train data: {np.isnan(train_data).sum()}")
    # print(f"NaN in test data: {np.isnan(test_data).sum()}")
    # print(f"NaN in train cond: {np.isnan(train_cond_data).sum()}")

    
    return train_loader, test_loader, n_cfeat, customer_ids, train_data, test_data
        
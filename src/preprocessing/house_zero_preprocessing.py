import numpy as np
import pandas as pd
from pyts.image import GramianAngularField

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split


def load_data(feature_subset):
    data_dir_y1 = './preprocessing/data/meter_data/48190963_Loads_hourly.csv'
    data_dir_y2 = './preprocessing/data/meter_data/48190948_Loads_hourly.csv'

    cond_data_dir_y1 = "./preprocessing/data/conditioning_data/48190936_Weather.csv"
    cond_data_dir_y2 = "./preprocessing/data/conditioning_data/48190930_Weather.csv"

    data_y1 = pd.read_csv(data_dir_y1)
    data_y2 = pd.read_csv(data_dir_y2)

    cond_data_y1 = pd.read_csv(cond_data_dir_y1, encoding='unicode_escape')
    cond_data_y2 = pd.read_csv(cond_data_dir_y2, encoding='unicode_escape')

    meter_data = pd.concat([data_y1.reset_index(drop=True), data_y2.reset_index(drop=True)], axis=0)
    cond_data = pd.concat([cond_data_y1.reset_index(drop=True), cond_data_y2.reset_index(drop=True)], axis=0)
    
    meter_data = meter_data[feature_subset]
    cond_data = cond_data.iloc[:, 1:]
        
    return meter_data, cond_data

def ts_to_gaf(meter_data):
    multi_var_ts = np.asarray(meter_data)
    gaf = GramianAngularField(image_size=multi_var_ts.shape[1], method='summation')
    gaf_images = [gaf.fit_transform(ts.reshape(1, -1))[0] for ts in multi_var_ts]
    ts_images = np.asarray(gaf_images)
    return ts_images

def preprocess(meter_data, cond_data, channels):
    X = np.tile(meter_data, (1, 1, 1, channels))
    X = np.transpose(X, (1, 0, 2, 3))
    
    y = np.asarray(cond_data)
    return X, y

def serve_data_hz(batch_size, channels, feature_subset):
    meter_data, cond_data = load_data(feature_subset)
    meter_data_img = ts_to_gaf(meter_data)
    
    X, y = preprocess(meter_data_img, cond_data, channels)
    
    n_cfeat = int(y.shape[1])
    
    tts_split = 0.8
    train_size = int(len(meter_data_img) * tts_split)
    test_size = len(meter_data_img) - train_size
    X_train, X_test = random_split(X, [train_size, test_size])
    X_train, X_test = np.asarray(X_train), np.asarray(X_test)
    
    train_size = int(len(cond_data) * tts_split)
    test_size = len(cond_data) - train_size
    y_train, y_test = random_split(y, [train_size, test_size])
    y_train, y_test = np.asarray(y_train), np.asarray(y_test)
    
    train_x_tensor = torch.from_numpy(X_train)
    train_y_tensor = torch.from_numpy(y_train)
    train_dataset = TensorDataset(train_x_tensor, train_y_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=5)
    
    test_x_tensor = torch.from_numpy(X_test)
    test_y_tensor = torch.from_numpy(y_test)
    test_dataset = TensorDataset(test_x_tensor, test_y_tensor)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=5)
    
    return train_dataloader, test_dataloader, n_cfeat
    
    



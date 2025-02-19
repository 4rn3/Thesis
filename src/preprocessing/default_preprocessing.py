import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split

def normalize(data):
    
    min_val = np.min(np.min(data, axis=0), axis=0)
    data = data - min_val

    max_val = np.max(np.max(data, axis=0), axis=0)
    data = data / (max_val + 1e-7)
    
    data = data.astype(np.float32)
    
    return data

class MakeDATA(Dataset):
    def __init__(self, data, seq_len):
        data = np.asarray(data, dtype=np.float32)
        norm_data = normalize(data)
        seq_data = []
        for i in range(len(norm_data) - seq_len + 1):
            x = norm_data[i : i + seq_len]
            seq_data.append(x)
        self.samples = []
        idx = torch.randperm(len(seq_data))
        for i in range(len(seq_data)):
            self.samples.append(seq_data[idx[i]])
        self.samples = np.asarray(self.samples, dtype=np.float32)
            
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def data_preprocess(subset_len=25000):
    
    data_dir = f'./preprocessing/data/meter_data/LCL-June2015v2_0.csv'
    cond_data_dir = "./preprocessing/data/conditioning_data/weather_hourly_darksky.csv"
    
    data = pd.read_csv(data_dir)
    data.replace('Null', 0, inplace=True)
    
    cond_data = pd.read_csv(cond_data_dir)
    cond_data = cond_data[["time", "humidity", "temperature", "windSpeed"]]
    cond_data['time'] = pd.to_datetime(cond_data['time'])
    cond_data.set_index('time', inplace=True)
    cond_data = cond_data.resample('30min').interpolate(method='linear')
    
    if subset_len != "None":
       data = data.iloc[:subset_len, 3]
       cond_data = cond_data.iloc[:subset_len, :]
    else:    
        data = data.iloc[:len(cond_data), 3] #for this test dataset cond data is less than actual data
    
    return data, cond_data

def LoadData(seq_len, subset_len):
    tts_split = 0.8
    data, cond_data = data_preprocess(subset_len) 
    data = MakeDATA(data, seq_len)
    cond_data = MakeDATA(cond_data, seq_len)
    
    train_size = int(len(data) * tts_split)
    test_size = len(data) - train_size
    train_data, test_data = random_split(data, [train_size, test_size])
    
    train_size = int(len(cond_data) * tts_split)
    test_size = len(cond_data) - train_size
    cond_data_train, cond_data_test = random_split(cond_data, [train_size, test_size])
    
    return train_data, test_data, cond_data_train, cond_data_test

def serve_data(seq_len, batch_size, subset_len):
    train_data, test_data, cond_data_train, cond_data_test = LoadData(seq_len=seq_len, subset_len=subset_len)
    train_data, test_data, cond_data_train, cond_data_test= np.asarray(train_data), np.asarray(test_data), np.asarray(cond_data_train), np.asarray(cond_data_test)
    
    if len(train_data.shape) < 3:
        train_data = np.expand_dims(train_data, axis=-1)
        test_data = np.expand_dims(test_data, axis=-1)
    
    features = train_data.shape[2]
    cond_features = cond_data_train.shape[2]
    
    print(f"num of channels in transformer: {features} \nnum of cond feature: {cond_features}")

    train_data, test_data, cond_data_train, cond_data_test = train_data.transpose(0,2,1), test_data.transpose(0,2,1), cond_data_train.transpose(0,2,1), cond_data_test.transpose(0,2,1)
    print(f"Train shape (batch, features, seq_len): {train_data.shape}")
    print(f"Cond shape (batch, features, seq_len): {cond_data_train.shape}")
    
    train_dataset = TensorDataset(torch.from_numpy(train_data), torch.from_numpy(cond_data_train))
    train_loader = DataLoader(train_dataset, batch_size)

    test_dataset = TensorDataset(torch.from_numpy(test_data), torch.from_numpy(cond_data_test))
    test_loader = DataLoader(test_dataset, batch_size)

    real_data, real_cond_data = next(iter(train_loader))
    print(f"batched data shape: {real_data.shape}")
    
    return train_loader, test_loader, features, cond_features

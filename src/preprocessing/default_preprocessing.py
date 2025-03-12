import os
import pandas as pd
import numpy as np

from datetime import datetime
import math

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split

DATA_DIR = "./preprocessing/data/customer_led_network_revolution/"

def normalize(data):
    
    min_val = np.min(np.min(data, axis=0), axis=0)
    data = data - min_val

    max_val = np.max(np.max(data, axis=0), axis=0)
    data = data / (max_val + 1e-7)
    
    data = data.astype(np.float32)
    
    return data

def calc_cutoff(cutoff, start_date = datetime(2011, 5, 1), end_date = datetime(2013, 10, 1), frequency=2):

    delta = end_date - start_date
    delta_days = delta.days

    delta_hourly = delta_days * 24
    delta_df_frequency = delta_hourly * frequency
    
    return math.floor(delta_df_frequency * cutoff)

def impute_ts(og_df):
    df = og_df.copy()
    for col in df.columns:
        df[col] = df[col].interpolate(method="time")
    
    return df

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

def load_weather_data(data_dir, indices):
    
    weather_data = pd.read_csv(os.path.join(data_dir, "london_weather.csv"))
    weather_data['date'] = pd.to_datetime(weather_data['date'], format='%Y%m%d')
    weather_data.set_index('date', inplace=True)

    weather_data_resampled = weather_data.resample('30min').interpolate(method='linear')
    weather_data_resampled = weather_data_resampled[weather_data_resampled.index.isin(indices)]
    return weather_data_resampled  

def data_preprocessing_revolution(percent_cutoff, num_customers):
    data_dir = "./preprocessing/data/customer_led_network_revolution/"
    cut_off = calc_cutoff(percent_cutoff)
    
    domestic_smart_meter_data = pd.read_csv(os.path.join(data_dir, "TC1a/TrialMonitoringDataHH.csv"), index_col=1, usecols=["Location ID", "Date and Time of capture", "Parameter"])
    
    vc = domestic_smart_meter_data["Location ID"].value_counts()
    vc = vc[vc >= cut_off]
    customer_ids = vc.sample(num_customers).keys().tolist()
    
    filtered_smart_meter_data = domestic_smart_meter_data[domestic_smart_meter_data["Location ID"].isin(customer_ids)]
    
    del domestic_smart_meter_data
    
    filtered_smart_meter_data.index = pd.to_datetime(filtered_smart_meter_data.index, format='%d/%m/%Y %H:%M:%S')
    multi_var_data = filtered_smart_meter_data.pivot_table(index=filtered_smart_meter_data.index, columns='Location ID', values='Parameter')
    
    weather_data = load_weather_data(data_dir, multi_var_data.index.tolist())

    return multi_var_data, weather_data, customer_ids

def split_tts(data):
    tts_split = 0.8
    train_size = int(len(data) * tts_split)
    
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    return train_data, test_data
    

def LoadData(seq_len, percent_cutoff, num_customers):
    data, cond_data, customer_ids = data_preprocessing_revolution(percent_cutoff=percent_cutoff, num_customers=num_customers)
    
    train_data, test_data = split_tts(data)
    cond_train_data, cond_test_data = split_tts(cond_data)
    
    imputed_cond_train_data = impute_ts(cond_train_data)
    imputed_cond_test_data = impute_ts(cond_test_data)

    imputed_train_data = impute_ts(train_data)
    imputed_test_data = impute_ts(test_data)
    
    train_data = MakeDATA(imputed_train_data, seq_len)
    train_cond_data = MakeDATA(imputed_cond_train_data, seq_len)
    
    test_data = MakeDATA(imputed_test_data, seq_len)
    test_cond_data = MakeDATA(imputed_cond_test_data, seq_len)
    
    return train_data, test_data, train_cond_data, test_cond_data, customer_ids

def write_to_disk(seq_len, batch_size, percent_cutoff, num_customers):
    
    train_data, test_data, cond_data_train, cond_data_test, customer_ids = LoadData(seq_len=seq_len, percent_cutoff=percent_cutoff, num_customers=num_customers)
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
    
    print("writing customer ids to disk")
    np.save(os.path.join(DATA_DIR, "preprocessed/customer_ids.npy"), np.asarray(customer_ids))
    print("writing preprocessed meter data to disk")
    np.save(os.path.join(DATA_DIR, "preprocessed/train_data.npy"), np.asarray(train_data))
    np.save(os.path.join(DATA_DIR, "preprocessed/test_data.npy"), np.asarray(test_data))
    print("writing preprocessed condition data to disk")
    np.save(os.path.join(DATA_DIR, "preprocessed/cond_train_data_seq.npy"), np.asarray(cond_data_train))
    np.save(os.path.join(DATA_DIR, "preprocessed/cond_test_data_seq.npy"), np.asarray(cond_data_test))
    
    return train_loader, test_loader, features, cond_features, customer_ids

def serve_data(seq_len=15, batch_size=32, percent_cutoff=0.95, num_customers=4096, overwrite=False):
    
    if not os.path.isfile(os.path.join(DATA_DIR, "preprocessed/train_data.npy")) or overwrite:
        print("writing data to disk")
        write_to_disk(seq_len=seq_len, batch_size=batch_size, percent_cutoff=percent_cutoff, num_customers=num_customers)
        
    print("loading meter data")
    train_data= np.load(os.path.join(DATA_DIR, "preprocessed/train_data.npy"))
    test_data = np.load(os.path.join(DATA_DIR, "preprocessed/test_data.npy"))
    print(f"train_data shape: {train_data.shape}")
    print(f"test_data shape: {test_data.shape}")
    print("loading condition data")
    train_cond_data = np.load(os.path.join(DATA_DIR, "preprocessed/cond_train_data_seq.npy"))
    test_cond_data = np.load(os.path.join(DATA_DIR, "preprocessed/cond_test_data_seq.npy"))
    print(f"cond_train_data shape: {train_cond_data.shape}")
    print(f"cond_test_data shape: {test_cond_data.shape}")
    train_dataset = TensorDataset(torch.from_numpy(train_data), torch.from_numpy(train_cond_data))
    train_loader = DataLoader(train_dataset, batch_size)
    
    test_dataset = TensorDataset(torch.from_numpy(test_data), torch.from_numpy(test_cond_data))
    test_loader = DataLoader(test_dataset, batch_size)
    
    features = train_data.shape[1]
    n_cfeat = train_cond_data.shape[1]
    print(f"features: {features} \nconditioning features: {n_cfeat}")
    print("loading customer ids")
    customer_ids = np.load(os.path.join(DATA_DIR,"preprocessed/customer_ids.npy")).tolist()
    print(f"Number of customers: {len(customer_ids)}")
    
    return train_loader, test_loader, features, n_cfeat, customer_ids, train_data, test_data
    
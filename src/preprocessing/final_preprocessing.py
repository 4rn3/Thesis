import warnings
warnings.filterwarnings("ignore", message="Could not find the number of physical cores")

import os
import math

import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split as tts

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

PREPROCESSED_DIR = "./preprocessing/data/customer_led_network_revolution/preprocessed/"
METER_DATA_DIR = "./preprocessing/data/customer_led_network_revolution/"
PREPROCESSING_DIR = "./preprocessing/data/customer_led_network_revolution/preprocessing/"
COND_DATA_DIR =  "./preprocessing/data/customer_led_network_revolution/cond_data/"

def load_dataframes_from_npy(directory, types):
    files = os.listdir(directory)
    
    data_files = {f.replace("_cols.npy", ""): f for f in files if "cols" in f}
    dfs = []
    
    for name, col_file in data_files.items():
        data_file = name + ".npy"

        if data_file in files and data_file[:2] in types:
            cols = np.load(os.path.join(directory, col_file))
            data = np.load(os.path.join(directory, data_file))

            dfs.append(pd.DataFrame(data, columns=cols))

    return dfs

def train_test_split(data, ratio=0.8):
    train_size = int(math.floor(len(data) * ratio))
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data

def remove_outliers_iqr(data):
    df = data.copy()
    Q1 = df[df>0].quantile(0.05)
    Q3 = df[df>0].quantile(0.95)
    IQR = Q3 - Q1
    lower_bound = Q1 - 0.5 * IQR
    upper_bound = Q3 + 1.0 * IQR
    df[(df < lower_bound) | (df > upper_bound)] = np.nan
    return df

def normalize_data(data):
    df = data.copy()

    for col in df.columns:
        min_val = df[col].min()
        max_val = df[col].max()
        range_val = max_val - min_val

        if range_val != 0:
            df[col] = (df[col] - min_val) / range_val
        else:
            df[col] = 0

    return df

def interpolate_data(data):
    
    data = data.fillna(method='ffill').fillna(method='bfill')
    
    if data.isna().sum().sum() > 0:
        raise ValueError("There are still NaN values after interpolation!")

    return data

def preprocess(df):
    no_outlier_df = remove_outliers_iqr(df)
    
    train, test = train_test_split(no_outlier_df)
    
    imputed_train = interpolate_data(train)
    imputed_test = interpolate_data(test)
    
    norm_train = normalize_data(imputed_train)
    norm_test = normalize_data(imputed_test)
    
    return norm_train, norm_test

def filter_data(data, threshold=0.95):
    min_non_na_count = int(data.shape[0] * threshold)
    cleaned_data = data.dropna(thresh=min_non_na_count, axis=1)
    return cleaned_data

def write_combined_to_disk(types):
    concat_dfs = load_dataframes_from_npy(PREPROCESSING_DIR, types)
    least_rows = min([df.shape[0] for df in concat_dfs])
    concat_dfs = [df.reset_index(drop=True) for df in concat_dfs]
    concat_dfs = [df.iloc[:least_rows] for df in concat_dfs]
    meter_df = pd.concat(concat_dfs, ignore_index=True, axis=1)
    
    meter_df = filter_data(meter_df)

    train, test = preprocess(meter_df)
    
    np.save(os.path.join(PREPROCESSED_DIR, "meter_train_df.npy"), np.asarray(train))
    np.save(os.path.join(PREPROCESSED_DIR, "meter_train_cols.npy"), np.asarray(train.columns.tolist()))

    np.save(os.path.join(PREPROCESSED_DIR, "meter_test_df.npy"), np.asarray(test))
    np.save(os.path.join(PREPROCESSED_DIR, "meter_test_cols.npy"), np.asarray(test.columns.tolist()))
    
    
def load_data():
    train_arr = np.load(os.path.join(PREPROCESSED_DIR, "meter_train_df.npy"))
    train_cols = np.load(os.path.join(PREPROCESSED_DIR, "meter_train_cols.npy"))
    
    test_arr = np.load(os.path.join(PREPROCESSED_DIR, "meter_test_df.npy"))
    test_cols = np.load(os.path.join(PREPROCESSED_DIR, "meter_test_cols.npy"))
    
    train_df = pd.DataFrame(train_arr, columns=train_cols)
    test_df = pd.DataFrame(test_arr, columns=test_cols)
    return train_df, test_df

class MakeDATA(Dataset):
    def __init__(self, data, seq_len):
        data = np.asarray(data, dtype=np.float32)
        seq_data = []
        for i in range(len(data) - seq_len + 1):
            x = data[i : i + seq_len]
            seq_data.append(x)
        self.samples = np.asarray(seq_data, dtype=np.float32) 

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def create_timestamp_sequences(timestamps, seq_len):
    timestamps = pd.to_datetime(timestamps)
    
    hours = timestamps.dt.hour.values
    dayofweek = timestamps.dt.dayofweek.values
    is_weekend = (timestamps.dt.dayofweek >= 5).astype(int).values
    
    features = np.stack([hours, dayofweek, is_weekend], axis=1)
    
    sequences = []
    for i in range(len(features) - seq_len + 1):
        seq = features[i:i + seq_len]
        sequences.append(seq)
    
    return np.asarray(sequences)

def create_cond_unet(timestamps):
    timestamps = pd.to_datetime(timestamps)
    
    hours = timestamps.dt.hour.values
    dayofweek = timestamps.dt.dayofweek.values
    is_weekend = (timestamps.dt.dayofweek >= 5).astype(int).values
    features = np.stack([hours, dayofweek, is_weekend], axis=1)
    
    return np.asarray(features)
    
    
def load_and_preprocess_weather():
    df = pd.read_csv(os.path.join(COND_DATA_DIR, "london_weather.csv"))
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df.set_index('date', inplace=True)
    
    train, test = train_test_split(df)
    
    train = train.resample('30min').interpolate(method='time')
    test = test.resample('30min').interpolate(method='time')
    
    imputed_train = interpolate_data(train)
    imputed_test = interpolate_data(test)
    
    norm_train = normalize_data(imputed_train)
    norm_test = normalize_data(imputed_test)
    
    return norm_train, norm_test

def cluster(data, k):
    kmeans = KMeans(n_clusters=k)
    cluster_labels = kmeans.fit_predict(data.T)
    
    clustered_data = []
    for cluster in range(kmeans.n_clusters):
        cluster_data = data.iloc[:, cluster_labels == cluster].mean(axis=1)
        clustered_data.append(cluster_data)
        
    return pd.DataFrame(clustered_data).T


def serve_data(types=["ev","hp","pv","re"], seq_len=336, batch_size=256, overwrite=False, kmeans=True, k=5, cond=False):
    if not os.path.isfile(os.path.join(PREPROCESSED_DIR, "meter_train_df.npy")) or overwrite:
        print("Writing to disk")
        write_combined_to_disk(types)
    
    train_df, test_df = load_data()
    print(f"Init data shape train:{train_df.shape}, test: {test_df.shape}")
    
    if kmeans:
        train_df = cluster(train_df, k)
        test_df = cluster(test_df, k)
        print(f"K-means data shape train:{train_df.shape}, test: {test_df.shape}")
            
    features = train_df.shape[1]
        
    train_cols = train_df.columns.tolist()
    test_cols = test_df.columns.tolist()
    
    idx = np.load(os.path.join(PREPROCESSING_DIR, "all_df_idx.npy"), allow_pickle=True)
    idx_df = pd.DataFrame(idx, columns=["Date"])
    cond_train, cond_test = train_test_split(idx_df)
    cond_train = create_timestamp_sequences(cond_train["Date"], seq_len)
    cond_test = create_timestamp_sequences(cond_test["Date"], seq_len)
    print(f"timestamp sequenced: {cond_train.shape}, {cond_test.shape}")
    
    if cond:
        print("Load weather data")
        weather_train, weather_test = load_and_preprocess_weather()
        #TODO concate cond with weathet
    
    
    train = np.asarray(MakeDATA(train_df, seq_len=seq_len))
    test = np.asarray(MakeDATA(test_df, seq_len=seq_len))
    
    if train.shape[0] > cond_train.shape[0]:
        train = train[cond_train.shape[0], : ,:]
        test = test[cond_test.shape[0], :, :]
    if train.shape[0] < cond_train.shape[0]:
        cond_train = cond_train[train.shape[0], : , :]
        cond_test = cond_test[test.shape[0], : , :]
    
    print(f"Train & Test shape after sequence: {train.shape}, {test.shape}")
  
    cond_features = cond_train.shape[1]
    print(f"Conditioning features: {cond_features}")
        
    train, test = train.transpose(0,2,1), test.transpose(0,2,1)
    print(f"Transposed data shape train:{train.shape}, test: {test.shape}")
    
    cond_train, cond_test = cond_train.transpose(0,2,1), cond_test.transpose(0,2,1)
    print(f"Transpose Cond train: {cond_train.shape}, Transpose Cond test: {cond_test.shape}")
    
    train_dataset = TensorDataset(torch.from_numpy(train), torch.from_numpy(cond_train))
    train_loader = DataLoader(train_dataset, batch_size, shuffle=False)
    
    test_dataset = TensorDataset(torch.from_numpy(test), torch.from_numpy(cond_test))
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    
    return train_loader, test_loader, train_cols, test_cols, test, features, cond_features
    
def serve_data_unet(types=["ev","hp","pv","re"], batch_size=10, overwrite=False, customers=9216, cond=False):
    if not os.path.isfile(os.path.join(PREPROCESSED_DIR, "meter_train_df.npy")) or overwrite:
        print("write to file")
        write_combined_to_disk(types)
    
    train_df, test_df = load_data()
    print(f"Init data shape train:{train_df.shape}, test: {test_df.shape}")
    
    train_df = train_df.iloc[:, :customers]
    test_df = test_df.iloc[:, :customers]

    train_cols = train_df.columns.tolist()
    test_cols = test_df.columns.tolist()
    
    train_arr = np.asarray(train_df)
    test_arr = np.asarray(test_df)
    
    col_row = int(np.sqrt(customers))
    
    img_train = train_arr.reshape(-1, 1, col_row, col_row)
    img_test = test_arr.reshape(-1, 1, col_row, col_row)
    print(f"img data: {img_train.shape}, {img_test.shape}")
    
    idx = np.load(os.path.join(PREPROCESSING_DIR, "all_df_idx.npy"), allow_pickle=True)
    idx_df = pd.DataFrame(idx, columns=["Date"])
    cond_train, cond_test = train_test_split(idx_df)
    cond_train = create_cond_unet(cond_train["Date"])
    cond_test = create_cond_unet(cond_test["Date"])
    print(f"timestamp sequenced: {cond_train.shape}, {cond_test.shape}")
        
    if img_train.shape[0] > cond_train.shape[0]:
        img_train = img_train[cond_train.shape[0], :]
        img_test = img_test[cond_test.shape[0], :]
    if img_train.shape[0] < cond_train.shape[0]:
        cond_train = cond_train[img_train.shape[0], :]
        cond_test = cond_test[img_test.shape[0], :]
        
    print(f"Filtered cond: {cond_train.shape}, {cond_test.shape}")
    
    cond_features = cond_train.shape[1]
    
    train_dataset = TensorDataset(torch.from_numpy(img_train), torch.from_numpy(cond_train))
    train_loader = DataLoader(train_dataset, batch_size, shuffle=False)
    
    test_dataset = TensorDataset(torch.from_numpy(img_test), torch.from_numpy(cond_test))
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    
    return  train_loader, test_loader, cond_features, (train_cols, test_cols), img_train, img_test     

    
    
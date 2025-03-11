import os
from datetime import datetime
import math

import pandas as pd
import numpy as np

import tensorflow as tf

pd.options.mode.chained_assignment = None

DATA_DIR = "./preprocessing/data/customer_led_network_revolution/"

def calc_cutoff(start_date=datetime(2011, 5, 1), end_date=datetime(2013, 10, 1), percent_cutoff=0.95, frequency=1):
    delta = end_date - start_date 
    delta_days = delta.days
    delta_hours = delta_days * 24
    delta_df_frequency = delta_hours * frequency
    
    cut_off = math.floor(delta_df_frequency * percent_cutoff)
    return cut_off

def train_test_split(data, ratio=0.8):
    train_size = int(math.floor(len(data) * ratio))
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data

def interpolate_data(data):
    for col in data.columns:
        data[col] = data[col].interpolate(method="time")
        
    return data

def normalize(data):
    # norm authors
    # train_data = (train_data-train_data.min())/(train_data.max()-train_data.min())
    # train_data = (train_data-0.5)*2
    min_vals = data.min()
    max_vals = data.max()
    
    # Add a small epsilon to prevent division by zero
    range_vals = (max_vals - min_vals)
    range_vals = np.where(range_vals == 0, 1e-7, range_vals)
    
    normalized_data = (data - min_vals) / range_vals
    return normalized_data.astype(np.float32)



def standardize(data):
    mean = np.mean(data, axis=0)
    data = data - mean

    std = np.std(data, axis=0)
    # Prevent division by zero
    std = np.where(std < 1e-7, 1e-7, std)
    data = data / std
    
    # Clip extreme values to prevent gradient explosions
    data = np.clip(data, -5, 5)
    
    return data.astype(np.float32)

def load_weather_data(data_dir):
    weather_data = pd.read_csv(os.path.join(data_dir, "london_weather.csv"))
    weather_data['date'] = pd.to_datetime(weather_data['date'], format='%Y%m%d')
    weather_data.set_index('date', inplace=True)
    return weather_data  

def write_meter_data(num_customers=4096, img=False):
    
    meter_data = pd.read_csv(os.path.join(DATA_DIR, "TC1a/TrialMonitoringDataHH.csv"), index_col=1, usecols=["Location ID", "Date and Time of capture", "Parameter"])
    
    cut_off = calc_cutoff(frequency=2) #half hourly data
    
    vc = meter_data["Location ID"].value_counts()
    vc = vc[vc >= cut_off]
    
    customer_ids = vc.sample(num_customers).keys().tolist()
    print("writing customer ids to disk")
    np.save(os.path.join(DATA_DIR, "preprocessed/customer_ids_img.npy"), customer_ids)
    
    filtered_meter_data = meter_data[meter_data["Location ID"].isin(customer_ids)]
    del meter_data
    
    filtered_meter_data.index = pd.to_datetime(filtered_meter_data.index, format='%d/%m/%Y %H:%M:%S')
    multi_var_data = filtered_meter_data.pivot_table(index=filtered_meter_data.index, columns='Location ID', values='Parameter')
    del filtered_meter_data
    
    indices = multi_var_data.index.tolist()
    
    train_data, test_data = train_test_split(multi_var_data)
    
    print("interpolating data")
    interpolated_train_data = interpolate_data(train_data)
    interpolated_test_data = interpolate_data(test_data)
    
    print("normalizing data")
    normalized_train_data = normalize(interpolated_train_data)
    normalized_test_data = normalize(interpolated_test_data)
    
    print(f"NaN values in training data: {np.isnan(normalized_train_data).sum()}")
    print(f"NaN values in test data: {np.isnan(normalized_test_data).sum()}")
    
    normalized_train_data = np.nan_to_num(normalized_train_data, nan=0.0)
    normalized_test_data = np.nan_to_num(normalized_test_data, nan=0.0)
    
    normalized_train_data = (normalized_train_data - 0.5)*2
    normalized_test_data = (normalized_test_data - 0.5)*2
    
    # standardized_train_data = standardize(interpolated_train_data)
    # standardized_test_data = standardize(interpolated_test_data)
    
    if not img:
        np.save(os.path.join(DATA_DIR, "preprocessed/train_data.npy"), np.asarray(normalized_train_data))
        np.save(os.path.join(DATA_DIR, "preprocessed/test_data.npy"), np.asarray(normalized_test_data))
        
    if img:
        img_train_data = np.asarray(normalized_train_data).reshape(-1, 64, 64)
        img_test_data =  np.asarray(normalized_test_data).reshape(-1, 64, 64)
        
        img_train_data = np.expand_dims(img_train_data, axis=3)
        img_train_data = tf.image.resize(img_train_data, [64, 64]).numpy()
        
        img_test_data = np.expand_dims(img_test_data, axis=3)
        img_test_data = tf.image.resize(img_test_data, [64, 64]).numpy()
        
        img_train_data = np.tile(img_train_data, (1, 1, 1, 1))
        img_train_data = np.transpose(img_train_data, (0, 3, 1, 2))
        
        img_test_data = np.tile(img_test_data, (1, 1, 1, 1))
        img_test_data = np.transpose(img_test_data, (0, 3, 1, 2))

        print("writing img data to disk")
        np.save(os.path.join(DATA_DIR, "preprocessed/img_train_data.npy"), img_train_data)
        np.save(os.path.join(DATA_DIR, "preprocessed/img_test_data.npy"), img_test_data)
        
    return indices
    

def write_cond_data(indices):
    weather_data = load_weather_data(DATA_DIR)
    
    resampled_weather_data = weather_data.resample('30min').interpolate(method='linear')
    filtered_weather_data = resampled_weather_data[resampled_weather_data.index.isin(indices)]
    
    train_cond_data, test_cond_data = train_test_split(filtered_weather_data)
    
    normalized_train_data = normalize(train_cond_data)
    normalized_test_data = normalize(test_cond_data)
    print("writing cond data to disk")
    np.save(os.path.join(DATA_DIR, "preprocessed/cond_train_data.npy"), np.asarray(normalized_train_data))
    np.save(os.path.join(DATA_DIR, "preprocessed/cond_test_data.npy"), np.asarray(normalized_test_data))
    
    
    
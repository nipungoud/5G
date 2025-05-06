import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

def load_and_prepare_data(path, sequence_length=10):
    df = pd.read_csv(path)
    df = df.drop(columns=["timestamp"])  # Drop timestamp

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    X, y = [], []
    for i in range(len(scaled) - sequence_length):
        X.append(scaled[i:i+sequence_length, :-1])  # all features except bandwidth
        y.append(scaled[i+sequence_length, -1])     # target is future bandwidth

    return torch.tensor(X).float(), torch.tensor(y).float(), scaler

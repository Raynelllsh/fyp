import pandas as pd
import numpy as np
import joblib
import torch
import matplotlib.pyplot as plt
from src.config import SCALER_PATH

def load_scaler(scaler_name):
    """Load a saved scaler from disk"""
    path = f"{SCALER_PATH}{scaler_name}.pkl"
    try:
        return joblib.load(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Scaler not found at {path}. Run data merging notebook first.")

def create_sequences(features, targets, seq_len):
    """
    Creates sequences for LSTM/Transformer training.
    
    Args:
        features (np.array): Input features (X)
        targets (np.array): Target values to predict (y)
        seq_len (int): Length of historical sequence
        
    Returns:
        np.array, np.array: X sequences, y targets
    """
    xs, ys = [], []
    # Start loop so we have enough data for one full sequence
    for i in range(len(features) - seq_len):
        x = features[i:(i + seq_len)]
        y = targets[i + seq_len]
        xs.append(x)
        ys.append(y)
        
    return np.array(xs), np.array(ys)

def plot_training_loss(train_losses, val_losses):
    """Plot training and validation loss history"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Model Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    return plt

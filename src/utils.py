import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from src.config import *

def get_sentiment(text, tokenizer, model):
    """
    Get sentiment score from FinBERT for financial text.
    Returns: sentiment score (-1 to 1) and label
    """
    inputs = tokenizer(text, return_tensors="pt", 
                      padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # FinBERT labels: negative, neutral, positive
    sentiment_score = probs[0][2].item() - probs[0][0].item()  # positive - negative
    label = ['negative', 'neutral', 'positive'][probs[0].argmax().item()]
    
    return sentiment_score, label

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def add_technical_indicators(df):
    """Add technical indicators to price dataframe"""
    # Moving averages
    for period in MA_PERIODS:
        df[f'MA_{period}'] = df['Close'].rolling(window=period).mean()
    
    # RSI
    df['RSI'] = calculate_rsi(df['Close'], RSI_PERIOD)
    
    # Price changes
    df['Price_Change'] = df['Close'].pct_change()
    df['Volume_Change'] = df['Volume'].pct_change()
    
    # Volatility (standard deviation)
    df['Volatility'] = df['Close'].rolling(window=20).std()
    
    return df

def create_sequences(data, seq_len):
    """Create sequences for time series prediction"""
    sequences = []
    targets = []
    
    for i in range(len(data) - seq_len):
        sequences.append(data[i:i+seq_len])
        targets.append(data[i+seq_len])
    
    return np.array(sequences), np.array(targets)

def save_scaler(scaler, name):
    """Save MinMaxScaler object"""
    joblib.dump(scaler, f"{SCALER_PATH}{name}.pkl")

def load_scaler(name):
    """Load MinMaxScaler object"""
    return joblib.load(f"{SCALER_PATH}{name}.pkl")

def plot_results(actual, predicted, title="Prediction Results"):
    """Plot actual vs predicted values"""
    plt.figure(figsize=(15, 6))
    plt.plot(actual, label='Actual', alpha=0.7)
    plt.plot(predicted, label='Predicted', alpha=0.7)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt

def plot_training_loss(train_losses, val_losses=None):
    """Plot training and validation loss"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    if val_losses:
        plt.plot(val_losses, label='Validation Loss')
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------- Feature engineering (PAST-ONLY) --------
def engineer_features_past_only(df: pd.DataFrame, sentiment_window: int = 7) -> pd.DataFrame:
    df = df.copy()

    # Sentiment smoothing (past-only rolling window)
    if "sentiment_score" in df.columns:
        df["Sentiment_MA7"] = df["sentiment_score"].rolling(window=sentiment_window, min_periods=1).mean()
        df["Sentiment_Momentum"] = df["sentiment_score"] - df["Sentiment_MA7"]
    else:
        df["Sentiment_MA7"] = 0.0
        df["Sentiment_Momentum"] = 0.0

    # Example: past-only returns/volatility (optional but safe)
    if "Close" in df.columns:
        df["Return_1D"] = df["Close"].pct_change().fillna(0.0)
        df["Volatility_10D"] = df["Return_1D"].rolling(10, min_periods=2).std().fillna(0.0)

    return df


# -------- Label creation (FUTURE) --------
def add_week_ahead_labels(
    df: pd.DataFrame,
    look_ahead: int = 5,
    threshold: float = 0.0, # Threshold doesn't matter for pure binary, use 0.0
    close_col: str = "Close",
) -> pd.DataFrame:
    """
    Adds:
    - Target_Class: 0=Sell, 1=Buy (Binary)
    """
    df = df.copy()
    df["Future_Close"] = df[close_col].shift(-look_ahead)
    df["Future_Return"] = (df["Future_Close"] - df[close_col]) / df[close_col]

    # Binary Classification:
    # If return is positive (> 0) -> Buy (1)
    # If return is negative (<= 0) -> Sell (0)
    df["Target_Class"] = (df["Future_Return"] > 0).astype(int)

    # Drop rows where the future label is not available
    df = df.dropna(subset=["Future_Close", "Future_Return"])
    
    return df


# -------- Sequence building --------
def create_sequences(features: np.ndarray, targets: np.ndarray, seq_len: int):
    X, y = [], []
    # Target at index i+seq_len corresponds to the day AFTER the input window ends (your original convention)
    for i in range(len(features) - seq_len):
        X.append(features[i : i + seq_len])
        y.append(targets[i + seq_len])
    return np.array(X), np.array(y)


# -------- Train/Val split that avoids overlap --------
def time_split_with_gap(df: pd.DataFrame, train_split: float, gap: int):
    """
    Splits by time index:
      train = [0 : train_end)
      val   = [train_end + gap : end)
    The gap prevents the last training window from sharing timesteps with the first validation window.
    """
    n = len(df)
    train_end = int(n * train_split)
    val_start = min(n, train_end + gap)
    df_train = df.iloc[:train_end].copy()
    df_val = df.iloc[val_start:].copy()
    return df_train, df_val


def plot_training_loss(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("Model Training History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    return plt

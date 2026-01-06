import os
import json
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from src.config import *
from src.dataset import StockDataset
from src.model import LSTMModel
from src.utils import (
    engineer_features_past_only,
    add_week_ahead_labels,
    create_sequences,
    time_split_with_gap,
    plot_training_loss,
)

EXCLUDE_COLS = {
    "Date",
    "Open", "High", "Low", "Close", "Volume",
    # Binary target columns from old pipeline
    "Target", "Return", "Recalculated_Target",
    # Multi-class labels / future analysis columns
    "Target_Class", "Future_Close", "Future_Return",
}

def compute_class_weights(y: np.ndarray, num_classes: int = 3):
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)

def train_model(model, train_loader, val_loader, device, num_epochs=NUM_EPOCHS):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    patience, patience_counter = 15, 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct, total = 0, 0

        for Xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            Xb = Xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(Xb)              # [B, 3]
            loss = criterion(logits, yb)    # CE expects class indices
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            pred = torch.argmax(logits, dim=1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)

        avg_train = total_loss / max(1, len(train_loader))
        train_acc = correct / max(1, total)
        train_losses.append(avg_train)

        model.eval()
        total_loss = 0.0
        correct, total = 0, 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb = Xb.to(device)
                yb = yb.to(device)
                logits = model(Xb)
                loss = criterion(logits, yb)
                total_loss += loss.item()
                pred = torch.argmax(logits, dim=1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)

        avg_val = total_loss / max(1, len(val_loader))
        val_acc = correct / max(1, total)
        val_losses.append(avg_val)

        scheduler.step(avg_val)
        print(f"Epoch {epoch+1}: Train Loss={avg_train:.4f} (Acc={train_acc:.2%}) | Val Loss={avg_val:.4f} (Acc={val_acc:.2%})")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            os.makedirs(MODEL_PATH, exist_ok=True)
            torch.save(model.state_dict(), f"{MODEL_PATH}best_model_weekly_3class.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    return train_losses, val_losses


if __name__ == "__main__":
    path = f"{PROCESSED_DATA_PATH}training_data.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")

    df = pd.read_csv(path)

    # Make sure time is sorted
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

    # Past-only features + week-ahead label
    df = engineer_features_past_only(df, sentiment_window=SENTIMENT_WINDOW)
    df = add_week_ahead_labels(df, look_ahead=5, threshold=0.015)

    # Split BEFORE sequences + add gap to avoid overlap leakage
    df_train, df_val = time_split_with_gap(df, train_split=TRAIN_SPLIT, gap=SEQ_LEN)
    print(f"Rows -> train: {len(df_train)}, val: {len(df_val)}")

    # Define feature columns safely
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    if len(feature_cols) == 0:
        raise ValueError("No feature columns found. Check EXCLUDE_COLS vs your CSV columns.")

    # Scale (fit only on train)
    scaler = StandardScaler()
    scaler.fit(df_train[feature_cols].values)

    os.makedirs(SCALER_PATH, exist_ok=True)
    joblib.dump(scaler, f"{SCALER_PATH}feature_scaler_weekly.pkl")

    # Transform
    train_feat = scaler.transform(df_train[feature_cols].values)
    val_feat = scaler.transform(df_val[feature_cols].values)

    y_train_raw = df_train["Target_Class"].astype(int).values
    y_val_raw = df_val["Target_Class"].astype(int).values

    # Build sequences inside each split (no overlap between train and val windows)
    X_train, y_train = create_sequences(train_feat, y_train_raw, SEQ_LEN)
    X_val, y_val = create_sequences(val_feat, y_val_raw, SEQ_LEN)

    print(f"Sequences -> X_train: {X_train.shape}, X_val: {X_val.shape}")
    print("Train class counts:", np.bincount(y_train, minlength=3))
    print("Val class counts:", np.bincount(y_val, minlength=3))

    # Save config for inference
    os.makedirs(MODEL_PATH, exist_ok=True)
    with open(f"{MODEL_PATH}model_config_weekly.json", "w") as f:
        json.dump({"feature_cols": feature_cols, "input_dim": len(feature_cols), "seq_len": SEQ_LEN}, f)

    # Data loaders
    train_loader = DataLoader(StockDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(StockDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    # Model (3-class)
    model = LSTMModel(
        input_dim=len(feature_cols),
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        output_dim=2, # Explicitly set to 2
    ).to(DEVICE)

    train_losses, val_losses = train_model(model, train_loader, val_loader, DEVICE)

    plt = plot_training_loss(train_losses, val_losses)
    plt.savefig(f"{MODEL_PATH}training_history_weekly_3class.png")
    print(f"Saved: {MODEL_PATH}training_history_weekly_3class.png")

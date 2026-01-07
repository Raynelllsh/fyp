import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from src.config import *
from src.model import LSTMModel
from src.utils import engineer_features_past_only, add_week_ahead_labels, create_sequences, time_split_with_gap

# Columns to exclude from features
EXCLUDE_COLS = {
    "Date",
    "Open", "High", "Low", "Close", "Volume",
    "Target", "Return", "Recalculated_Target",
    "Target_Class", "Future_Close", "Future_Return",
}

def backtest():
    # 1. LOAD DATA
    path = f"{PROCESSED_DATA_PATH}training_data.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing data file: {path}")
        
    df = pd.read_csv(path)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

    # 2. PREPROCESS (Must match training exactly)
    # Engineer past-only features
    df = engineer_features_past_only(df, sentiment_window=SENTIMENT_WINDOW)
    
    # Create labels (binary: 0=Sell, 1=Buy)
    # We set threshold=0.0 to force binary separation
    df = add_week_ahead_labels(df, look_ahead=5, threshold=0.0) 

    # Split with gap to avoid leakage
    df_train, df_val = time_split_with_gap(df, train_split=TRAIN_SPLIT, gap=SEQ_LEN)

    print(f"Backtesting on Validation Set: {len(df_val)} rows")

    # 3. LOAD SCALER & CONFIG
    scaler_path = f"{SCALER_PATH}feature_scaler_weekly.pkl"
    config_path = f"{MODEL_PATH}model_config_weekly.json"
    
    if not os.path.exists(scaler_path) or not os.path.exists(config_path):
        raise FileNotFoundError("Scaler or Config missing. Train the model first.")

    scaler = joblib.load(scaler_path)
    with open(config_path, "r") as f:
        cfg = json.load(f)
    
    feature_cols = cfg["feature_cols"]
    
    # Transform validation features
    val_feat = scaler.transform(df_val[feature_cols].values)
    y_val_raw = df_val["Target_Class"].astype(int).values

    # Create Sequences
    X_val, y_val = create_sequences(val_feat, y_val_raw, SEQ_LEN)

    # 4. ALIGN FUTURE RETURNS
    # We want the Future_Return that corresponds to the prediction made at time t.
    # The prediction at index i (using data t-29...t) predicts return for t+1...t+6
    # This return is stored in df at index t + SEQ_LEN (because create_sequences aligns y at i+seq_len)
    future_ret = df_val["Future_Return"].values
    aligned_future_ret = []

    for i in range(len(val_feat) - SEQ_LEN):
        aligned_future_ret.append(future_ret[i + SEQ_LEN])
    
    aligned_future_ret = np.array(aligned_future_ret)

    # 5. LOAD MODEL (Binary Mode)
    model = LSTMModel(
        input_dim=len(feature_cols),
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        output_dim=2,  # FORCE BINARY OUTPUT
    ).to(DEVICE)

    model_path = f"{MODEL_PATH}best_model_weekly_binary.pth" # Ensure your train script saves to this name
    if not os.path.exists(model_path):
        # Fallback to generic name if specific binary name not found
        model_path = f"{MODEL_PATH}best_model_weekly_3class.pth"
        print(f"Warning: Loading {model_path}. Ensure it was trained with output_dim=2!")
        
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # 6. RUN INFERENCE
    probs = []
    with torch.no_grad():
        batch_size = 256
        for i in range(0, len(X_val), batch_size):
            xb = torch.tensor(X_val[i:i+batch_size], dtype=torch.float32).to(DEVICE)
            logits = model(xb)
            pb = F.softmax(logits, dim=1).cpu().numpy()
            probs.append(pb)
    
    probs = np.vstack(probs)
    pred_class = np.argmax(probs, axis=1) # 0 or 1

    # 7. STATEFUL TRADING SIMULATION
    initial = 10000.0
    balance = initial
    equity = [balance]
    
    in_market = False  # Track if we are currently holding a position
    COST = 0.001       # 0.1% transaction cost per trade
    step = 5           # Re-evaluate every 5 days (1 trading week)

    print(f"Starting Simulation | Initial: {initial}")

    for t in range(0, len(pred_class), step):
        action = pred_class[t]        # 0=Sell, 1=Buy
        r = aligned_future_ret[t]     # Return for the UPCOMING week
        
        # --- DECISION LOGIC ---
        if in_market:
            if action == 0: # SELL Signal -> Exit
                balance = balance * (1.0 - COST)
                in_market = False
                # We exited, so we MISS the return 'r'
            elif action == 1: # BUY Signal -> Hold
                pass # Stay invested
                
        else: # Currently in Cash
            if action == 1: # BUY Signal -> Enter
                balance = balance * (1.0 - COST)
                in_market = True
                # We entered, so we CATCH the return 'r'
            elif action == 0: # SELL Signal -> Stay Out
                pass # Stay in cash

        # --- APPLY RETURN ---
        if in_market:
            balance = balance * (1.0 + r)
        
        equity.append(balance)

    # 8. RESULTS & PLOTTING
    wins = 0
    losses = 0
    total_trades = 0
    
    # Iterate through the decisions again to count
    for t in range(0, len(pred_class), step):
        action = pred_class[t]
        r = aligned_future_ret[t]
        
        # We only care if we took a position (Action = 1 -> Buy)
        # Note: In your binary version, 1 is Buy.
        if action == 1: 
            total_trades += 1
            if r > 0:
                wins += 1
            else:
                losses += 1

    if total_trades > 0:
        win_rate = (wins / total_trades) * 100
        print(f"Total Trades: {total_trades}")
        print(f"Wins: {wins} | Losses: {losses}")
        print(f"Win Rate: {win_rate:.2f}%")
    else:
        print("No trades were taken.")

    total_return = (balance - initial) / initial * 100.0
    print(f"Final balance: {balance:.2f} | Total return: {total_return:.2f}%")
    
    # Calculate Buy & Hold for comparison
    buy_hold_return = (1 + aligned_future_ret).cumprod()[-1] - 1
    print(f"Buy & Hold Return (Benchmark): {buy_hold_return * 100:.2f}%")

    plt.figure(figsize=(12, 6))
    plt.plot(equity, label="AI Strategy (Binary Force)", color="blue")
    
    # Optional: Plot Buy & Hold equity curve for visual comparison
    # bh_equity = initial * (1 + aligned_future_ret).cumprod()
    # bh_equity_sampled = bh_equity[::step] # Sample to match step size roughly
    # plt.plot(bh_equity_sampled, label="Buy & Hold", color="gray", alpha=0.5, linestyle="--")

    plt.title(f"Weekly Backtest (Binary Force) | Initial {initial:.0f} -> Final {balance:.2f}")
    plt.xlabel("Weeks")
    plt.ylabel("Portfolio Value")
    plt.grid(True)
    plt.legend()
    
    os.makedirs(MODEL_PATH, exist_ok=True)
    save_path = f"{MODEL_PATH}backtest_weekly_binary.png"
    plt.savefig(save_path)
    print(f"Saved plot to: {save_path}")

if __name__ == "__main__":
    backtest()

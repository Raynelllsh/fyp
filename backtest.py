import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.config import *
from src.model import TransformerModel
from src.utils import load_scaler

# Ensure DEVICE is defined
if 'DEVICE' not in locals():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def backtest_model(days_to_test=50):
    print(f"Starting Backtest on {DEVICE} for last {days_to_test} days...")

    # 1. Load Data
    df = pd.read_csv(f'{PROCESSED_DATA_PATH}training_data.csv')
    
    # 2. Load Scalers
    try:
        return_scaler = load_scaler('return_scaler')
        price_scaler = load_scaler('target_scaler')
    except Exception as e:
        print(f"Error loading scalers: {e}")
        return

    # 3. Prepare Features (Exact same logic as inference.py)
    drop_cols = ['Date', 'Close']
    feature_cols = [col for col in df.columns if col not in drop_cols]
    
    # We need the RAW close prices for calculating actual returns and errors
    # (Assuming 'Close' in CSV is scaled, we need to unscale it to get real $)
    scaled_close_prices = df['Close'].values.reshape(-1, 1)
    real_close_prices = price_scaler.inverse_transform(scaled_close_prices).flatten()
    
    # Features for the model
    all_features = df[feature_cols].values

    # 4. Load Model
    input_dim = len(feature_cols)
    model = TransformerModel(
        input_dim=input_dim,
        d_model=HIDDEN_DIM,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        output_dim=1
    ).to(DEVICE)
    
    model.load_state_dict(torch.load(f'{MODEL_PATH}best_model.pth', map_location=DEVICE))
    model.eval()

    # 5. Backtest Loop
    predictions = []
    actuals = []
    
    # We start 'days_to_test' days ago. 
    # For each day 'i', we look at the window [i-SEQ_LEN : i] to predict i+1
    start_index = len(df) - days_to_test
    
    print(f"{'Day':<10} {'Actual':<10} {'Predicted':<10} {'Diff %':<10} {'Direction'}")
    print("-" * 55)

    correct_direction_count = 0

    with torch.no_grad():
        for i in range(days_to_test):
            current_idx = start_index + i
            
            # Data available BEFORE today (window of size SEQ_LEN)
            # We want to predict the close price at 'current_idx'
            # So we use data from [current_idx - SEQ_LEN : current_idx]
            
            input_window = all_features[current_idx - SEQ_LEN : current_idx]
            
            # Check if we have enough data
            if len(input_window) != SEQ_LEN:
                print(f"Skipping index {current_idx}: insufficient history")
                continue

            # PREDICT
            input_tensor = torch.FloatTensor(input_window).unsqueeze(0).to(DEVICE)
            pred_scaled_return = model(input_tensor).cpu().item()
            
            # Unscale the predicted return
            pred_return = return_scaler.inverse_transform([[pred_scaled_return]])[0][0]
            
            # Calculate Predicted Price
            # Previous day's actual close (real $)
            prev_close = real_close_prices[current_idx - 1]
            pred_price = prev_close * (1 + pred_return)
            
            # Actual Price for this day
            actual_price = real_close_prices[current_idx]
            
            # Metrics
            diff_pct = ((pred_price - actual_price) / actual_price) * 100
            
            # Direction Accuracy
            actual_move = actual_price - prev_close
            pred_move = pred_price - prev_close
            
            direction_match = (actual_move > 0 and pred_move > 0) or (actual_move < 0 and pred_move < 0)
            if direction_match:
                correct_direction_count += 1
                dir_str = "correct"
            else:
                dir_str = "incorrect"

            predictions.append(pred_price)
            actuals.append(actual_price)
            
            print(f"{i+1:<10} {actual_price:<10.2f} {pred_price:<10.2f} {diff_pct:<10.2f} {dir_str}")

    # 6. Summary Stats
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    mae = np.mean(np.abs(predictions - actuals))
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    direction_acc = (correct_direction_count / days_to_test) * 100

    print("-" * 55)
    print(f"Backtest Complete.")
    print(f"MAE: ${mae:.2f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"Directional Accuracy: {direction_acc:.2f}%")

    # 7. Plot
    plt.figure(figsize=(12, 6))
    plt.plot(actuals, label='Actual Price', color='black')
    plt.plot(predictions, label='Predicted Price', color='blue', linestyle='--')
    plt.title(f'Backtest Last {days_to_test} Days')
    plt.xlabel('Days')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{MODEL_PATH}backtest_result.png')
    print(f"Chart saved to {MODEL_PATH}backtest_result.png")

if __name__ == "__main__":
    backtest_model(days_to_test=50)

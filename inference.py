"""

import torch
import pandas as pd
import numpy as np
from src.config import *
from src.model import TransformerModel, DirectionalLoss
from src.utils import load_scaler

# Ensure DEVICE is defined if config.py doesn't export it globally
if 'DEVICE' not in locals():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_next_day(model, recent_data, device):
    # Predict the next day's closing price
    model.eval()
    with torch.no_grad():
        # Convert to tensor: (1, seq_len, input_dim)
        input_tensor = torch.FloatTensor(recent_data).unsqueeze(0).to(device)
        prediction = model(input_tensor)
    
    # Return the scalar value (still scaled 0-1)
    return prediction.cpu().item()

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    # 1. Load processed data
    # We need the last SEQ_LEN rows to predict "tomorrow"
    df = pd.read_csv(f'{PROCESSED_DATA_PATH}training_data.csv')
    
    # 2. Load scalers
    # We need 'target_scaler' to convert the 0-1 prediction back to $$$
    try:
        target_scaler = load_scaler('target_scaler')
        print("Target scaler loaded.")
    except Exception as e:
        print(f"Error loading scaler: {e}")
        exit()

    # 3. Prepare features
    # Drop 'Date' and 'Close' because 'Close' is the target (y), not a feature (X)
    # UNLESS your model uses 'Close' as an input feature (autoregressive).
    # CHECK: Did you train with 'Close' inside X? 
    # If yes (common for time series), remove 'Close' from the drop list below.
    
    # Assuming 'Close' IS used as an input feature (lagged value):
    # Replace the dynamic list with this explicit list
    # inference.py

# REPLACE THIS:
# feature_cols = ['Open', ..., 'Sentiment_Score', 'Another_Feature', 'One_More']

# WITH SOMETHING LIKE THIS (Check your specific CSV headers):


    feature_cols = [
        'High', 'Low', 'Open', 'Volume', 'sentiment_score', 
        'MA_5', 'MA_10', 'MA_20', 'MA_50', 'RSI', 
        'Price_Change', 'Volume_Change', 'Volatility'
    ]

    input_dim = len(feature_cols)
    print(f"Input Features: {input_dim}")

    # 4. Load model
    model = TransformerModel(
        input_dim=input_dim,
        d_model=HIDDEN_DIM,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        output_dim=1 # We predict 1 value (Price)
    ).to(DEVICE)
    
    try:
        model.load_state_dict(torch.load(f'{MODEL_PATH}best_model.pth', map_location=DEVICE))
        print("Model loaded successfully")
    except RuntimeError as e:
        print(f"Model load error: {e}")
        print("Check if INPUT_DIM matches the model training (13 vs 5 mismatch?)")
        exit()

    # 5. Get last SEQ_LEN days
    # This grabs the last 60 rows of ALL feature columns
    recent_data = df[feature_cols].values[-SEQ_LEN:]
    
    if len(recent_data) < SEQ_LEN:
        print(f"Error: Not enough data. Need {SEQ_LEN} rows, got {len(recent_data)}")
        exit()

    # 6. Predict
    prediction_scaled = predict_next_day(model, recent_data, DEVICE)
    print(f"Raw Scaled Prediction: {prediction_scaled}")

    # 7. Inverse transform
    # The scaler expects a 2D array [[value]]
    prediction_real = target_scaler.inverse_transform([[prediction_scaled]])[0][0]
    
    # FIX: Get the last actual close (which is currently scaled 0-1) and unscale it
    last_actual_scaled = df['Close'].iloc[-1]
    last_actual_close = target_scaler.inverse_transform([[last_actual_scaled]])[0][0]

    print("-" * 30)
    print(f"PREDICTION REPORT")
    print("-" * 30)
    print(f"Last Actual Close:    ${last_actual_close:.2f}")
    print(f"Predicted Next Close: ${prediction_real:.2f}")
    
    change_pct = ((prediction_real / last_actual_close) - 1) * 100
    direction = "UP" if change_pct > 0 else "DOWN"
    
    print(f"Predicted Move:       {direction} ({change_pct:.2f}%)")
    print("-" * 30)

"""

import torch
import pandas as pd
import numpy as np
from src.config import *
from src.model import TransformerModel
from src.utils import load_scaler

# Ensure DEVICE is defined
if 'DEVICE' not in locals():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_next_day(model, recent_data, device):
    """Predict the next day's % return"""
    model.eval()
    with torch.no_grad():
        input_tensor = torch.FloatTensor(recent_data).unsqueeze(0).to(device)
        prediction = model(input_tensor)
    return prediction.cpu().item()

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    # 1. Load processed data
    df = pd.read_csv(f'{PROCESSED_DATA_PATH}training_data.csv')
    
    # 2. Load scalers
    # We need 'return_scaler' for the prediction, and 'target_scaler' to get the real price of yesterday
    try:
        return_scaler = load_scaler('return_scaler') # The new one created by train.py
        price_scaler = load_scaler('target_scaler')  # The original one for prices
        print("Scalers loaded.")
    except Exception as e:
        print(f"Error loading scalers: {e}")
        print("Did you run the UPDATED train.py? It creates 'return_scaler.pkl'.")
        exit()

    # 3. Prepare features
    # Ensure this list matches exactly what you trained on (excluding Date/Close)
    # Based on your previous files, this logic removes Date/Close automatically
    drop_cols = ['Date', 'Close']
    feature_cols = [col for col in df.columns if col not in drop_cols]
    
    input_dim = len(feature_cols)
    print(f"Input Features: {input_dim}")

    # 4. Load model
    model = TransformerModel(
        input_dim=input_dim,
        d_model=HIDDEN_DIM,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        output_dim=1 
    ).to(DEVICE)
    
    try:
        model.load_state_dict(torch.load(f'{MODEL_PATH}best_model.pth', map_location=DEVICE))
        print("Model loaded successfully")
    except RuntimeError as e:
        print(f"Model load error: {e}")
        exit()

    # 5. Get last SEQ_LEN days
    recent_data = df[feature_cols].values[-SEQ_LEN:]
    
    if len(recent_data) < SEQ_LEN:
        print(f"Error: Not enough data. Need {SEQ_LEN} rows, got {len(recent_data)}")
        exit()

    # 6. Predict (Output is Scaled Return)
    prediction_scaled = predict_next_day(model, recent_data, DEVICE)
    print(f"Raw Scaled Prediction (Return): {prediction_scaled}")

    # 7. Inverse transform to get Real Percentage Return
    # The scaler expects [[value]]
    predicted_return_pct = return_scaler.inverse_transform([[prediction_scaled]])[0][0]
    
    # 8. Calculate Price
    # Get last actual close price (Unscale it first)
    last_actual_scaled = df['Close'].iloc[-1]
    last_actual_close = price_scaler.inverse_transform([[last_actual_scaled]])[0][0]
    
    # Apply return
    predicted_price = last_actual_close * (1 + predicted_return_pct)
    
    print("-" * 30)
    print(f"PREDICTION REPORT (BOLD STRATEGY)")
    print("-" * 30)
    print(f"Last Actual Close:    ${last_actual_close:.2f}")
    print(f"Predicted Return:     {predicted_return_pct*100:.4f}%")
    print(f"Predicted Next Close: ${predicted_price:.2f}")
    
    direction = "UP" if predicted_return_pct > 0 else "DOWN"
    print(f"Predicted Move:       {direction}")
    print("-" * 30)

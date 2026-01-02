import torch
import pandas as pd
import numpy as np
from src.config import *
from src.model import TransformerModel
from src.utils import load_scaler

def predict_next_day(model, recent_data, device):
    """Predict the next day's closing price"""
    model.eval()
    with torch.no_grad():
        input_tensor = torch.FloatTensor(recent_data).unsqueeze(0).to(device)
        prediction = model(input_tensor)
    return prediction.cpu().numpy()[0][0]

if __name__ == "__main__":
    # Load processed data
    df = pd.read_csv(f'{PROCESSED_DATA_PATH}training_data.csv')
    
    # Load scaler
    scaler = load_scaler('price_scaler')
    
    # Prepare features
    feature_cols = [col for col in df.columns if col not in ['Date', 'Close']]
    input_dim = len(feature_cols)
    
    # Load model
    model = TransformerModel(
        input_dim=input_dim,
        d_model=HIDDEN_DIM,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)
    
    model.load_state_dict(torch.load(f'{MODEL_PATH}best_model.pth'))
    print("Model loaded successfully")
    
    # Get last SEQ_LEN days
    recent_data = df[feature_cols].values[-SEQ_LEN:]
    
    # Predict
    prediction_scaled = predict_next_day(model, recent_data, DEVICE)
    
    # Inverse transform (assuming Close price is first feature)
    prediction = scaler.inverse_transform([[prediction_scaled]])[0][0]
    
    print(f"\nPredicted HSI closing price for next trading day: {prediction:.2f}")
    print(f"Last actual closing price: {df['Close'].iloc[-1]:.2f}")
    print(f"Predicted change: {((prediction / df['Close'].iloc[-1]) - 1) * 100:.2f}%")

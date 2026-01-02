import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

from src.config import *
from src.dataset import StockDataset
from src.model import TransformerModel, LSTMModel, HybridModel
from src.utils import plot_training_loss, load_scaler

def train_model(model, train_loader, val_loader, num_epochs, device):
    """Train the model with early stopping"""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for sequences, targets in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            sequences = sequences.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences = sequences.to(device)
                targets = targets.to(device)
                outputs = model(sequences)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        scheduler.step(avg_val_loss)
        
        print(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}')
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f'{MODEL_PATH}best_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print("Early stopping triggered")
                break
    
    return train_losses, val_losses

if __name__ == "__main__":
    # Load processed data
    print("Loading data...")
    df = pd.read_csv(f'{PROCESSED_DATA_PATH}training_data.csv')
    
    # Prepare features (exclude Date and target)
    feature_cols = [col for col in df.columns if col not in ['Date', 'Close']]
    features = df[feature_cols].values
    targets = df['Close'].values.reshape(-1, 1)
    
    # Create sequences
    from src.utils import create_sequences
    X, y = create_sequences(features, SEQ_LEN)
    
    # Split data
    train_size = int(len(X) * TRAIN_SPLIT)
    val_size = int(len(X) * VAL_SPLIT)
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    
    # Create datasets
    train_dataset = StockDataset(X_train, y_train)
    val_dataset = StockDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model
    input_dim = X_train.shape[2]
    model = TransformerModel(
        input_dim=input_dim,
        d_model=HIDDEN_DIM,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)
    
    print(f"Model initialized on {DEVICE}")
    print(f"Input dimension: {input_dim}")
    
    # Train
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, NUM_EPOCHS, DEVICE
    )
    
    # Plot and save
    plt = plot_training_loss(train_losses, val_losses)
    plt.savefig(f'{MODEL_PATH}training_history.png')
    print(f"Training complete! Model saved to {MODEL_PATH}best_model.pth")

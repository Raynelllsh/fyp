"""
train.py - Leakage-Free Version
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Import project modules
from src.config import *
from src.dataset import StockDataset
from src.model import TransformerModel, DirectionalLoss
from src.utils import plot_training_loss, create_sequences, load_scaler

def train_model(model, train_loader, val_loader, num_epochs, device):
    criterion = DirectionalLoss(alpha=10.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0

    print(f"Starting training on {device}...")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=True)

        for sequences, targets in loop:
            sequences = sequences.float().to(device)
            targets = targets.float().to(device)

            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences = sequences.float().to(device)
                targets = targets.float().to(device)

                outputs = model(sequences)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        scheduler.step(avg_val_loss)

        print(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(MODEL_PATH, exist_ok=True)
            torch.save(model.state_dict(), f'{MODEL_PATH}best_model.pth')
            patience_counter = 0
            print(" Model saved (New best validation loss)")
        else:
            patience_counter += 1
            if patience_counter >= 15:
                print(" Early stopping triggered")
                break

    return train_losses, val_losses

if __name__ == "__main__":
    print("Loading processed data...")
    if not os.path.exists(f'{PROCESSED_DATA_PATH}training_data.csv'):
        raise FileNotFoundError("Run data processing notebooks first!")

    df = pd.read_csv(f'{PROCESSED_DATA_PATH}training_data.csv')

    # ---------------------------------------------------------
    # 1. SPLIT DATA FIRST (To prevent leakage)
    # ---------------------------------------------------------
    # We need to preserve temporal order.
    # We will split the raw DataFrame indices first.
    total_len = len(df)
    train_end = int(total_len * TRAIN_SPLIT)
    val_end = int(total_len * (TRAIN_SPLIT + VAL_SPLIT))

    # Indices
    train_indices = range(0, train_end)
    val_indices = range(train_end, val_end)
    test_indices = range(val_end, total_len)

    print(f"Total samples: {total_len}")
    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")

    # ---------------------------------------------------------
    # 2. PREPARE TARGETS (Returns)
    # ---------------------------------------------------------
    try:
        # Load the original price scaler (needed to unscale 'Close')
        price_scaler = load_scaler('target_scaler')
        
        # Unscale Close prices to get real values
        scaled_close = df['Close'].values.reshape(-1, 1)
        real_close = price_scaler.inverse_transform(scaled_close).flatten()
        
        # Calculate Percentage Returns
        returns = pd.Series(real_close).pct_change().fillna(0).values.reshape(-1, 1)

        # --- LEAKAGE FIX: FIT SCALER ONLY ON TRAIN DATA ---
        train_returns = returns[train_indices]
        
        return_scaler = MinMaxScaler(feature_range=(-1, 1))
        return_scaler.fit(train_returns)  # Fit ONLY on training data
        
        # Transform everything using the training scaler
        returns_scaled = return_scaler.transform(returns)
        
        # Save this rigorous scaler
        joblib.dump(return_scaler, f'{SCALER_PATH}return_scaler.pkl')
        print("New leakage-free 'return_scaler.pkl' saved.")
        
        targets = returns_scaled

    except Exception as e:
        print(f"Error processing targets: {e}")
        exit()

    # ---------------------------------------------------------
    # 3. PREPARE FEATURES
    # ---------------------------------------------------------
    drop_cols = ['Date', 'Close']
    feature_cols = [col for col in df.columns if col not in drop_cols]
    features = df[feature_cols].values

    # ---------------------------------------------------------
    # 4. CREATE SEQUENCES
    # ---------------------------------------------------------
    # We create sequences for the WHOLE dataset, then index them based on our split
    X, y = create_sequences(features, targets, SEQ_LEN)

    # Adjust split indices because create_sequences reduces length by SEQ_LEN
    # We need to align the split with the new array length
    # Logic: The first sequence starts at index 0 of features, but effectively represents time [0..SEQ_LEN]
    # Simple approach: Split the resulting X, y arrays using the same ratios
    
    total_seqs = len(X)
    train_seq_end = int(total_seqs * TRAIN_SPLIT)
    val_seq_end = int(total_seqs * (TRAIN_SPLIT + VAL_SPLIT))

    X_train = X[:train_seq_end]
    y_train = y[:train_seq_end]
    
    X_val = X[train_seq_end:val_seq_end]
    y_val = y[train_seq_end:val_seq_end]
    
    # (Optional) X_test = X[val_seq_end:]

    print(f"Training sequences: {X_train.shape}")
    print(f"Validation sequences: {X_val.shape}")

    # ---------------------------------------------------------
    # 5. DATA LOADERS
    # ---------------------------------------------------------
    train_dataset = StockDataset(X_train, y_train)
    val_dataset = StockDataset(X_val, y_val)

    use_pin_memory = True if DEVICE.type == 'cuda' else False

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,  # Shuffle ONLY training data
        pin_memory=use_pin_memory,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False, # Never shuffle validation
        pin_memory=use_pin_memory,
        num_workers=0
    )

    # ---------------------------------------------------------
    # 6. MODEL SETUP & TRAINING
    # ---------------------------------------------------------
    input_dim = X_train.shape[2]
    
    model = TransformerModel(
        input_dim=input_dim,
        d_model=HIDDEN_DIM,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)

    print(f"Model initialized on {DEVICE}")
    
    try:
        train_losses, val_losses = train_model(
            model, train_loader, val_loader, NUM_EPOCHS, DEVICE
        )

        plt = plot_training_loss(train_losses, val_losses)
        plt.savefig(f'{MODEL_PATH}training_history.png')
        print(f"Training complete! Chart saved to {MODEL_PATH}training_history.png")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current progress...")

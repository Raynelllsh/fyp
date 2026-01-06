import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]
    
class DirectionalLoss(nn.Module):
    def __init__(self, alpha=10.0): # alpha is how hard to punish wrong direction
        super(DirectionalLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.alpha = alpha 

    def forward(self, pred, target):
        loss = self.mse(pred, target)
        # 1 if signs differ, 0 if they match
        sign_mismatch = torch.where(pred * target < 0, 1.0, 0.0)
        # Add penalty only where signs are wrong
        penalty = self.alpha * torch.mean(sign_mismatch * torch.abs(pred - target))
        return loss + penalty

class TransformerModel(nn.Module):
    """
    Transformer-based time series prediction model
    """
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=3, 
                 dropout=0.2, output_dim=1):
        super(TransformerModel, self).__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, 
            num_layers=num_layers
        )
        
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        # Take the last timestep
        x = x[:, -1, :]
        x = self.fc(x)
        return x
    

class LSTMModel(nn.Module):
    # Update output_dim default to 2
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2, output_dim=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim) # Output is now size 2
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        return self.fc(last_out)
class HybridModel(nn.Module):
    """
    Hybrid Transformer + LSTM model
    """
    def __init__(self, input_dim, d_model=128, nhead=8, 
                 hidden_dim=128, dropout=0.2, output_dim=1):
        super(HybridModel, self).__init__()
        
        # Transformer branch
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, 
            dim_feedforward=d_model * 4, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # LSTM branch
        self.lstm = nn.LSTM(d_model, hidden_dim, num_layers=2, 
                           batch_first=True, dropout=dropout)
        
        # Output layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, x):
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])
        return output

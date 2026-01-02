import torch

# Data parameters
SEQ_LEN = 60  # Number of days to look back
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1

# Model parameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
HIDDEN_DIM = 128
NUM_LAYERS = 3
DROPOUT = 0.2
NHEAD = 8  # Number of attention heads for Transformer

# Sentiment parameters
FINBERT_MODEL = "ProsusAI/finbert"
SENTIMENT_WINDOW = 7  # Days to aggregate sentiment

# Technical indicators
RSI_PERIOD = 14
MA_PERIODS = [5, 10, 20, 50]

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
RAW_DATA_PATH = "data/raw/"
PROCESSED_DATA_PATH = "data/processed/"
MODEL_PATH = "models/"
SCALER_PATH = "data/processed/scalers/"

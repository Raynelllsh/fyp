import torch

# -------------------
# DEVICE CONFIGURATION
# -------------------
# Check for CUDA (Windows) or MPS (Mac)
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using Apple Silicon MPS")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU")

# -------------------
# DATA PARAMETERS
# -------------------
SEQ_LEN = 30         # Lookback window
TRAIN_SPLIT = 0.8    # 80% training data
VAL_SPLIT = 0.1      # 10% validation data

# -------------------
# MODEL PARAMETERS
# -------------------
BATCH_SIZE = 128      # Larger batch size = faster training on GPU
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
HIDDEN_DIM = 16
NUM_LAYERS = 1
DROPOUT = 0.6
NHEAD = 8            # Attention heads (Transformer only)

# -------------------
# SENTIMENT PARAMETERS
# -------------------
FINBERT_MODEL = "ProsusAI/finbert"
SENTIMENT_WINDOW = 7

# -------------------
# TECHNICAL INDICATORS
# -------------------
RSI_PERIOD = 14
MA_PERIODS = [5, 10, 20, 50]

# -------------------
# PATHS
# -------------------
RAW_DATA_PATH = "data/raw/"
PROCESSED_DATA_PATH = "data/processed/"
MODEL_PATH = "models/"
SCALER_PATH = "data/processed/scalers/"

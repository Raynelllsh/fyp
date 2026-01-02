import torch
from torch.utils.data import Dataset
import numpy as np

class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for time series with windowing
    """
    def __init__(self, data, seq_len, target_idx=0):
        """
        Args:
            data: numpy array of shape (num_samples, num_features)
            seq_len: length of sequence window
            target_idx: index of target feature in data
        """
        self.seq_len = seq_len
        self.target_idx = target_idx
        
        # Create sequences
        self.sequences = []
        self.targets = []
        
        for i in range(len(data) - seq_len):
            self.sequences.append(data[i:i+seq_len])
            self.targets.append(data[i+seq_len, target_idx])
        
        self.sequences = np.array(self.sequences)
        self.targets = np.array(self.targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = torch.FloatTensor(self.sequences[idx])
        target = torch.FloatTensor([self.targets[idx]])
        return sequence, target

class StockDataset(Dataset):
    """
    Enhanced dataset with multiple features
    """
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

import os
import torch
import numpy as np
from datetime import datetime
import warnings


warnings.filterwarnings("ignore")
seed = 105


def set_seed(seed=seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_save_dir():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("output_folder", timestamp)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir, timestamp


def train_val_test_split(dataX, datay, shuffle=False, train_percentage=0.6, val_percentage=0.2, test_percentage=0.2):
    if shuffle:
        indices = np.arange(len(dataX))
        np.random.shuffle(indices)
        dataX, datay = dataX[indices], datay[indices]

    total_samples = len(dataX)
    train_idx = int(total_samples * train_percentage)
    val_idx = int(total_samples * (train_percentage + val_percentage))

    train_X, train_y = dataX[:train_idx], datay[:train_idx]
    val_X, val_y = dataX[train_idx:val_idx], datay[train_idx:val_idx]
    test_X, test_y = dataX[val_idx:], datay[val_idx:]


    return train_X, train_y, val_X, val_y, test_X, test_y

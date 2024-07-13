import os
import numpy as np
import torch
from typing import Optional, Callable
from scipy.signal import resample, butter, filtfilt

def preprocess_data(data, target_sr, lowcut=0.5, highcut=40.0):
    # リサンプリング
    original_sr = 200  # 元のサンプリングレート
    data = resample(data, int(len(data) * target_sr / original_sr))
    
    # フィルタリング
    nyquist = 0.5 * target_sr
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(1, [low, high], btype='band')
    data = filtfilt(b, a, data)
    
    # ベースライン補正
    baseline = np.mean(data[:int(target_sr * 0.2)])  # 最初の200msをベースラインとして使用
    data = data - baseline
    
    return data

class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data", scaler=None, preprocess_fn: Optional[Callable] = None) -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        self.scaler = scaler
        self.preprocess_fn = preprocess_fn

        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt")).numpy()
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))

        if self.scaler:
            self.X = self.scaler.fit_transform(self.X.reshape(-1, self.X.shape[-1])).reshape(self.X.shape)

        self.X = torch.tensor(self.X, dtype=torch.float32)

        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."
        
        # データの前処理を適用する
        if self.preprocess_fn is not None:
            self.X = self._apply_preprocessing(self.X)
        
        # Add num_subjects attribute
        self.num_subjects = len(torch.unique(self.subject_idxs))
            
    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        if hasattr(self, "y"):
            return self.X[i], self.y[i], self.subject_idxs[i]
        else:
            return self.X[i], self.subject_idxs[i]
        
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]
        
    def _apply_preprocessing(self, data):
        return torch.tensor(np.array([self.preprocess_fn(sample.numpy()) for sample in data]), dtype=torch.float32)

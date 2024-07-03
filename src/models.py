import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size: int = 3, p_drop: float = 0.1) -> None:
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        
        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            X = self.conv0(X) + X  # skip connection
        else:
            X = self.conv0(X)

        X = F.gelu(self.batchnorm0(X))

        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))

        return self.dropout(X)

class BasicConvClassifier(nn.Module):
    def __init__(self, num_classes: int, seq_len: int, in_channels: int, hid_dim: int = 128) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim),
            ConvBlock(hid_dim, hid_dim),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim, num_classes),
        )
        
        self.dropout = nn.Dropout(0.5)  # ドロップアウト層を追加

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.blocks(X)
        X = self.head(X)
        X = self.dropout(X)  # ドロップアウトを適用
        
        return X

class SubjectSpecificConvClassifier(nn.Module):
    def __init__(self, num_classes, seq_len, num_channels, num_subjects):
        super(SubjectSpecificConvClassifier, self).__init__()
        self.num_subjects = num_subjects

        self.conv1 = nn.Conv2d(1, 32, kernel_size=(num_channels, 1))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(1, 3))
        self.fc1 = nn.Linear(64 * (seq_len - 2), 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x, subject_idxs):
        x = x.unsqueeze(1)  # (batch_size, 1, num_channels, seq_len)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # (batch_size, 64 * (seq_len - 2))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
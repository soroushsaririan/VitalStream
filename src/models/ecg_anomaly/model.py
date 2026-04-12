"""
models/ecg_anomaly/model.py

1D-CNN + BiLSTM ensemble for ECG anomaly detection.

Architecture rationale:
  - 1D-CNN layers: extract local morphological features from ECG waveforms
    (QRS complex shape, P-wave presence, ST-segment deviation). Receptive
    field grows with depth via strided convolutions.
  - BiLSTM: capture temporal dependencies across the window (rhythm regularity,
    R-R interval patterns). Bidirectional to see both past and future context
    within the fixed window.
  - The combination outperforms either architecture alone on imbalanced
    arrhythmia datasets (PhysioNet Challenge benchmark).

Input:  (batch_size, 1, WINDOW_SAMPLES)  — single-channel, normalised
Output: (batch_size, num_classes)         — logits (not softmax)

Training loss: Focal Loss with class-frequency-derived gamma to handle the
severe class imbalance typical in arrhythmia datasets (~95% NORMAL).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Conv1d → BatchNorm → GELU → optional residual skip.

    GELU activation outperforms ReLU on time-series classification tasks
    (smoother gradient landscape for the LSTM downstream).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()

        self.residual: nn.Module | None = None
        if use_residual and (in_channels != out_channels or stride != 1):
            self.residual = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.bn(self.conv(x)))
        if self.residual is not None:
            out = out + self.residual(x)
        return out


class ECGAnomalyDetector(nn.Module):
    """
    1D-CNN + BiLSTM anomaly detector for single-channel ECG windows.

    Parameters
    ----------
    window_samples : int
        Number of samples in the input window (default: 2500 for 5s @ 500 Hz).
    num_classes : int
        Number of output classes. Default 5: NORMAL, AFIB, VT, STEMI, NOISE.
    cnn_channels : list[int]
        Output channel counts for each CNN stage.
    lstm_hidden : int
        Hidden state size for the BiLSTM. Output is 2x this (bidirectional).
    lstm_layers : int
        Number of stacked LSTM layers.
    dropout : float
        Dropout probability applied between LSTM layers and before classifier.
    """

    CLASS_LABELS = ["NORMAL", "AFIB", "VT", "STEMI", "NOISE"]

    def __init__(
        self,
        window_samples: int = 2500,
        num_classes: int = 5,
        cnn_channels: list[int] | None = None,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        if cnn_channels is None:
            cnn_channels = [32, 64, 128, 256]

        # CNN feature extractor
        # Input: (B, 1, window_samples) → (B, cnn_channels[-1], reduced_len)
        cnn_layers: list[nn.Module] = []
        in_ch = 1
        for i, out_ch in enumerate(cnn_channels):
            stride = 2 if i % 2 == 1 else 1  # downsample every other block
            cnn_layers.append(
                ConvBlock(in_ch, out_ch, kernel_size=7, stride=stride, use_residual=True)
            )
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_layers)

        # Global average pool → (B, cnn_channels[-1]) — removes sequence dim variance
        self.gap = nn.AdaptiveAvgPool1d(64)  # fixed 64 time steps into LSTM

        # BiLSTM temporal modelling
        # Input: (B, 64, cnn_channels[-1]) → permute to (64, B, cnn_channels[-1])
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=False,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # Classifier head
        lstm_out_size = lstm_hidden * 2  # bidirectional
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_out_size, 64),
            nn.GELU(),
            nn.Linear(64, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, 1, window_samples), dtype float32, z-score normalised.

        Returns
        -------
        torch.Tensor
            Logits of shape (batch_size, num_classes).
        """
        # CNN: (B, 1, W) → (B, C, W')
        features = self.cnn(x)

        # GAP to fixed length: (B, C, W') → (B, C, 64)
        features = self.gap(features)

        # LSTM expects (seq_len, batch, input_size)
        features = features.permute(2, 0, 1)  # (64, B, C)
        lstm_out, _ = self.lstm(features)      # (64, B, lstm_hidden*2)

        # Use final timestep output for classification
        final_hidden = lstm_out[-1]            # (B, lstm_hidden*2)

        return self.classifier(final_hidden)   # (B, num_classes)


class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced multi-class classification.

    Down-weights easy negatives (abundant NORMAL class) and forces the model
    to focus on hard-to-classify minority classes (VT, STEMI).

    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
    """

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(
            logits, targets, weight=self.weight, reduction="none"
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

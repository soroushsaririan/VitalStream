"""
features/signal_processing.py

Shared digital signal processing utilities used by both the stream-processing
Lambda (real-time inference path) and the SageMaker training pipeline
(offline feature engineering).

Keeping this module shared ensures the feature computation at training time
is byte-for-byte identical to the feature computation at inference time,
eliminating training-serving skew.

All functions operate on 1-D numpy arrays (single-channel windows).
Multi-channel support is handled by the caller iterating over channels.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch


def bandpass_filter(
    signal: np.ndarray,
    lowcut: float,
    highcut: float,
    fs: float,
    order: int = 4,
) -> np.ndarray:
    """
    Zero-phase Butterworth bandpass filter using second-order sections (SOS).

    Uses sosfiltfilt (forward-backward pass) to achieve zero phase distortion,
    which is critical for preserving morphological features in ECG/EMG signals
    used for classification.

    sosfiltfilt is numerically more stable than the equivalent ba-form filter,
    especially at higher filter orders on signals with low-frequency drift.

    Parameters
    ----------
    signal : np.ndarray
        1-D array of raw signal samples (µV).
    lowcut : float
        Lower cutoff frequency in Hz.
    highcut : float
        Upper cutoff frequency in Hz.
    fs : float
        Sampling frequency in Hz.
    order : int
        Butterworth filter order. Default 4 provides good roll-off with
        minimal phase distortion artefact from sosfiltfilt.

    Returns
    -------
    np.ndarray
        Filtered signal, same shape as input.

    Notes
    -----
    Recommended cutoffs by modality:
      ECG:  lowcut=0.5,  highcut=40.0  (AHA/AAMI EC11 standard)
      sEMG: lowcut=20.0, highcut=450.0 (SENIAM guidelines)
      cBP:  lowcut=0.1,  highcut=15.0
    """
    nyq = fs / 2.0
    if lowcut >= nyq or highcut >= nyq:
        raise ValueError(
            f"Filter cutoffs ({lowcut}, {highcut}) must be below Nyquist ({nyq} Hz)"
        )
    sos = butter(order, [lowcut / nyq, highcut / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, signal).astype(np.float32)


def spectral_entropy(signal: np.ndarray, fs: float, nperseg: int = 256) -> float:
    """
    Normalised spectral entropy via Welch's power spectral density estimate.

    Interpretation:
      Low entropy  → periodic / rhythmic signal (e.g., normal sinus rhythm)
      High entropy → irregular / noisy signal (e.g., atrial fibrillation, artifact)

    Useful as a pre-filter to flag likely-artifact windows before invoking the
    more expensive SageMaker inference endpoint.

    Parameters
    ----------
    signal : np.ndarray
        1-D filtered signal array.
    fs : float
        Sampling frequency in Hz.
    nperseg : int
        Welch segment length. Trades frequency resolution for variance.
        Default 256 is appropriate for 500 Hz ECG at 5-second windows.

    Returns
    -------
    float
        Normalised spectral entropy in [0, log2(N/2 + 1)].
        Normalise to [0, 1] by dividing by log2(len(psd)).
    """
    _, psd = welch(signal, fs=fs, nperseg=min(nperseg, len(signal)))
    psd_norm = psd / (psd.sum() + 1e-12)
    raw_entropy = float(-np.sum(psd_norm * np.log2(psd_norm + 1e-12)))
    max_entropy = np.log2(len(psd_norm))
    return raw_entropy / max_entropy if max_entropy > 0 else 0.0


def validate_signal_quality(signal: np.ndarray, flatline_threshold: float = 1e-6) -> bool:
    """
    Basic signal quality gate applied before feature extraction.

    Checks for:
      1. Flat-line (zero variance) — indicates lead-off or disconnected sensor.
      2. Saturation clipping — more than 5% of samples at ±ADC rail.
         Clipped signals produce unreliable spectral features.

    Parameters
    ----------
    signal : np.ndarray
        Raw (unfiltered) 1-D signal window.
    flatline_threshold : float
        Minimum standard deviation to pass the flat-line check.

    Returns
    -------
    bool
        True if signal passes quality checks; False if it should be discarded.
    """
    if signal.std() < flatline_threshold:
        return False

    # Clipping check: flag if >5% of samples are within 0.1% of min/max rail
    sig_range = signal.max() - signal.min()
    if sig_range < flatline_threshold:
        return False
    rail_margin = sig_range * 0.001
    clipped = np.sum(
        (signal <= signal.min() + rail_margin) | (signal >= signal.max() - rail_margin)
    )
    if clipped / len(signal) > 0.05:
        return False

    return True


def compute_hrv_rmssd(rr_intervals_ms: np.ndarray) -> float:
    """
    Root Mean Square of Successive Differences (RMSSD) — time-domain HRV metric.

    RMSSD reflects short-term autonomic nervous system variability and is the
    most commonly used HRV metric in clinical telemetry literature.

    Parameters
    ----------
    rr_intervals_ms : np.ndarray
        1-D array of R-R intervals in milliseconds, derived from ECG R-peak
        detection (e.g., Pan-Tompkins algorithm).

    Returns
    -------
    float
        RMSSD in milliseconds. Returns 0.0 if fewer than 2 intervals provided.
    """
    if len(rr_intervals_ms) < 2:
        return 0.0
    successive_diffs = np.diff(rr_intervals_ms.astype(np.float64))
    return float(np.sqrt(np.mean(successive_diffs ** 2)))


def normalise_window(signal: np.ndarray) -> np.ndarray:
    """
    Z-score normalise a signal window to zero mean and unit variance.

    Applied immediately before model inference to remove amplitude-scale
    variation between patients and devices (different ADC gains, electrode
    impedances). The model is trained on normalised windows.

    Parameters
    ----------
    signal : np.ndarray
        1-D filtered signal window.

    Returns
    -------
    np.ndarray
        Normalised signal. Returns zeros if standard deviation is near zero
        (degenerate window that passed the quality gate at a relaxed threshold).
    """
    std = signal.std()
    if std < 1e-9:
        return np.zeros_like(signal, dtype=np.float32)
    return ((signal - signal.mean()) / std).astype(np.float32)

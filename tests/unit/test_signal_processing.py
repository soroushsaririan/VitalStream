"""
Unit tests for src/features/signal_processing.py

These tests have no AWS dependencies. They validate DSP correctness
and edge-case handling for the feature extraction functions.
"""

import numpy as np
import pytest

from src.features.signal_processing import (
    bandpass_filter,
    compute_hrv_rmssd,
    normalise_window,
    spectral_entropy,
    validate_signal_quality,
)

FS = 500.0  # Hz
N  = 2500   # samples (5 seconds at 500 Hz)


# ---------------------------------------------------------------------------
# bandpass_filter
# ---------------------------------------------------------------------------

class TestBandpassFilter:
    def test_output_shape_unchanged(self):
        signal = np.random.randn(N).astype(np.float32)
        filtered = bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=FS)
        assert filtered.shape == signal.shape

    def test_rejects_frequencies_outside_passband(self):
        """
        A 100 Hz sinusoid filtered through a 0.5–40 Hz bandpass should be
        substantially attenuated. We measure steady-state attenuation by
        discarding the first and last 10% of the signal (sosfiltfilt edge
        transients), then assert >10x RMS reduction in the stable interior.
        """
        # Use 30 seconds so edge transients are a small fraction of the window
        n_long = int(FS * 30)
        t = np.linspace(0, n_long / FS, n_long, endpoint=False)
        signal = np.sin(2 * np.pi * 100.0 * t).astype(np.float32)
        filtered = bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=FS)

        # Trim 10% from each end to exclude transient effects
        trim = n_long // 10
        interior_in  = signal[trim:-trim]
        interior_out = filtered[trim:-trim]

        attenuation_ratio = interior_in.std() / (interior_out.std() + 1e-12)
        assert attenuation_ratio > 10.0, (
            f"Expected >10x steady-state attenuation at 100 Hz, got {attenuation_ratio:.2f}x"
        )

    def test_passes_frequencies_inside_passband(self):
        """A 10 Hz sinusoid should pass through with minimal attenuation."""
        t = np.linspace(0, N / FS, N, endpoint=False)
        signal = np.sin(2 * np.pi * 10.0 * t).astype(np.float32)
        filtered = bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=FS)
        # RMS of output should be close to RMS of input (1/sqrt(2) ≈ 0.707)
        assert abs(filtered.std() - signal.std()) < 0.1

    def test_raises_on_cutoff_above_nyquist(self):
        signal = np.random.randn(N).astype(np.float32)
        with pytest.raises(ValueError, match="Nyquist"):
            bandpass_filter(signal, lowcut=0.5, highcut=300.0, fs=FS)

    def test_output_dtype_is_float32(self):
        signal = np.random.randn(N).astype(np.float64)
        filtered = bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=FS)
        assert filtered.dtype == np.float32


# ---------------------------------------------------------------------------
# spectral_entropy
# ---------------------------------------------------------------------------

class TestSpectralEntropy:
    def test_white_noise_has_high_entropy(self):
        """White noise has a flat PSD → maximum entropy."""
        noise = np.random.randn(N).astype(np.float32)
        entropy = spectral_entropy(noise, fs=FS)
        assert entropy > 0.9

    def test_pure_sinusoid_has_low_entropy(self):
        """A single-frequency sinusoid concentrates all power → low entropy."""
        t = np.linspace(0, N / FS, N, endpoint=False)
        signal = np.sin(2 * np.pi * 10.0 * t).astype(np.float32)
        entropy = spectral_entropy(signal, fs=FS)
        assert entropy < 0.2

    def test_output_in_zero_one_range(self):
        signal = np.random.randn(N).astype(np.float32)
        entropy = spectral_entropy(signal, fs=FS)
        assert 0.0 <= entropy <= 1.0


# ---------------------------------------------------------------------------
# validate_signal_quality
# ---------------------------------------------------------------------------

class TestValidateSignalQuality:
    def test_flat_line_fails(self):
        signal = np.zeros(N, dtype=np.float32)
        assert validate_signal_quality(signal) is False

    def test_constant_non_zero_fails(self):
        signal = np.full(N, 1.5, dtype=np.float32)
        assert validate_signal_quality(signal) is False

    def test_normal_ecg_like_signal_passes(self):
        rng = np.random.default_rng(42)
        signal = rng.normal(0, 100, N).astype(np.float32)
        assert validate_signal_quality(signal) is True

    def test_clipped_signal_fails(self):
        """Signal where >5% of samples are at the rail should fail."""
        signal = np.random.randn(N).astype(np.float32)
        signal[:200] = signal.max()   # clip 8% of samples to max rail
        assert validate_signal_quality(signal) is False


# ---------------------------------------------------------------------------
# compute_hrv_rmssd
# ---------------------------------------------------------------------------

class TestComputeHRVRMSSD:
    def test_regular_rr_intervals_returns_zero(self):
        """Perfectly regular rhythm → successive differences all zero → RMSSD=0."""
        rr = np.full(100, 800.0)  # 800ms = 75 bpm, perfectly regular
        assert compute_hrv_rmssd(rr) == pytest.approx(0.0)

    def test_single_interval_returns_zero(self):
        rr = np.array([800.0])
        assert compute_hrv_rmssd(rr) == pytest.approx(0.0)

    def test_known_rmssd_value(self):
        """Manually computed: [800, 820, 780, 800] → diffs=[20,-40,20] → RMSSD=sqrt(mean([400,1600,400]))=sqrt(800)≈28.28"""
        rr = np.array([800.0, 820.0, 780.0, 800.0])
        expected = np.sqrt(800.0)
        assert compute_hrv_rmssd(rr) == pytest.approx(expected, rel=1e-4)


# ---------------------------------------------------------------------------
# normalise_window
# ---------------------------------------------------------------------------

class TestNormaliseWindow:
    def test_output_mean_near_zero(self):
        rng = np.random.default_rng(0)
        signal = rng.normal(1000, 200, N).astype(np.float32)
        normed = normalise_window(signal)
        assert abs(normed.mean()) < 1e-4

    def test_output_std_near_one(self):
        rng = np.random.default_rng(0)
        signal = rng.normal(1000, 200, N).astype(np.float32)
        normed = normalise_window(signal)
        assert abs(normed.std() - 1.0) < 1e-3

    def test_degenerate_input_returns_zeros(self):
        signal = np.full(N, 5.0, dtype=np.float32)
        normed = normalise_window(signal)
        assert np.all(normed == 0.0)

    def test_output_dtype_is_float32(self):
        signal = np.random.randn(N).astype(np.float64)
        normed = normalise_window(signal)
        assert normed.dtype == np.float32

import math

import numpy as np

from scripts.preprocess_and_plot import bandpass_envelope


def test_bandpass_envelope_output_length_matches_downsample_step():
    fs = 512
    n_samples = 3072
    signal = np.random.default_rng(0).normal(size=n_samples)
    env = bandpass_envelope(signal, fs, lo=70, hi=150, window_ms=488, downsample=10)
    assert len(env) == math.ceil(n_samples / 10)


def test_bandpass_envelope_passes_high_gamma_and_rejects_low_frequency():
    fs = 512.0
    n_samples = 2048
    t = np.arange(n_samples) / fs

    in_band = np.sin(2 * np.pi * 100 * t)  # inside 70-150 Hz
    out_of_band = np.sin(2 * np.pi * 10 * t)  # well below 70 Hz

    env_in_band = bandpass_envelope(in_band, fs, lo=70, hi=150, window_ms=488, downsample=1)
    env_out_of_band = bandpass_envelope(
        out_of_band, fs, lo=70, hi=150, window_ms=488, downsample=1
    )

    assert env_in_band.mean() > 5 * env_out_of_band.mean()

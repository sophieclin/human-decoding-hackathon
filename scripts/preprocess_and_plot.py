"""
Signal Preprocessing & Visualization

Two deliverables against a viable sEEG channel:
  1. Raw vs. Common-Average-Referenced (CAR) vs. Laplacian-Referenced trace comparison,
     plus their Welch power spectral densities, to verify the high-gamma (70-150 Hz)
     band stands out.
  2. A 70-150 Hz high-gamma amplitude envelope (bandpass -> Hilbert -> moving average ->
     downsample, per docs/DecodingLogic_Hackathon.pdf) diagnostic plot of one channel's
     activity across trials/task epochs.

Usage:
    python3 scripts/preprocess_and_plot.py <path/to/dataset.mat> [--channel-label LFP12] [--out-dir out/]
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, filtfilt, hilbert, welch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.seeg_io import drop_malformed_trials, load_seeg, viable_channels

FS = 512  # Hz, sampling rate of the sEEG recordings
MIN_TRIAL_SAMPLES = 500  # drop obviously truncated/corrupt trials


def bandpass_envelope(signal, fs, lo=70.0, hi=150.0, window_ms=488.0, downsample=10):
    """70-150 Hz high-gamma amplitude envelope: Butterworth bandpass -> Hilbert
    amplitude -> boxcar moving-average smoothing -> downsampling.

    Matches the preprocessing described in docs/DecodingLogic_Hackathon.pdf: a
    ~488 ms moving-average window (~250 samples at 512 Hz) followed by a step-10
    downsample (~19.5 ms per output sample).
    """
    signal = np.asarray(signal, dtype=np.float64)
    nyq = 0.5 * fs

    b, a = butter(4, [lo / nyq, hi / nyq], btype="bandpass")
    filtered = filtfilt(b, a, signal)

    amplitude = np.abs(hilbert(filtered))

    window_samples = max(1, round(window_ms / 1000.0 * fs))
    smoothed = uniform_filter1d(amplitude, size=window_samples, mode="reflect")

    return smoothed[::downsample]


def _select_channel(channels, channel_label=None, subject=None):
    pool = channels
    if subject is not None:
        pool = [c for c in pool if c.sub == subject]
    if channel_label is not None:
        pool = [c for c in pool if c.channel_label == channel_label]
    if not pool:
        raise ValueError(
            f"No channel matches channel_label={channel_label!r}, subject={subject!r}"
        )
    return pool[0]


def plot_reference_comparison(channel, trial_idx, out_dir):
    trial = channel.trials[trial_idx]
    t = np.arange(len(trial.trial_data)) / FS

    fig, axes = plt.subplots(3, 2, figsize=(12, 8))
    fig.suptitle(
        f"{channel.sub} | {channel.channel_label} | Class {trial.class_label} "
        f"({channel.prefrontal_subdiv})"
    )

    signals = [
        ("Raw sEEG LFP", trial.trial_data, "black"),
        ("Common-Average-Referenced (CAR)", trial.common, "tab:blue"),
        ("Laplacian-Referenced", trial.laplacian, "tab:red"),
    ]

    for row, (title, sig, color) in enumerate(signals):
        axes[row, 0].plot(t, sig, color=color, linewidth=1)
        axes[row, 0].set_title(title)
        axes[row, 0].set_ylabel(r"Amplitude ($\mu$V)")
        axes[row, 0].grid(True, linestyle="--", alpha=0.6)

        freqs, psd = welch(sig, fs=FS, nperseg=min(256, len(sig)))
        axes[row, 1].semilogy(freqs, psd, color=color)
        axes[row, 1].axvspan(70, 150, color="gray", alpha=0.2, label="High-gamma (70-150 Hz)")
        axes[row, 1].set_title(f"{title} - PSD")
        axes[row, 1].set_xlabel("Frequency (Hz)")
        axes[row, 1].set_ylabel("PSD")
        axes[row, 1].legend(fontsize=8)

    axes[-1, 0].set_xlabel("Time (s)")
    fig.tight_layout()

    out_path = os.path.join(out_dir, "reference_comparison.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_high_gamma_diagnostic(channel, out_dir):
    envelopes = [
        bandpass_envelope(trial.laplacian, FS) for trial in channel.trials
    ]
    min_len = min(len(e) for e in envelopes)
    envelopes = np.stack([e[:min_len] for e in envelopes])

    step_s = 10 / FS
    t = np.arange(min_len) * step_s

    fig, ax = plt.subplots(figsize=(10, 5))
    for e in envelopes:
        ax.plot(t, e, color="tab:red", alpha=0.15, linewidth=1)
    ax.plot(t, envelopes.mean(axis=0), color="tab:red", linewidth=2, label="Trial-mean envelope")

    ax.set_title(
        f"High-gamma (70-150 Hz) envelope across task epoch\n"
        f"{channel.sub} | {channel.channel_label} | {channel.prefrontal_subdiv} "
        f"({len(channel.trials)} trials, Laplacian reference)"
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("High-gamma amplitude envelope")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()

    out_path = os.path.join(out_dir, "high_gamma_diagnostic.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_path")
    parser.add_argument("--channel-label", default=None)
    parser.add_argument("--subject", default=None)
    parser.add_argument("--trial-idx", type=int, default=0)
    parser.add_argument("--out-dir", default="out")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading {args.data_path}...")
    channels = load_seeg(args.data_path)
    channels = drop_malformed_trials(channels, min_samples=MIN_TRIAL_SAMPLES)
    kept = viable_channels(channels, min_trials_per_class=3, require_all_classes=True)
    print(f"{len(kept)} / {len(channels)} channels are viable (>=3 trials/class).")

    channel = _select_channel(kept, channel_label=args.channel_label, subject=args.subject)
    print(f"Selected channel: {channel.sub} / {channel.channel_label} ({channel.prefrontal_subdiv})")

    ref_path = plot_reference_comparison(channel, args.trial_idx, args.out_dir)
    print(f"Saved raw/CAR/Laplacian + PSD comparison to {ref_path}")

    env_path = plot_high_gamma_diagnostic(channel, args.out_dir)
    print(f"Saved high-gamma diagnostic plot to {env_path}")


if __name__ == "__main__":
    main()

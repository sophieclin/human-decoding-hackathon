"""
Data Cleaning & Integrity Check

Loads a raw sEEG .mat file, computes trials-per-class per channel, and reports how
many channels survive a minimum-trial-count filter (the same standardization the
baseline decoding pipeline in docs/DecodingLogic_Hackathon.pdf applies: at least
`min_trials_per_class` trials for every stimulus class).

Usage:
    python3 scripts/data_integrity_check.py <path/to/dataset.mat> [min_trials_per_class]
"""

import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.seeg_io import drop_malformed_trials, load_seeg, viable_channels

MIN_TRIAL_SAMPLES = 500  # drop obviously truncated/corrupt trials before counting


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    data_path = sys.argv[1]
    min_trials_per_class = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    print(f"Loading {data_path}...")
    channels = load_seeg(data_path)
    print(f"Loaded {len(channels)} channels.")

    n_trials_before = sum(len(c.trials) for c in channels)
    channels = drop_malformed_trials(channels, min_samples=MIN_TRIAL_SAMPLES)
    n_trials_after = sum(len(c.trials) for c in channels)
    if n_trials_after < n_trials_before:
        print(
            f"Dropped {n_trials_before - n_trials_after} corrupt/truncated trial(s) "
            f"(< {MIN_TRIAL_SAMPLES} samples)."
        )

    kept = viable_channels(
        channels, min_trials_per_class=min_trials_per_class, require_all_classes=True
    )
    dropped = len(channels) - len(kept)

    print(
        f"\n{len(kept)} / {len(channels)} channels retained "
        f"(min {min_trials_per_class} trials/class, all classes present); "
        f"{dropped} dropped."
    )

    by_subdiv = Counter(c.prefrontal_subdiv for c in kept)
    print("\nRetained channels by Prefrontal_subdiv:")
    for subdiv, count in sorted(by_subdiv.items()):
        print(f"  {subdiv}: {count}")

    by_subject = Counter(c.sub for c in kept)
    print(f"\nRetained channels span {len(by_subject)} subjects.")


if __name__ == "__main__":
    main()

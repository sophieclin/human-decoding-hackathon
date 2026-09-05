"""
Loader for the BrainHack sEEG datasets, which are stored as MATLAB v7.3 (HDF5) files.

scipy.io.loadmat cannot read v7.3 files (confirmed against the real dataset: it raises
NotImplementedError). MATLAB's HDF5 writer stores a struct array as one HDF5 dataset per
field, each holding one element (or reference) per array index, so we open the file with
h5py and walk each field ourselves.

Two reference patterns show up in the real files:
  - scalar text fields (Sub, Task, Channel_Label, Hemisphere): one reference per channel
    pointing directly at a uint16 char-code dataset.
  - categorical-looking text fields (subRegion, Prefrontal_subdiv): one reference per
    channel pointing at a 1x1 object array which itself holds a reference to the
    char-code dataset (double-wrapped).
  - CorrectTrials: one reference per channel pointing at a group with Class/TrialData/
    Common/Laplacian sub-fields, each itself a reference per trial to a numeric dataset.

`_resolve` follows reference chains of any depth so the loader doesn't need to hardcode
how many levels of wrapping a given field uses.
"""

from dataclasses import dataclass, field, replace

import h5py
import numpy as np


@dataclass
class Trial:
    class_label: int
    trial_data: np.ndarray
    common: np.ndarray
    laplacian: np.ndarray


@dataclass
class ChannelRecord:
    sub: str
    task: str
    channel: float
    channel_label: str
    hemisphere: str
    subregion: str
    prefrontal_subdiv: str
    trials: list = field(default_factory=list)


def _resolve(f, value):
    """Follow HDF5 object references (of any nesting depth) down to real data."""
    if isinstance(value, h5py.Reference):
        return _resolve(f, f[value][()])
    arr = np.asarray(value)
    if arr.dtype == object:
        flat_resolved = [_resolve(f, v) for v in arr.flatten()]
        if len(flat_resolved) == 1:
            return flat_resolved[0]
        return np.array(flat_resolved, dtype=object).reshape(arr.shape)
    return arr


def _decode_string(f, value):
    resolved = _resolve(f, value)
    codes = np.asarray(resolved).flatten()
    return "".join(chr(int(c)) for c in codes)


def _decode_scalar(f, value):
    resolved = _resolve(f, value)
    return float(np.asarray(resolved).flatten()[0])


def load_seeg(path):
    """Load every channel of a v7.3 sEEG .mat file into a list of ChannelRecord."""
    channels = []
    with h5py.File(path, "r") as f:
        seeg = f["SEEG"]
        n_channels = seeg["Sub"].shape[0]

        for i in range(n_channels):
            trials = []
            ct = f[seeg["CorrectTrials"][i, 0]]
            n_trials = ct["Class"].shape[0] if ct["Class"].ndim > 0 else 0
            for j in range(n_trials):
                trials.append(
                    Trial(
                        class_label=int(_decode_scalar(f, ct["Class"][j, 0])),
                        trial_data=_resolve(f, ct["TrialData"][j, 0]).flatten(),
                        common=_resolve(f, ct["Common"][j, 0]).flatten(),
                        laplacian=_resolve(f, ct["Laplacian"][j, 0]).flatten(),
                    )
                )

            channels.append(
                ChannelRecord(
                    sub=_decode_string(f, seeg["Sub"][i, 0]),
                    task=_decode_string(f, seeg["Task"][i, 0]),
                    channel=_decode_scalar(f, seeg["Channel"][i, 0]),
                    channel_label=_decode_string(f, seeg["Channel_Label"][i, 0]),
                    hemisphere=_decode_string(f, seeg["Hemisphere"][i, 0]),
                    subregion=_decode_string(f, seeg["subRegion"][i, 0]),
                    prefrontal_subdiv=_decode_string(f, seeg["Prefrontal_subdiv"][i, 0]),
                    trials=trials,
                )
            )
    return channels


def viable_channels(
    channels, min_trials_per_class=3, require_all_classes=True, expected_classes=None
):
    """Keep channels with at least `min_trials_per_class` trials for every class.

    If require_all_classes is True, a channel must also cover every class in
    `expected_classes` (defaults to the set of classes observed across all input
    channels) to be considered viable.
    """
    if expected_classes is None:
        expected_classes = {t.class_label for c in channels for t in c.trials}

    kept = []
    for c in channels:
        counts = {}
        for t in c.trials:
            counts[t.class_label] = counts.get(t.class_label, 0) + 1

        if require_all_classes and not expected_classes.issubset(counts.keys()):
            continue
        if not counts or min(counts.values()) < min_trials_per_class:
            continue

        kept.append(c)
    return kept


def drop_malformed_trials(channels, min_samples):
    """Drop trials whose signal is shorter than `min_samples`.

    Real recordings can contain corrupted/truncated trials (observed in the wild: a
    2-sample "trial" instead of the expected ~3000+ samples) that would otherwise
    crash or silently corrupt downstream filtering and feature extraction.
    """
    cleaned = []
    for c in channels:
        good_trials = [
            t
            for t in c.trials
            if len(t.trial_data) >= min_samples
            and len(t.common) >= min_samples
            and len(t.laplacian) >= min_samples
        ]
        cleaned.append(replace(c, trials=good_trials))
    return cleaned

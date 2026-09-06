"""
Builds a small synthetic MATLAB v7.3 (HDF5) .mat file that replicates the exact
struct-of-arrays-of-references layout observed in the real sEEG datasets, so unit
tests can run in milliseconds instead of loading a multi-GB real file.
"""

import itertools

import h5py
import numpy as np

_name_counter = itertools.count()


def _unique_name():
    return f"obj{next(_name_counter)}"


def _write_char_dataset(f, group, text):
    codes = np.array([ord(c) for c in text], dtype=np.uint16).reshape(-1, 1)
    return group.create_dataset(_unique_name(), data=codes)


def _write_scalar_string_field(f, refs_group, seeg_group, field_name, values):
    """Mimics fields like Sub/Task/Channel_Label/Hemisphere: one ref per channel
    pointing directly at a char-code dataset."""
    ds = seeg_group.create_dataset(field_name, (len(values), 1), dtype=h5py.ref_dtype)
    for i, text in enumerate(values):
        char_ds = _write_char_dataset(f, refs_group, text)
        ds[i, 0] = char_ds.ref


def _write_double_wrapped_string_field(f, refs_group, seeg_group, field_name, values):
    """Mimics fields like subRegion/Prefrontal_subdiv: one ref per channel pointing
    at a 1x1 object array which itself holds a ref to the char-code dataset."""
    ds = seeg_group.create_dataset(field_name, (len(values), 1), dtype=h5py.ref_dtype)
    for i, text in enumerate(values):
        char_ds = _write_char_dataset(f, refs_group, text)
        inner = refs_group.create_dataset(_unique_name(), (1, 1), dtype=h5py.ref_dtype)
        inner[0, 0] = char_ds.ref
        ds[i, 0] = inner.ref


def _write_numeric_scalar_field(seeg_group, field_name, values):
    """Mimics the Channel field: value stored inline as a variable-length array of
    small numeric arrays (not a true HDF5 object reference)."""
    ds = seeg_group.create_dataset(
        field_name, (len(values), 1), dtype=h5py.vlen_dtype(np.float64)
    )
    for i, v in enumerate(values):
        ds[i, 0] = np.array([float(v)])


def build_fixture(path, channels):
    """
    channels: list of dicts with keys:
        sub, task, channel, channel_label, hemisphere, subregion, prefrontal_subdiv,
        trials: list of dicts with keys class_label, trial_data, common, laplacian
                (1-D numpy arrays of equal length within a channel)
    """
    with h5py.File(path, "w") as f:
        refs = f.create_group("#refs#")
        seeg = f.create_group("SEEG")

        n = len(channels)
        _write_scalar_string_field(f, refs, seeg, "Sub", [c["sub"] for c in channels])
        _write_scalar_string_field(f, refs, seeg, "Task", [c["task"] for c in channels])
        _write_scalar_string_field(
            f, refs, seeg, "Channel_Label", [c["channel_label"] for c in channels]
        )
        _write_scalar_string_field(
            f, refs, seeg, "Hemisphere", [c["hemisphere"] for c in channels]
        )
        _write_double_wrapped_string_field(
            f, refs, seeg, "subRegion", [c["subregion"] for c in channels]
        )
        _write_double_wrapped_string_field(
            f, refs, seeg, "Prefrontal_subdiv", [c["prefrontal_subdiv"] for c in channels]
        )
        _write_numeric_scalar_field(seeg, "Channel", [c["channel"] for c in channels])
        _write_scalar_string_field(
            f, refs, seeg, "Condition", [c.get("condition", "") for c in channels]
        )

        ct_ds = seeg.create_dataset("CorrectTrials", (n, 1), dtype=h5py.ref_dtype)
        for i, chan in enumerate(channels):
            trials = chan["trials"]
            ct_group = refs.create_group(_unique_name())
            n_tr = len(trials)

            class_ds = ct_group.create_dataset("Class", (n_tr, 1), dtype=h5py.ref_dtype)
            data_ds = ct_group.create_dataset("TrialData", (n_tr, 1), dtype=h5py.ref_dtype)
            common_ds = ct_group.create_dataset("Common", (n_tr, 1), dtype=h5py.ref_dtype)
            lap_ds = ct_group.create_dataset("Laplacian", (n_tr, 1), dtype=h5py.ref_dtype)

            for j, tr in enumerate(trials):
                cls_arr = refs.create_dataset(
                    _unique_name(), data=np.array([[float(tr["class_label"])]])
                )
                class_ds[j, 0] = cls_arr.ref

                td_arr = refs.create_dataset(
                    _unique_name(),
                    data=np.asarray(tr["trial_data"], dtype=np.float64).reshape(-1, 1),
                )
                data_ds[j, 0] = td_arr.ref

                cm_arr = refs.create_dataset(
                    _unique_name(),
                    data=np.asarray(tr["common"], dtype=np.float64).reshape(-1, 1),
                )
                common_ds[j, 0] = cm_arr.ref

                lp_arr = refs.create_dataset(
                    _unique_name(),
                    data=np.asarray(tr["laplacian"], dtype=np.float64).reshape(-1, 1),
                )
                lap_ds[j, 0] = lp_arr.ref

            ct_ds[i, 0] = ct_group.ref

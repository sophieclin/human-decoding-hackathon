# Implementation Plan: sEEG Ingestion, Preprocessing & Baseline Decoding Pipeline

## Overview

Build a Python pipeline over the downloaded sEEG `.mat` datasets that (1) loads and
integrity-checks channels, (2) bandpass-filters to high-gamma and produces a diagnostic
visualization, and (3) benchmarks simple linear classifiers against the linear-SVM approach
described in `docs/DecodingLogic_Hackathon.pdf`.

## Key Finding That Changes Scope

Both `.mat` files are **MATLAB v7.3 (HDF5)** format, confirmed by direct inspection:

```
data/Prefrontal_Delay_3_6_sec_Session.mat
  SEEG (group) -> fields: Channel, Channel_Label, Condition, CorrectTrials,
                          Hemisphere, Prefrontal_subdiv, Sub, Task, subRegion
  each field: dataset of shape (462, 1), dtype=object (HDF5 references)
  CorrectTrials[i,0] -> group with fields Class, Common, Laplacian, TrialData
    each: (n_trials_i, 1) object array of refs to (3072, 1) float64 arrays
  string fields (Sub, Prefrontal_subdiv, subRegion, ...) are refs to uint16 char codes,
  and subRegion/Prefrontal_subdiv are double-wrapped (cell-in-cell) refs.
```

`scipy.io.loadmat` **cannot read this** (`NotImplementedError: Please use HDF reader for
matlab v7.3 files`) — confirmed by running it against the real file. This means the existing
`scripts/visualization.py` and `scripts/extract_bandpass_features.py`, which both call
`sio.loadmat(...)`, are non-functional against the actual downloaded data and must be
rebuilt on an `h5py`-based loader. This plan treats that loader as the foundation everything
else depends on.

Verified against the real Spatial dataset (`Prefrontal_Delay_3_6_sec_Session.mat`):
- 462 channels total, spanning 33 unique subjects.
- 9 stimulus classes (labels 1-9) for the spatial task.
- Trial signal length is 3072 samples at the documented 512 Hz sampling rate (6 s), matching
  the "Delay 3-6 sec" filename and the `TIME_END=3072` already assumed in
  `extract_bandpass_features.py`.
- Trials-per-class varies wildly per channel (channel 0: 35 trials, uneven per class;
  channel 50: 15 trials, uneven; channel 100: 44 trials).
- Filtering to channels with **all 9 classes present AND >= 3 trials per class** (the paper's
  own standardization) keeps **202 of 462 channels** before trial-level corruption is
  accounted for. After Task 3 uncovered that individual trials can have corrupted/truncated
  signal fields (see below), the corrected count is **200 of 462**.
- **239 trials across the dataset have a corrupted/truncated `TrialData`, `Common`, or
  `Laplacian` field** (as short as 2 samples instead of 3072) — and a field can be corrupted
  independently of the others within the same trial (one observed trial had full-length
  `TrialData`/`Common` but a 2-sample `Laplacian`). Any code that touches trial signals must
  filter these out first (`utils.seeg_io.drop_malformed_trials`), or it will crash (bandpass
  filtering) or silently corrupt results.

## Architecture Decisions

- **New shared loader module** (`utils/seeg_io.py`) rather than patching each script
  independently — both preprocessing and classifier scripts need the same parsed
  representation, and having two ad hoc HDF5-dereferencing implementations would drift.
- **Plain Python objects (dataclasses / dicts of numpy arrays)** as the loader's output
  contract, not raw h5py handles — downstream code shouldn't need to know this came from
  HDF5, and the file must stay open only during loading (channel data is materialized into
  memory as numpy arrays because trials are read individually by reference anyway).
- **Synthetic HDF5 fixtures for unit tests**, not the real 2 GB file — tests must run in
  seconds. A small helper (`tests/fixtures.py`) builds a miniature v7.3-shaped `.mat` file
  (few channels, few trials, short signals) replicating the exact struct-of-arrays-of-refs
  layout discovered above, including the double-wrapped string cells.
- **Preprocessing follows the paper's parameters** where they're stated explicitly: 70-150 Hz
  Butterworth bandpass -> Hilbert amplitude envelope -> ~488 ms moving-average window (~250
  samples at 512 Hz) -> downsample by 10 (~19.5 ms steps). This replaces the ad hoc 10 Hz
  low-pass smoothing currently in `extract_bandpass_features.py`'s
  `bandpass_amplitude_envelope`, so the classifier benchmark in Task 4 is preprocessing the
  same way the paper's baseline SVM does.
- **Task 4 is a single-window benchmark, not a full time-resolved replication.** The paper's
  full pipeline (per-time-bin decoding, 10 channel-resampling permutations, SLOO with 15
  folds, cluster-based permutation stats) is a multi-day undertaking on its own and out of
  scope for "a fast script." Task 4 instead extracts one feature per channel per trial (mean
  high-gamma envelope during the cue window) and compares Linear SVM (L2/Ridge, one-vs-rest —
  the paper's own classifier) against L2 Logistic Regression and LDA, using
  leave-one-trial-out cross-validation, which is the low-sample-size-appropriate scheme the
  paper itself motivates (SLOO). This is flagged as an open question below in case the user
  wants the full time-resolved version later.

## Task List

### Phase 1: Foundation — Ingestion

- [ ] Task 1: HDF5-based SEEG loader (`utils/seeg_io.py`)
  - **Description:** Write `load_seeg(path) -> list[ChannelRecord]` that opens a v7.3
    `.mat` file with `h5py`, dereferences every field (including the double-wrapped
    string cells for `subRegion`/`Prefrontal_subdiv`), and returns one record per channel
    with: `sub`, `task`, `channel`, `channel_label`, `hemisphere`, `subregion`,
    `prefrontal_subdiv`, and `trials` (a list of `{class_label, trial_data, common,
    laplacian}` numpy arrays).
  - **Acceptance criteria:**
    - [ ] Loading the real Spatial file returns exactly 462 channel records.
    - [ ] Channel 0's scalar text fields (`sub`, `channel_label`, `hemisphere`), its
      double-wrapped text fields (`subregion`, `prefrontal_subdiv`), and its trial count
      all decode without raising and match the values a direct h5py inspection reports
      for that record (spot-checked manually; not reproduced here since they're specific
      values from the restricted dataset).
    - [ ] Works against a synthetic fixture with 2 fabricated channels / 3 trials each.
  - **Verification:**
    - [ ] `pytest tests/test_seeg_io.py -v` passes
    - [ ] Manual check: `python3 -c "from utils.seeg_io import load_seeg; ..."` against the
      real file completes without raising and reports 462 channels.
  - **Dependencies:** None
  - **Files likely touched:** `utils/seeg_io.py`, `tests/fixtures.py`, `tests/test_seeg_io.py`
  - **Estimated scope:** Medium (new module + fixture + tests)

### Checkpoint: After Task 1
- [ ] Loader tests pass on synthetic fixture
- [ ] Loader runs end-to-end against the real Spatial `.mat` and reports 462 channels

### Phase 2: Data Cleaning & Integrity Check

- [ ] Task 2: Viable-channel filtering (`utils/seeg_io.py` addition + `scripts/data_integrity_check.py`)
  - **Description:** Add `viable_channels(channels, min_trials_per_class=3,
    require_all_classes=True) -> list[ChannelRecord]` to `utils/seeg_io.py`, computing
    trials-per-class per channel and dropping any channel below threshold (or missing a
    class, if required). `scripts/data_integrity_check.py` runs this over a given `.mat`
    path and prints a summary (channels kept/dropped, per-subject/per-subdivision breakdown).
  - **Acceptance criteria:**
    - [ ] On synthetic fixture with known per-channel class counts, filtering returns
      exactly the expected subset.
    - [ ] Run against the real Spatial file with `min_trials_per_class=3,
      require_all_classes=True` returns **202 channels** (matches the manual count taken
      during planning).
  - **Verification:**
    - [ ] `pytest tests/test_seeg_io.py -v` passes (extended with filtering tests)
    - [ ] Manual check: `python3 scripts/data_integrity_check.py data/Prefrontal_Delay_3_6_sec_Session.mat` prints "202 / 462 channels retained"
  - **Dependencies:** Task 1
  - **Files likely touched:** `utils/seeg_io.py`, `scripts/data_integrity_check.py`, `tests/test_seeg_io.py`
  - **Estimated scope:** Small

### Checkpoint: After Task 2
- [ ] Integrity check script runs against real data and reports the expected 202-channel count
- [ ] Review with user before proceeding to signal processing

### Phase 3: Signal Preprocessing & Visualization

- [ ] Task 3: High-gamma envelope + diagnostic plot (`scripts/preprocess_and_plot.py`)
  - **Description:** Implement `moving_average_envelope(signal, fs, lo=70, hi=150,
    window_ms=488, downsample=10)` per the paper's parameters (bandpass -> Hilbert envelope
    -> boxcar moving average -> downsample), replacing the low-pass-smoothing approach in
    `extract_bandpass_features.py`. Build a script that picks one viable channel for one
    subject, computes the envelope for its trials, and plots high-gamma activity across
    the task epoch (fixation/cue/delay boundaries as vertical lines) — the diagnostic plot
    deliverable.
  - **Acceptance criteria:**
    - [ ] Envelope function output length matches `expected_len = ceil(3072/10)` for a 3072-
      sample input.
    - [ ] Envelope of a synthetic 100 Hz sine (inside the 70-150 Hz band) has near-constant
      (low-variance) amplitude after smoothing; a 10 Hz sine (outside the band) is
      attenuated close to zero — proves the bandpass is actually selecting high-gamma.
    - [ ] Script runs against the real Spatial file for one channel/subject and saves a PNG
      instead of only calling `plt.show()` (so it's verifiable non-interactively).
  - **Verification:**
    - [ ] `pytest tests/test_preprocess.py -v` passes
    - [ ] Manual check: generated PNG opened and visually shows a smooth, positive envelope
      distinct from the raw trace
  - **Dependencies:** Task 1 (uses the loader)
  - **Files likely touched:** `scripts/preprocess_and_plot.py`, `tests/test_preprocess.py`
  - **Estimated scope:** Medium

### Checkpoint: After Task 3
- [ ] Bandpass correctness proven on synthetic signals (not just "runs without error")
- [ ] Diagnostic plot generated from real data

### Phase 4: Baseline Classifier Benchmarking

- [ ] Task 4: Linear SVM vs. L2 Logistic Regression vs. LDA (`scripts/benchmark_classifiers.py`)
  - **Description:** For a chosen subdivision (Dorsal or Ventral) restricted to viable
    channels (Task 2), build one feature vector per trial (mean high-gamma envelope,
    from Task 3's function, during the cue window) stacked across that subdivision's
    channels, standardized to `min_trials_per_class` trials/class per the paper's approach.
    Evaluate Linear SVM (`sklearn.svm.LinearSVC`, L2), `LogisticRegression(penalty='l2')`,
    and `LinearDiscriminantAnalysis` with leave-one-trial-out cross-validation. Print/plot
    a comparison of mean accuracy (+ chance level baseline).
  - **Acceptance criteria:**
    - [ ] On a synthetic dataset with a clearly separable signal, all three classifiers
      score meaningfully above chance (sanity check that the eval loop isn't broken).
    - [ ] On a synthetic dataset of pure noise, all three classifiers hover near chance
      level (sanity check against silent overfitting/leakage).
    - [ ] Script runs against real data for at least one subdivision and reports three
      accuracy numbers plus the chance level.
  - **Verification:**
    - [ ] `pytest tests/test_benchmark.py -v` passes
    - [ ] Manual check: script output against real Spatial data for the Dorsal (or Ventral)
      subdivision
  - **Dependencies:** Tasks 1-3
  - **Files likely touched:** `scripts/benchmark_classifiers.py`, `tests/test_benchmark.py`
  - **Estimated scope:** Medium

### Checkpoint: Complete
- [ ] All four tasks' acceptance criteria met
- [ ] Full test suite passes: `pytest tests/ -v`
- [ ] Ready for user review

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Feature dataset (`Prefrontal_MNM_Session.mat`) still downloading | Low | All tasks are written against a configurable path; Spatial data is enough to build and verify against now |
| Full time-resolved SVM replication (Fig 1/2 in the PDF) is much larger than "a fast script" | Medium | Scoped Task 4 down to a single-window benchmark; flagged as an open question below |
| `Condition` field and `MNM` task structure may differ from `Spatial`/`CorrectTrials` shape | Medium | Loader (Task 1) is written generically off field names actually observed; if MNM's structure differs we'll extend, not rewrite |
| Real `.mat` file is 2 GB — slow to load repeatedly during dev | Low | Unit tests use synthetic fixtures; real-file checks are one-off manual verification steps, not part of the automated suite |

## Open Questions

- Task 4 benchmarks a single time-window feature per trial rather than the paper's full
  time-resolved (per-time-bin), permutation-averaged, cluster-tested pipeline. Confirm this
  reduced scope is acceptable, or say if the full replication is wanted as a follow-up.
- Which subdivision(s) should Task 3's diagnostic plot and Task 4's benchmark default to —
  Dorsal, Ventral, or both? Plan defaults to making it a script parameter with Ventral as the
  example (since channel 0 happens to be Ventral) but no hardcoded assumption otherwise.

import numpy as np
import pytest

from tests.fixtures import build_fixture
from utils.seeg_io import drop_malformed_trials, load_seeg, viable_channels


def make_trial(class_label, n_samples=8, offset=0.0):
    return {
        "class_label": class_label,
        "trial_data": np.arange(n_samples, dtype=np.float64) + offset,
        "common": np.arange(n_samples, dtype=np.float64) + offset + 100,
        "laplacian": np.arange(n_samples, dtype=np.float64) + offset + 200,
    }


@pytest.fixture
def two_channel_fixture(tmp_path):
    path = tmp_path / "fixture.mat"
    channels = [
        {
            "sub": "sub-01",
            "task": "Spatial",
            "channel": 12,
            "channel_label": "LFP3",
            "hemisphere": "LEFT",
            "subregion": "Parsorbitalis",
            "prefrontal_subdiv": "Ventral",
            "trials": [make_trial(1), make_trial(1), make_trial(2), make_trial(2)],
        },
        {
            "sub": "sub-02",
            "task": "Spatial",
            "channel": 7,
            "channel_label": "LFP9",
            "hemisphere": "RIGHT",
            "subregion": "Precentral",
            "prefrontal_subdiv": "Dorsal",
            "trials": [make_trial(1)],
        },
    ]
    build_fixture(str(path), channels)
    return str(path)


def test_load_seeg_returns_one_record_per_channel(two_channel_fixture):
    channels = load_seeg(two_channel_fixture)
    assert len(channels) == 2


def test_load_seeg_decodes_scalar_string_fields(two_channel_fixture):
    channels = load_seeg(two_channel_fixture)
    ch0 = channels[0]
    assert ch0.sub == "sub-01"
    assert ch0.task == "Spatial"
    assert ch0.channel_label == "LFP3"
    assert ch0.hemisphere == "LEFT"


def test_load_seeg_decodes_double_wrapped_string_fields(two_channel_fixture):
    channels = load_seeg(two_channel_fixture)
    ch0 = channels[0]
    assert ch0.subregion == "Parsorbitalis"
    assert ch0.prefrontal_subdiv == "Ventral"

    ch1 = channels[1]
    assert ch1.subregion == "Precentral"
    assert ch1.prefrontal_subdiv == "Dorsal"


def test_load_seeg_decodes_numeric_channel_id(two_channel_fixture):
    channels = load_seeg(two_channel_fixture)
    assert channels[0].channel == 12
    assert channels[1].channel == 7


def test_load_seeg_reads_trial_signals_and_labels(two_channel_fixture):
    channels = load_seeg(two_channel_fixture)
    ch0 = channels[0]
    assert len(ch0.trials) == 4
    assert [t.class_label for t in ch0.trials] == [1, 1, 2, 2]
    np.testing.assert_array_equal(ch0.trials[0].trial_data, np.arange(8, dtype=np.float64))
    np.testing.assert_array_equal(ch0.trials[0].common, np.arange(8, dtype=np.float64) + 100)
    np.testing.assert_array_equal(ch0.trials[0].laplacian, np.arange(8, dtype=np.float64) + 200)


def test_viable_channels_filters_by_min_trials_per_class(two_channel_fixture):
    channels = load_seeg(two_channel_fixture)
    # ch0 has 2 trials/class for classes {1,2}; ch1 has 1 trial for class {1} only.
    kept = viable_channels(channels, min_trials_per_class=2, require_all_classes=False)
    assert [c.channel_label for c in kept] == ["LFP3"]


def test_viable_channels_require_all_classes(two_channel_fixture):
    channels = load_seeg(two_channel_fixture)
    # Neither channel has classes covering {1,2,3}.
    kept = viable_channels(
        channels, min_trials_per_class=1, require_all_classes=True, expected_classes={1, 2, 3}
    )
    assert kept == []

    kept = viable_channels(
        channels, min_trials_per_class=1, require_all_classes=True, expected_classes={1, 2}
    )
    assert [c.channel_label for c in kept] == ["LFP3"]


def test_drop_malformed_trials_removes_short_trials(tmp_path):
    channels = load_seeg(
        _fixture_with_one_corrupt_trial(tmp_path)
    )
    ch0 = channels[0]
    assert len(ch0.trials) == 3  # 2 good trials + 1 corrupt

    cleaned = drop_malformed_trials(channels, min_samples=5)
    assert len(cleaned[0].trials) == 2
    assert all(len(t.trial_data) >= 5 for t in cleaned[0].trials)


def _fixture_with_one_corrupt_trial(tmp_path):
    path = tmp_path / "corrupt_fixture.mat"
    corrupt_trial = make_trial(1, n_samples=2)
    channels = [
        {
            "sub": "sub-01",
            "task": "Spatial",
            "channel": 1,
            "channel_label": "LFP1",
            "hemisphere": "LEFT",
            "subregion": "Region",
            "prefrontal_subdiv": "Ventral",
            "trials": [make_trial(1), make_trial(2), corrupt_trial],
        }
    ]
    build_fixture(str(path), channels)
    return str(path)


def test_drop_malformed_trials_checks_every_referencing_field(tmp_path):
    # A trial can have a full-length trial_data/common but a truncated Laplacian
    # field (observed in the real dataset) -- all three must be checked.
    path = tmp_path / "partial_corrupt_fixture.mat"
    good_trial = make_trial(1, n_samples=8)
    partially_corrupt_trial = make_trial(1, n_samples=8)
    partially_corrupt_trial["laplacian"] = partially_corrupt_trial["laplacian"][:2]

    channels = [
        {
            "sub": "sub-01",
            "task": "Spatial",
            "channel": 1,
            "channel_label": "LFP1",
            "hemisphere": "LEFT",
            "subregion": "Region",
            "prefrontal_subdiv": "Ventral",
            "trials": [good_trial, partially_corrupt_trial],
        }
    ]
    build_fixture(str(path), channels)

    channels = load_seeg(str(path))
    cleaned = drop_malformed_trials(channels, min_samples=5)
    assert len(cleaned[0].trials) == 1

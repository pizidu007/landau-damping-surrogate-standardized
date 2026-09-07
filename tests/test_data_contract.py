
from landau_surrogate.data.rollout_cache import OneStepPairDataset, load_case_sequence
from landau_surrogate.data.rollout_windows import RolloutWindowDataset
from landau_surrogate.data.snapshot_cache import SnapshotCacheDataset


def test_snapshot_reader(tiny_cache):
    dataset = SnapshotCacheDataset(tiny_cache, split="train")
    assert len(dataset) == 4
    sample = dataset[0]
    assert tuple(sample["field"].shape) == (8, 9)
    assert sample["case_id"] == "train"


def test_pair_reader_never_crosses_cases(tiny_cache):
    dataset = OneStepPairDataset(tiny_cache, split="train")
    assert len(dataset) == 3
    sample = dataset[-1]
    assert int(sample["time_index"]) == 2
    assert tuple(sample["current"].shape) == (8, 9)
    assert tuple(sample["target"].shape) == (8, 9)


def test_window_reader(tiny_cache):
    dataset = RolloutWindowDataset(tiny_cache, split="train", horizon=2)
    assert len(dataset) == 2
    assert tuple(dataset[0]["frames"].shape) == (3, 8, 9)


def test_load_sequence(tiny_cache):
    sequence = load_case_sequence(tiny_cache, 1)
    assert sequence["case_id"] == "val"
    assert sequence["field"].shape == (4, 8, 9)

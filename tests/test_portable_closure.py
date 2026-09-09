import hashlib
import json
import zipfile
import numpy as np
import pytest
import torch
from landau_surrogate.data.portable_closure import load_index, load_case, ClosureSequenceDataset, hp_gradient


@pytest.fixture()
def portable(tmp_path):
    records = []
    for index, split in enumerate(("train", "validation", "test")):
        state = np.zeros((161, 4, 16), np.float32)
        state[:, 0] = state[:, 2] = 1 + index * .1
        state[:, 1] = np.arange(161)[:, None] * .001
        arrays = {"state": state, "gradient": np.zeros((161,16), np.float32),
                  "time": np.arange(161) * .02, "dt": np.array(.02)}
        target = tmp_path / f"{split}.npz"
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_LZMA) as z:
            for name, value in arrays.items():
                with z.open(name + ".npy", "w") as file:
                    np.lib.format.write_array(file, value, allow_pickle=False)
        records.append({"case_id": split, "K": .4, "alpha": .01, "split": split,
                        "regime": "weak", "starter": split != "test", "path": target.name,
                        "shape": list(state.shape), "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
    (tmp_path / "manifest.json").write_text(json.dumps({"schema": "landau_pc_closure_v1", "cases": records}))
    return tmp_path


def test_lzma_arrays_roundtrip_and_digest_failure(portable):
    case = load_index(portable)[0]
    arrays = load_case(case, verify=True)
    assert arrays["state"].dtype == np.float32
    with case.path.open("ab") as f: f.write(b"changed")
    with pytest.raises(ValueError, match="SHA256"):
        load_case(case, verify=True)


def test_windows_preserve_case_and_causal_prefix_masks(portable):
    ds = ClosureSequenceDataset(portable, length=32, burn_in=16, stride=32)
    first = ds[0]
    assert first["case_id"] == "train" and first["state"].shape == (48,4,16)
    assert not first["valid"][:16].any() and first["valid"][16:].all()
    assert not first["loss_mask"][:16].any() and first["loss_mask"][16:].all()
    assert first["time"][16] == 0
    assert np.array_equal(first["state"][0].numpy(), first["state"][16].numpy())
    second = ds[1]
    assert second["valid"].all() and second["window_start_index"] == 32
    assert len(ds) == 5


def test_test_split_is_not_available_to_training_windows(portable):
    with pytest.raises(ValueError, match="Diagnostic test"):
        ClosureSequenceDataset(portable, split="test")
    assert len(load_index(portable, "test", "full")) == 1


def test_missing_extension_is_explicit_not_silently_a_smaller_split(portable):
    (portable / "train.npz").unlink()
    with pytest.raises(FileNotFoundError, match="Missing"):
        load_index(portable, "train", "full")


def test_case_path_cannot_escape_dataset(portable):
    p = portable / "manifest.json"
    manifest = json.loads(p.read_text()); manifest["cases"][0]["path"] = "../elsewhere.npz"
    p.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="escapes"):
        load_index(portable)


def test_hp_target_has_physical_wavenumber_sign_and_no_mean():
    x = torch.arange(128) * 2 * torch.pi / 128
    state = torch.zeros(2,3,4,128); state[:,:,0] = 1
    state[:,:,2] = 1 + .02 * torch.cos(2*x)
    k = torch.tensor([.3,.5])[:,None]
    actual = hp_gradient(state, k, 1.2)
    expected = (1.2 * .02 * 2 * k[:,:,None] * torch.cos(2*x)).expand(-1,3,-1)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-4)

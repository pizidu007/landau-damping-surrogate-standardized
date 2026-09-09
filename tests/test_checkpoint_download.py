"""Release downloads must verify bytes and preserve unrelated local assets."""
import hashlib
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture()
def downloader():
    script = Path(__file__).resolve().parents[1] / "scripts/download_checkpoints.py"
    spec = importlib.util.spec_from_file_location("checkpoint_download_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def asset(tmp_path, content=b"a small frozen checkpoint"):
    source = tmp_path / "remote.pt"
    source.write_bytes(content)
    return {"id": "test", "path": "models/test.pt", "size_bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(), "download_url": source.as_uri()}


def test_verified_download_and_repeat_without_network(downloader, tmp_path, monkeypatch):
    item = asset(tmp_path)
    root = tmp_path / "clone"
    assert downloader.download_asset(item, root) == "downloaded and verified"
    assert (root / item["path"]).read_bytes() == (tmp_path / "remote.pt").read_bytes()
    def forbidden(*args, **kwargs):
        raise AssertionError("A verified local checkpoint must not be downloaded again")
    monkeypatch.setattr(downloader, "urlopen", forbidden)
    assert downloader.download_asset(item, root) == "already verified"


@pytest.mark.parametrize("failure", ["hash", "short", "oversize"])
def test_invalid_download_is_not_installed(downloader, tmp_path, failure):
    item = asset(tmp_path)
    if failure == "hash":
        item["sha256"] = "0" * 64
    elif failure == "short":
        item["size_bytes"] += 1
    else:
        item["size_bytes"] -= 1
    root = tmp_path / "clone"
    with pytest.raises(ValueError):
        downloader.download_asset(item, root)
    assert not (root / item["path"]).exists()
    assert not list(root.rglob("*.part"))


def test_existing_different_checkpoint_is_preserved(downloader, tmp_path):
    item = asset(tmp_path)
    root = tmp_path / "clone"
    target = root / item["path"]
    target.parent.mkdir(parents=True)
    target.write_bytes(b"my independent experiment")
    with pytest.raises(FileExistsError):
        downloader.download_asset(item, root)
    assert target.read_bytes() == b"my independent experiment"


@pytest.mark.parametrize("relative", ["../outside.pt", "/absolute.pt"])
def test_manifest_cannot_escape_output_directory(downloader, tmp_path, relative):
    item = {**asset(tmp_path), "path": relative}
    with pytest.raises(ValueError, match="Unsafe"):
        downloader.download_asset(item, tmp_path / "clone")


def test_symlinked_model_directory_cannot_redirect_download(downloader, tmp_path):
    item = asset(tmp_path)
    shared = tmp_path / "shared"
    shared.mkdir()
    root = tmp_path / "clone"
    root.mkdir()
    (root / "models").symlink_to(shared, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        downloader.download_asset(item, root)
    assert not list(shared.iterdir())


def test_concurrent_new_file_is_never_overwritten(downloader, tmp_path, monkeypatch):
    item = asset(tmp_path)
    root = tmp_path / "clone"
    real_link = downloader.os.link
    def competing_writer(source, target):
        Path(target).write_bytes(b"concurrently created")
        real_link(source, target)
    monkeypatch.setattr(downloader.os, "link", competing_writer)
    with pytest.raises(FileExistsError):
        downloader.download_asset(item, root)
    assert (root / item["path"]).read_bytes() == b"concurrently created"
    assert not list(root.rglob("*.part"))

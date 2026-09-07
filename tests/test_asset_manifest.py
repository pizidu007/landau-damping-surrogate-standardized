import json
from pathlib import Path


def test_asset_manifest_has_unique_standard_paths():
    project_root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (project_root / "artifacts/manifests/assets.json").read_text(
            encoding="utf-8"
        )
    )
    assets = manifest["assets"]
    identifiers = [asset["id"] for asset in assets]
    paths = [asset["path"] for asset in assets]

    assert manifest["schema_version"] == 1
    assert len(assets) == 6
    assert len(identifiers) == len(set(identifiers))
    assert len(paths) == len(set(paths))
    external_paths = [
        asset["external_path"] for asset in assets if "external_path" in asset
    ]
    assert len(external_paths) == 3
    assert len(external_paths) == len(set(external_paths))
    assert all(not Path(path).is_absolute() for path in paths)
    assert len(manifest["source_references"]) == 1
    assert manifest["source_references"][0]["id"].endswith("9490dbf")
    assert (
        manifest["compatibility"][
            "conservative_cache_is_drop_in_checkpoint_cache"
        ]
        is False
    )

import argparse
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

from tools.asset_manager import (
    download_asset,
    ensure_cache,
    extract_dataset_asset,
    load_manifest,
    resolve_scene_root,
)


MIPNERF360_ARCHIVES = [
    {
        "name": "Mip-NeRF360 dataset part 1",
        "url": "https://storage.googleapis.com/gresearch/refraw360/360_v2.zip",
        "archive_filename": "360_v2.zip",
    },
    {
        "name": "Mip-NeRF360 dataset part 2",
        "url": "https://storage.googleapis.com/gresearch/refraw360/360_extra_scenes.zip",
        "archive_filename": "360_extra_scenes.zip",
    },
]

MIPNERF360_SCENES = {
    "bicycle": "images_4",
    "flowers": "images_4",
    "garden": "images_4",
    "stump": "images_4",
    "treehill": "images_4",
    "room": "images_2",
    "counter": "images_2",
    "kitchen": "images_2",
    "bonsai": "images_2",
}

TANDT_DB_SCENES = ["truck", "train", "drjohnson", "playroom"]


def download_file(url: str, destination: Path, force: bool = False) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        print(f"Reusing cached asset: {destination}")
        return destination

    part = destination.with_suffix(destination.suffix + ".part")
    part.unlink(missing_ok=True)
    with urllib.request.urlopen(url, timeout=60) as response, part.open("wb") as handle:
        total = int(response.headers.get("Content-Length") or 0)
        downloaded = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            downloaded += len(chunk)
            suffix = f" / {total} ({downloaded / total * 100:.1f}%)" if total else ""
            print(f"Downloading {destination.name}: {downloaded}{suffix}", end="\r")
    print()
    part.replace(destination)
    return destination


def safe_extract_zip(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            target = (destination / member.filename).resolve()
            if not (target == root or str(target).startswith(str(root) + "\\")):
                raise RuntimeError(f"Unsafe zip member: {member.filename}")
        zipped.extractall(destination)


def prepare_tandt_db(cache: Path, force: bool) -> dict:
    manifest = load_manifest()
    entry = manifest["datasets"][0]
    archive = download_asset(entry, cache, force=force)
    dataset_root = extract_dataset_asset(archive, cache, entry, manifest["manifest_version"], force=force)
    scenes = {scene: str(resolve_scene_root(dataset_root, entry, scene)) for scene in TANDT_DB_SCENES}
    return {"root": str(dataset_root), "scenes": scenes}


def prepare_mipnerf360(cache: Path, force: bool) -> dict:
    downloads = cache / "downloads"
    dataset_root = cache / "datasets" / "mipnerf360"
    if force and dataset_root.exists():
        shutil.rmtree(dataset_root)
    dataset_root.mkdir(parents=True, exist_ok=True)

    for asset in MIPNERF360_ARCHIVES:
        archive = download_file(asset["url"], downloads / asset["archive_filename"], force=force)
        safe_extract_zip(archive, dataset_root)

    scenes = {}
    missing = []
    for scene, image_dir in MIPNERF360_SCENES.items():
        scene_root = dataset_root / scene
        required = [scene_root / image_dir, scene_root / "sparse" / "0"]
        if all(path.exists() for path in required):
            scenes[scene] = {"root": str(scene_root), "images": image_dir}
        else:
            missing.append(scene)
    if missing:
        raise SystemExit("Mip-NeRF360 extraction is missing scenes: " + ", ".join(missing))
    return {"root": str(dataset_root), "scenes": scenes}


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and prepare the original 3DGS paper datasets.")
    parser.add_argument("--cache-dir", default="F:/Thesis/segs-demo-cache", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cache = ensure_cache(args.cache_dir)
    summary = {
        "mipnerf360": prepare_mipnerf360(cache, args.force),
        "tanks_and_temples_deep_blending": prepare_tandt_db(cache, args.force),
    }
    summary_path = cache / "datasets" / "original_paper_datasets.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

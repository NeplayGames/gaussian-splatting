import argparse
import gzip
import json
import math
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def load_annotations(category_root: Path) -> list[dict]:
    annotations_path = category_root / "frame_annotations.jgz"
    with gzip.open(annotations_path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def focal_ndc_to_pixels(focal_ndc: float, width: int, height: int) -> float:
    return float(focal_ndc) * min(width, height) / 2.0


def co3d_camera_to_blender_transform(viewpoint: dict) -> list[list[float]]:
    rotation = np.asarray(viewpoint["R"], dtype=np.float64)
    translation = np.asarray(viewpoint["T"], dtype=np.float64)

    world_to_camera = np.eye(4, dtype=np.float64)
    world_to_camera[:3, :3] = rotation
    world_to_camera[:3, 3] = translation

    camera_to_world = np.linalg.inv(world_to_camera)
    # The repo's Blender reader flips columns 1:3 before inverting. Store the
    # inverse flip here so the loaded camera matches the CO3D world-to-camera.
    camera_to_world[:3, 1:3] *= -1.0
    return camera_to_world.tolist()


def image_mean(path: Path) -> float:
    with Image.open(path) as image:
        return float(np.asarray(image.convert("RGB")).mean())


def convert_images(
    frames: list[dict],
    co3d_root: Path,
    output_root: Path,
    *,
    overwrite: bool,
) -> list[dict]:
    image_root = output_root / "images"
    image_root.mkdir(parents=True, exist_ok=True)
    converted = []

    for index, frame in enumerate(frames):
        source = co3d_root / frame["image"]["path"]
        stem = f"{index:05d}"
        target = image_root / f"{stem}.png"
        if overwrite or not target.exists():
            with Image.open(source) as image:
                image.convert("RGB").save(target)
        converted.append({**frame, "converted_stem": stem})

    return converted


def split_frames(frames: list[dict], test_stride: int) -> tuple[list[dict], list[dict]]:
    sorted_frames = sorted(frames, key=lambda item: item["frame_number"])
    test = [frame for index, frame in enumerate(sorted_frames) if index % test_stride == 0]
    train = [frame for index, frame in enumerate(sorted_frames) if index % test_stride != 0]
    if not train:
        train = sorted_frames
    if not test:
        test = sorted_frames[-1:]
    return train, test


def write_transforms(path: Path, frames: list[dict], camera_angle_x: float) -> None:
    payload = {
        "camera_angle_x": camera_angle_x,
        "frames": [
            {
                "file_path": f"images/{frame['converted_stem']}",
                "transform_matrix": co3d_camera_to_blender_transform(frame["viewpoint"]),
            }
            for frame in frames
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert one CO3D sequence to the repo's Blender-style loader format.")
    parser.add_argument("--co3d-root", required=True, type=Path)
    parser.add_argument("--category", default="banana")
    parser.add_argument("--sequence")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--test-stride", default=8, type=int)
    parser.add_argument("--min-image-mean", default=1.0, type=float, help="Skip source frames with mean RGB below this value.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite previously converted PNG files.")
    args = parser.parse_args()

    category_root = args.co3d_root / args.category
    annotations = load_annotations(category_root)
    sequence = args.sequence or sorted({frame["sequence_name"] for frame in annotations})[0]
    sequence_frames = sorted(
        [frame for frame in annotations if frame["sequence_name"] == sequence],
        key=lambda frame: frame["frame_number"],
    )
    if not sequence_frames:
        raise SystemExit(f"No frames found for sequence: {sequence}")

    usable_frames = [
        frame
        for frame in sequence_frames
        if image_mean(args.co3d_root / frame["image"]["path"]) >= args.min_image_mean
    ]
    if not usable_frames:
        raise SystemExit(f"No non-black frames found for sequence: {sequence}")

    args.output.mkdir(parents=True, exist_ok=True)
    converted = convert_images(usable_frames, args.co3d_root, args.output, overwrite=args.overwrite)
    train, test = split_frames(converted, max(args.test_stride, 1))

    width = int(converted[0]["image"]["size"][1])
    height = int(converted[0]["image"]["size"][0])
    focal_x = focal_ndc_to_pixels(converted[0]["viewpoint"]["focal_length"][0], width, height)
    camera_angle_x = 2.0 * math.atan(width / (2.0 * focal_x))

    write_transforms(args.output / "transforms_train.json", train, camera_angle_x)
    write_transforms(args.output / "transforms_test.json", test, camera_angle_x)
    metadata = {
        "source": str(args.co3d_root),
        "category": args.category,
        "sequence": sequence,
        "source_frames": len(sequence_frames),
        "skipped_dark_frames": len(sequence_frames) - len(converted),
        "min_image_mean": args.min_image_mean,
        "total_frames": len(converted),
        "train_frames": len(train),
        "test_frames": len(test),
        "camera_angle_x": camera_angle_x,
    }
    (args.output / "co3d_conversion.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

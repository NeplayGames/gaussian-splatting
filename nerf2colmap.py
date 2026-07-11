"""
prepare_blender_dataset.py
--------------------------
Automatically prepares Blender / HuggingFace NeRF datasets (with transforms_*.json)
for GraphDECO Gaussian Splatting.
"""

import os, json, shutil, argparse

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def restructure_dataset(src):
    print(f"🔧 Preparing dataset at: {src}")

    # --- Move train/test/val folders into images/
    img_root = os.path.join(src, "images")
    ensure_dir(img_root)
    for split in ["train", "test", "val"]:
        split_src = os.path.join(src, split)
        split_dst = os.path.join(img_root, split)
        if os.path.exists(split_src):
            ensure_dir(split_dst)
            for f in os.listdir(split_src):
                shutil.move(os.path.join(split_src, f), os.path.join(split_dst, f))
            shutil.rmtree(split_src)
            print(f"  Moved {split}/ → images/{split}/")

    # --- Rename JSON files if needed
    for f in os.listdir(src):
        if f.endswith(".json") and "text" in f:
            new_name = os.path.join(src, "transforms_test.json")
            os.rename(os.path.join(src, f), new_name)
            print(f"  Renamed {f} → transforms_test.json")

    # --- Ensure all JSONs reference correct file paths
    for name in ["transforms_train.json", "transforms_test.json", "transforms_val.json"]:
        path = os.path.join(src, name)
        if not os.path.exists(path): 
            continue
        with open(path, "r") as f: data = json.load(f)
        for frame in data.get("frames", []):
            fp = frame["file_path"]
            # Fix relative paths
            if not fp.startswith("images/"):
                frame["file_path"] = "images/" + fp
        with open(path, "w") as f: json.dump(data, f, indent=2)
        print(f"  Patched paths in {name}")

    # --- Create sparse structure placeholder
    ensure_dir(os.path.join(src, "sparse", "0"))
    print("  Created sparse/0/")

    print("✅ Dataset ready for Gaussian Splatting.")
    print(f"Next step:\n  python train.py -s {src} -m output/{os.path.basename(src)} --white_background")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Blender-style dataset for Gaussian Splatting")
    parser.add_argument("-s", "--source_path", required=True, type=str, help="Path to dataset root (e.g. chair/)")
    args = parser.parse_args()
    restructure_dataset(args.source_path)

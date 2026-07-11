import json, time
from pathlib import Path
IMG_EXT={'.png','.jpg','.jpeg'}
def validate_scene(root, dataset, scene, source_url='', checksum='', manifest_version=''):
    scene_dir=Path(root)/scene
    image_dir=next((p for p in [scene_dir/'images', scene_dir/'input'] if p.exists()), None)
    sparse_dir=scene_dir/'sparse'
    if not scene_dir.exists(): raise FileNotFoundError(f"Scene directory missing: {scene_dir}")
    if not image_dir: raise FileNotFoundError(f"Image directory missing in {scene_dir}")
    imgs=[p for p in image_dir.iterdir() if p.suffix.lower() in IMG_EXT]
    if not imgs: raise FileNotFoundError(f"No supported images in {image_dir}")
    if not sparse_dir.exists(): raise FileNotFoundError(f"COLMAP sparse directory missing: {sparse_dir}")
    return {"dataset":dataset,"scene":scene,"source_url":source_url,"archive_checksum":checksum,"manifest_version":manifest_version,"image_count":len(imgs),"image_directory":str(image_dir),"sparse_model_directory":str(sparse_dir),"validation_time":time.time()}

def write_validations(records, out): Path(out).write_text(json.dumps(records, indent=2))

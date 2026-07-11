import json, os, time
from pathlib import Path
IMG_EXT={'.png','.jpg','.jpeg','.JPG','.PNG'}

def _readable_nonempty(p): return p.exists() and p.is_file() and os.access(p, os.R_OK) and p.stat().st_size>0

def validate_scene_root(scene_dir, dataset, scene, source_url='', checksum='', archive_size=0, manifest_version=''):
    scene_dir=Path(scene_dir)
    if not scene_dir.is_dir(): raise FileNotFoundError(f"Scene directory missing: {scene_dir}")
    image_dir=next((p for p in (scene_dir/'images', scene_dir/'input') if p.is_dir()), None)
    if not image_dir: raise FileNotFoundError(f"Image directory missing in {scene_dir}")
    imgs=[p for p in image_dir.iterdir() if p.is_file() and p.suffix in IMG_EXT and os.access(p, os.R_OK)]
    if not imgs: raise FileNotFoundError(f"No supported readable images in {image_dir}")
    sparse_dir=next((p for p in (scene_dir/'sparse'/'0', scene_dir/'sparse') if p.is_dir()), None)
    if not sparse_dir: raise FileNotFoundError(f"COLMAP sparse directory missing: {scene_dir/'sparse'}")
    def pick(stem):
        for ext in ('.bin','.txt'):
            p=sparse_dir/f'{stem}{ext}'
            if _readable_nonempty(p): return p
        raise FileNotFoundError(f"COLMAP {stem} metadata missing or empty in {sparse_dir}")
    cameras=pick('cameras'); images_meta=pick('images'); points=pick('points3D')
    return {"dataset":dataset,"scene":scene,"scene_directory":str(scene_dir),"image_directory":str(image_dir),"image_count":len(imgs),"sparse_model_directory":str(sparse_dir),"cameras_file":str(cameras),"images_metadata_file":str(images_meta),"points3D_file":str(points),"source_url":source_url,"archive_checksum":checksum,"archive_size_bytes":archive_size,"manifest_version":manifest_version,"validation_time":time.time()}

def validate_scene(root, dataset, scene, source_url='', checksum='', manifest_version=''):
    return validate_scene_root(Path(root)/scene, dataset, scene, source_url, checksum, 0, manifest_version)

def write_validations(records, out):
    Path(out).parent.mkdir(parents=True, exist_ok=True); Path(out).write_text(json.dumps(records, indent=2))

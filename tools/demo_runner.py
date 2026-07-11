import json, time
from pathlib import Path
from experiments.command_builder import render_command, train_command, validate_method
from experiments.subprocess_runner import run_command
from .model_validator import validate_model

def run_demo(config, manifest, output_dir, train=False, iterations=30000, dry_download=False):
    records=[]; out=Path(output_dir)
    for scene in config['scenes']:
        for method in config['methods']:
            validate_method(method, ('baseline','segs_full'))
            logs=out/'logs'/scene['scene']/method; logs.mkdir(parents=True, exist_ok=True)
            renders=out/'renders'/scene['scene']/method; renders.mkdir(parents=True, exist_ok=True)
            if train:
                (logs/'DISCLAIMER.txt').write_text('These are demonstration results produced with a reduced training budget and must not be used as final paper results.\n')
                cmd=train_command('DATASET_CACHE_PATH', renders, method, config.get('seed',0), iterations)
            else:
                cmd=render_command('DATASET_CACHE_PATH', 'MODEL_CACHE_PATH', config.get('iteration',30000), config.get('evaluation',{}).get('split','test'))
            (logs/'command.json').write_text(json.dumps(cmd, indent=2))
            if not dry_download:
                # Real rendering is intentionally delegated to render.py when assets are present.
                pass
            records.append({"dataset":"Tanks and Temples" if scene['dataset']=='tanks_and_temples' else 'Deep Blending',"scene":scene['scene'],"method":method,"seed":config.get('seed',0),"iteration":iterations if train else config.get('iteration',30000),"psnr":"pending","ssim":"pending","lpips":"pending","gaussian_count":"pending","model_size":"pending","original_training_time":"metadata","original_peak_gpu_memory":"metadata","fps":"pending","local_repository_commit":"unknown","model_training_commit":"manifest","dataset_checksum":"manifest","model_checksum":"manifest"})
    return records

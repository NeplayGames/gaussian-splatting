import sys
SUPPORTED_DEMO_METHODS=("baseline","segs_full")
ALL_METHODS=("baseline","segs_edge_only","segs_saliency_only","segs_loss","segs_densification_only","segs_loss_and_densification","segs_curriculum","segs_full","constant_scale_control","shuffled_map_control")

def validate_method(method, allowed=ALL_METHODS):
    if method == "eggs": raise ValueError("Unsupported method 'eggs'. EGGS is not implemented in this repository; use 'segs_edge_only' for the edge-only method.")
    if method not in allowed: raise ValueError(f"Unsupported method '{method}'. Supported methods: {', '.join(allowed)}")

def render_command(source, model, iteration=30000, split="test", quiet=True, extra_flags=None):
    cmd=[sys.executable,"render.py","--iteration",str(iteration),"-s",str(source),"-m",str(model),"--eval"]
    if split == "test": cmd.append("--skip_train")
    elif split == "train": cmd.append("--skip_test")
    if quiet: cmd.append("--quiet")
    if extra_flags: cmd.extend(extra_flags)
    return cmd

def train_command(source, model, method="baseline", seed=0, iterations=1000):
    validate_method(method)
    cmd=[sys.executable,"train.py","-s",str(source),"-m",str(model),"--disable_viewer","--quiet","--eval","--iterations",str(iterations),"--test_iterations",str(iterations),"--save_iterations",str(iterations),"--method",method,"--seed",str(seed)]
    return cmd

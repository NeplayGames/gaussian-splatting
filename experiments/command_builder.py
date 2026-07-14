import sys
SUPPORTED_DEMO_METHODS=("baseline","eggs_saliency")
GUIDANCE_METHODS=("baseline","eggs_paper","eggs","saliency","eggs_saliency","eggs_norm","saliency_norm","eggs_saliency_norm")
ALL_METHODS=GUIDANCE_METHODS+("segs_edge_only","segs_saliency_only","segs_loss","segs_densification_only","segs_loss_and_densification","segs_curriculum","segs_full","constant_scale_control","shuffled_map_control")

def validate_method(method, allowed=ALL_METHODS):
    if method not in allowed: raise ValueError(f"Unsupported method '{method}'. Supported methods: {', '.join(allowed)}")

def render_command(source, model, iteration=30000, split="test", quiet=True, extra_flags=None):
    cmd=[sys.executable,"render.py","--iteration",str(iteration),"-s",str(source),"-m",str(model),"--eval"]
    if split == "test": cmd.append("--skip_train")
    elif split == "train": cmd.append("--skip_test")
    if quiet: cmd.append("--quiet")
    if extra_flags: cmd.extend(extra_flags)
    return cmd

def train_command(source, model, method="baseline", seed=0, iterations=1000, saliency_name=None, lambda_edge=None, lambda_saliency=None, eggs_beta=None, edge_p=None, extra_flags=None):
    validate_method(method)
    cmd=[sys.executable,"train.py","-s",str(source),"-m",str(model),"--disable_viewer","--quiet","--eval","--iterations",str(iterations),"--test_iterations",str(iterations),"--save_iterations",str(iterations),"--method",method,"--seed",str(seed)]
    if saliency_name:
        cmd.extend(["--saliency_name", str(saliency_name)])
    if lambda_edge is not None:
        cmd.extend(["--lambda_edge", str(lambda_edge)])
    if eggs_beta is not None:
        cmd.extend(["--eggs_beta", str(eggs_beta)])
    if edge_p is not None:
        cmd.extend(["--edge_p", str(edge_p)])
    if lambda_saliency is not None:
        cmd.extend(["--lambda_saliency", str(lambda_saliency)])
    if extra_flags:
        cmd.extend(extra_flags)
    return cmd

from tools.asset_manager import ensure_cache

def test_cache_dirs(tmp_path):
    root=ensure_cache(tmp_path)
    assert (root/'downloads').is_dir() and (root/'models').is_dir()

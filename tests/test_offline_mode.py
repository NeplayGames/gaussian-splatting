import pytest
from tools.asset_manager import download_asset, AssetError

def test_offline_missing_rejected(tmp_path):
    with pytest.raises(AssetError): download_asset({'archive_filename':'missing.bin','url':'http://invalid','sha256':'','size_bytes':-1}, tmp_path, offline=True)

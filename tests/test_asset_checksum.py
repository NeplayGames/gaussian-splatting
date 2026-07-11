import pytest
from tools.asset_manager import verify_file, AssetError, sha256_file

def test_invalid_checksum_rejected(tmp_path):
    p=tmp_path/'x'; p.write_text('abc')
    with pytest.raises(AssetError): verify_file(p, '0'*64)

def test_valid_checksum(tmp_path):
    p=tmp_path/'x'; p.write_text('abc')
    assert verify_file(p, sha256_file(p))

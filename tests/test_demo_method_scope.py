import pytest
from experiments.command_builder import validate_method

def test_eggs_rejected():
    with pytest.raises(ValueError): validate_method('eggs')

def test_demo_methods_allowed():
    validate_method('baseline', ('baseline','segs_full'))
    validate_method('segs_full', ('baseline','segs_full'))

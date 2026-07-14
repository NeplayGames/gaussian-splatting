import pytest
from experiments.command_builder import validate_method

def test_thesis_methods_allowed():
    for method in ('eggs_paper', 'eggs', 'saliency', 'eggs_saliency', 'eggs_norm', 'saliency_norm', 'eggs_saliency_norm'):
        validate_method(method)

def test_demo_methods_allowed():
    validate_method('baseline', ('baseline','eggs_saliency'))
    validate_method('eggs_saliency', ('baseline','eggs_saliency'))

def test_unknown_method_rejected():
    with pytest.raises(ValueError):
        validate_method('not_a_method')

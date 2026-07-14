from argparse import Namespace

import train


def configure(method):
    args = Namespace(method=method, weighting_control=False, shuffle_map_control=False)
    train._configure_segs_method(args)
    return args


def test_eggs_paper_is_edge_only_raw_loss():
    args = configure("eggs_paper")
    assert args.use_edge
    assert not args.use_saliency
    assert args.eggs_style_loss
    assert not args.adaptive_curriculum
    assert not args.segs_densification
    assert not args.weighting_control
    assert not args.shuffle_map_control


def test_normalized_controls_do_not_use_raw_eggs_style_loss():
    assert not configure("eggs_norm").eggs_style_loss
    assert not configure("saliency_norm").eggs_style_loss
    assert not configure("eggs_saliency_norm").eggs_style_loss

def test_package_imports():
    import rtgn_svdd

    assert rtgn_svdd.__all__ == [
        "EventSplits",
        "EventStream",
        "load_cic_csv",
        "load_config",
        "resolve_config",
        "split_chronologically",
        "write_config",
    ]


def test_core_runtime_dependencies_import():
    import numpy
    import torch
    import torch_geometric

    assert all(module is not None for module in (numpy, torch, torch_geometric))

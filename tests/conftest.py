"""
src.yolo.block_rules and src.yolo.fwd_hooks import
ultralytics.nn.modules.{block,conv,head} at module load time, to build
PROP_REGISTRY/FWD_HOOK_REGISTRY. This environment doesn't have
`ultralytics` installed (out of scope for these unit tests - see the
project's test plan), so importing either module would otherwise fail
before a single test could even run.

Stubs those three submodules with dummy torch.nn.Module subclasses under
the real class names the registries key on, so
src.yolo.block_rules/src.yolo.fwd_hooks can actually be imported and
their registry/selector/prop-function logic exercised directly. Only
kicks in if the real ultralytics isn't importable, so once it's
installed for real, these same tests exercise the real classes instead
of silently shadowing them with fakes.
"""

import sys
import types

import torch

try:
    import ultralytics.nn.modules.block  # noqa: F401
    import ultralytics.nn.modules.conv  # noqa: F401
    import ultralytics.nn.modules.head  # noqa: F401
except ImportError:
    _BLOCK_TYPES = (
        "C3",
        "C3k",
        "C2f",
        "C3k2",
        "C2PSA",
        "PSABlock",
        "Attention",
        "Conv",
        "Bottleneck",
        "SPPF",
        "DFL",
        "C2fCIB",
        "SCDown",
        "PSA",
        "CIB",
        "RepVGGDW",
        "RepCSP",
        "RepBottleneck",
        "RepConv",
        "AConv",
        "RepNCSPELAN4",
        "ELAN1",
        "SPPELAN",
    )
    _CONV_TYPES = ("DWConv", "Concat")
    _HEAD_TYPES = ("Detect", "v10Detect")

    block_mod = types.ModuleType("ultralytics.nn.modules.block")
    conv_mod = types.ModuleType("ultralytics.nn.modules.conv")
    head_mod = types.ModuleType("ultralytics.nn.modules.head")

    for _name in _BLOCK_TYPES:
        setattr(block_mod, _name, type(_name, (torch.nn.Module,), {}))
    for _name in _CONV_TYPES:
        setattr(conv_mod, _name, type(_name, (torch.nn.Module,), {}))
    for _name in _HEAD_TYPES:
        setattr(head_mod, _name, type(_name, (torch.nn.Module,), {}))

    modules_pkg = types.ModuleType("ultralytics.nn.modules")
    setattr(modules_pkg, "block", block_mod)
    setattr(modules_pkg, "conv", conv_mod)
    setattr(modules_pkg, "head", head_mod)

    sys.modules["ultralytics"] = types.ModuleType("ultralytics")
    sys.modules["ultralytics.nn"] = types.ModuleType("ultralytics.nn")
    sys.modules["ultralytics.nn.modules"] = modules_pkg
    sys.modules["ultralytics.nn.modules.block"] = block_mod
    sys.modules["ultralytics.nn.modules.conv"] = conv_mod
    sys.modules["ultralytics.nn.modules.head"] = head_mod

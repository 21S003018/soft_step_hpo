#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
FBNet model building blocks factory
"""

from . import ir_block


_PRIMITIVES = {
    "ir_k3": lambda in_channels, out_channels, stride, **kwargs: ir_block.IRFBlock(
        in_channels, out_channels, stride=stride, kernel_size=3, **kwargs
    ),
    "ir_k5": lambda in_channels, out_channels, stride, **kwargs: ir_block.IRFBlock(
        in_channels, out_channels, stride=stride, kernel_size=5, **kwargs
    ),
    "ir_k3_hs": lambda in_channels, out_channels, stride, **kwargs: ir_block.IRFBlock(
        in_channels,
        out_channels,
        stride=stride,
        kernel_size=3,
        relu_args="hswish",
        **kwargs,
    ),
    "ir_k5_hs": lambda in_channels, out_channels, stride, **kwargs: ir_block.IRFBlock(
        in_channels,
        out_channels,
        stride=stride,
        kernel_size=5,
        relu_args="hswish",
        **kwargs,
    ),
    "ir_k3_se": lambda in_channels, out_channels, stride, **kwargs: ir_block.IRFBlock(
        in_channels, out_channels, stride=stride, kernel_size=3, se_args="se", **kwargs
    ),
    "ir_k5_se": lambda in_channels, out_channels, stride, **kwargs: ir_block.IRFBlock(
        in_channels, out_channels, stride=stride, kernel_size=5, se_args="se", **kwargs
    ),
    "ir_k3_se_hs": lambda in_channels, out_channels, stride, **kwargs: ir_block.IRFBlock(  # noqa
        in_channels,
        out_channels,
        stride=stride,
        kernel_size=3,
        se_args="se_hsig",
        **kwargs,
    ),
    "ir_k5_se_hs": lambda in_channels, out_channels, stride, **kwargs: ir_block.IRFBlock(  # noqa
        in_channels,
        out_channels,
        stride=stride,
        kernel_size=5,
        se_args="se_hsig",
        **kwargs,
    ),
}


def get_block(block_name):
    try:
        return _PRIMITIVES[block_name]
    except:
        return

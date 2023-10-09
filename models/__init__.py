# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Customized for CryoEM protein particle Picking : CryoTransformer

from .detr import build


def build_model(args):
    return build(args)

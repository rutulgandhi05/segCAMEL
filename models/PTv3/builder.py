"""
Model Builder

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import copy
from models.PTv3.utils.registry import Registry

MODELS = Registry("models")
MODULES = Registry("modules")


def build_model(cfg):
    """Build models."""
    return MODELS.build(copy.deepcopy(cfg))

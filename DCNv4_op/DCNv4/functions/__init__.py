# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

# from .ms_flash_deform_attn_func import FlashMSDeformAttnFunction
from .flash_deform_attn_func import FlashDeformAttnFunction
from .dcnv4_func import DCNv4Function, dcnv4_core_pytorch
from collections.abc import Sequence
import math

import torch

from bitsandbytes.utils import _reverse_4bit_compress_format

from ..._ops import register_kernel
from ..utils import GAUDI_SW_VER


@register_kernel("bitsandbytes::quantize_4bit", "hpu")
def _(
    A: torch.Tensor, blocksize: int, quant_type: str, quant_storage: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    torch._check_is_size(blocksize)
    torch._check(quant_type == "nf4", lambda: f"quant_type must be nf4, got {quant_type}")
    torch._check(quant_storage == torch.uint8, lambda: f"quant_storage must be torch.uint8, got {quant_storage}")
    torch._check(
        A.dtype in [torch.bfloat16, torch.float32],
        lambda: f"Blockwise 4bit quantization on HPU only supports BF16/FP32 dtypes, but got {A.dtype}",
    )
    if A.dim() != 1:
        A = A.view(-1)
    packed, absmax = torch.ops.hpu.quantize_nf4(A, blocksize)
    return packed.view(-1, 1), absmax.to(torch.float32)


@register_kernel("bitsandbytes::dequantize_4bit", "hpu")
def _(
    A: torch.Tensor,
    absmax: torch.Tensor,
    blocksize: int,
    quant_type: str,
    shape: Sequence[int],
    dtype: torch.dtype,
) -> torch.Tensor:
    torch._check_is_size(blocksize)
    torch._check(quant_type == "nf4", lambda: f"quant_type must be nf4, got {quant_type}")
    torch._check(
        A.dtype in [torch.bfloat16, torch.uint8],
        lambda: f"quant_storage supports uint8 or bfloat16, but got {A.dtype}",
    )

    # Enable non uint8 dtype
    if A.dtype != torch.uint8:
        A = A.view(torch.uint8)

    A = A.reshape(-1)

    if GAUDI_SW_VER and (GAUDI_SW_VER.major < 1 or GAUDI_SW_VER.minor < 22):
        A = _reverse_4bit_compress_format(A)

    # HPU dequantization function for NF4 quantized tensors.
    out_dq = torch.ops.hpu.dequantize_nf4(
        A,
        absmax.to(dtype),
        blocksize,
        out_shape=(math.prod(shape),),
        out_dtype=dtype,
    )

    output = out_dq.reshape(shape)

    return output

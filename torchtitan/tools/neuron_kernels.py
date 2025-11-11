import torch
from torch_neuron import TorchNeuronNKIKernel
from torch_neuron.utils import get_logical_neuron_cores

import neuronxcc.nki.language as nl


# check for kernel in compiler package
try:
    from neuronxcc.nki._pre_prod_kernels.experimental.misc.apply_rope_nki import (
        nki_apply_rotary_embedding_with_cache
    )
except ImportError as e:
    nki_apply_rotary_embedding_with_cache = None


def get_rope_grid_size(vnc_degree, seq_len):
    grid_size = ()
    if (
        vnc_degree > 1
        and seq_len >= vnc_degree
        and seq_len % vnc_degree == 0
        and seq_len > nl.tile_size.pmax
    ):
        grid_size = (nl.nc(vnc_degree),)

    return grid_size


class RoPEFunctionNKI(torch.autograd.Function):
    """
    NKI-accelerated autograd function for Rotary Position Embedding (RoPE).
    """

    @staticmethod
    def forward(ctx, xq, xk, rope_cache):
        ctx.save_for_backward(rope_cache)
        xq_out = torch.empty(
            xq.shape, dtype=xq.dtype, device=xq.device, requires_grad=xq.requires_grad
        )
        xk_out = torch.empty(
            xk.shape, dtype=xk.dtype, device=xk.device, requires_grad=xk.requires_grad
        )
        seq_len = xq.shape[2]
        vnc_degree = int(get_logical_neuron_cores())
        grid = get_rope_grid_size(vnc_degree, seq_len)

        kernel = TorchNeuronNKIKernel(
            func=getattr(nki_apply_rotary_embedding_with_cache, "func", nki_apply_rotary_embedding_with_cache),
            grid=grid,
            kernel_return=True,
        )
        kernel(xq, xk, rope_cache, xq_out, xk_out, backward=False)
        return xq_out, xk_out

    @staticmethod
    def backward(ctx, grad_xq, grad_xk):
        rope_cache, = ctx.saved_tensors
        grad_xq_out = torch.empty(
            grad_xq.shape,
            dtype=grad_xq.dtype,
            device=grad_xq.device,
            requires_grad=grad_xq.requires_grad,
        )
        grad_xk_out = torch.empty(
            grad_xk.shape,
            dtype=grad_xk.dtype,
            device=grad_xk.device,
            requires_grad=grad_xk.requires_grad,
        )
        seq_len = grad_xq.shape[2]
        vnc_degree = int(get_logical_neuron_cores())
        grid = get_rope_grid_size(vnc_degree, seq_len)

        kernel = TorchNeuronNKIKernel(
            func=getattr(nki_apply_rotary_embedding_with_cache, "func", nki_apply_rotary_embedding_with_cache),
            grid=grid,
            kernel_return=True,
        )
        kernel(grad_xq, grad_xk, rope_cache, grad_xq_out, grad_xk_out, backward=True)
        return grad_xq_out, grad_xk_out, None, None


def apply_rope_kernel(original_rope, xq, xk, rope_cache):
    """
    xq: b, s, h_q, d
    xk: b, s, h_k, d
    rope_cache: s, d*2
    """
    
    try:
        # transpose to B, H, S, D
        xq_out, xk_out = RoPEFunctionNKI.apply(xq.transpose(1, 2), xk.transpose(1, 2), rope_cache)
        return xq_out.transpose(1, 2).contiguous(), xk_out.transpose(1, 2).contiguous()
    except:
        return original_rope(xq, xk, rope_cache)

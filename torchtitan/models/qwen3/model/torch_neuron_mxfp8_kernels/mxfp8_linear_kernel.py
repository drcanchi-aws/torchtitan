"""MXFP8 linear layer implementation using NKI kernels.

This module provides a PyTorch autograd function for MXFP8 linear operations.
"""

# No Neuron imports before setup_env_vars() is invoked (MXOCP env vars are set)
from ..common_utils.test_utils import setup_env_vars
setup_env_vars()

import torch
import torch_neuron
from torch_neuron import TorchNeuronNKIKernel
from ..mxfp8_kernels.matmul_tiled_mxfp8 import matmul_mxfp8_tiled
from ..common_utils.constants import KERNEL_RETURN, PLATFORM_TARGET
from typing import Optional, Tuple, Union


# Initialize the traced kernel
_matmul_mxfp8_tiled_traced = TorchNeuronNKIKernel(
    func=getattr(matmul_mxfp8_tiled, "func", matmul_mxfp8_tiled),
    platform_target=PLATFORM_TARGET,
    kernel_return=KERNEL_RETURN,
)

class _MXFP8LinearFunction(torch.autograd.Function):
    """PyTorch autograd function for MXFP8 linear operations"""
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        input_tensor: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of MXFP8 linear operation.
        
        Args:
            ctx: PyTorch autograd context for saving tensors
            input_tensor: Input tensor of shape (batch_size, in_features)
            weight: Weight tensor of shape (out_features, in_features)
            bias: Optional bias tensor of shape (out_features,)
            
        Returns:
            Output tensor of shape (batch_size, out_features)
            
        Raises:
            Exception: If tensor shapes are incompatible
        """
        if input_tensor.dim() != 2 or weight.dim() != 2:
            raise Exception(
                f"Expected 2D tensors, got input: {input_tensor.dim()}D, "
                f"weight: {weight.dim()}D"
            )
        
        if input_tensor.size(-1) != weight.size(-1):
            raise Exception(
                f"Input feature dimension {input_tensor.size(-1)} doesn't match "
                f"weight feature dimension {weight.size(-1)}"
            )
        
        ctx.save_for_backward(input_tensor, weight, bias)
        
        # Perform matrix multiplication using NKI kernel
        # output = input_tensor @ weight.T
        output = _matmul_mxfp8_tiled_traced(input_tensor.T, weight.T)
        
        if bias is not None:
            output = output + bias
            
        return output

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Backward pass of MXFP8 linear operation.
        
        Args:
            ctx: PyTorch autograd context containing saved tensors
            grad_output: Gradient of the output tensor
            
        Returns:
            Tuple of gradients for (input, weight, bias)
        """
        input_tensor, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            # grad_input = grad_output @ weight
            grad_input = _matmul_mxfp8_tiled_traced(grad_output.T, weight)
            
        if ctx.needs_input_grad[1]:
            # grad_weight = grad_output.T @ input_tensor
            grad_weight = _matmul_mxfp8_tiled_traced(grad_output, input_tensor)
            
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)
            
        return grad_input, grad_weight, grad_bias


def mxfp8_linear(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply MXFP8 linear transformation using NKI kernels.
    
    Performs a linear transformation: output = input @ weight.T + bias
    
    Args:
        input_tensor: Input tensor of shape (batch_size, in_features)
        weight: Weight tensor of shape (out_features, in_features)
        bias: Optional bias tensor of shape (out_features,)
        
    Returns:
        Output tensor of shape (batch_size, out_features)
        
    Example:
        >>> input_tensor = torch.randn(32, 128)
        >>> weight = torch.randn(256, 128)
        >>> bias = torch.randn(256)
        >>> output = mxfp8_linear(input_tensor, weight, bias)
        >>> output.shape
        torch.Size([32, 256])
    """
    return _MXFP8LinearFunction.apply(input_tensor, weight, bias)

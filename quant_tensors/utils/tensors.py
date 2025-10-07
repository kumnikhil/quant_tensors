import torch 
import numpy as np
from typing import Union, List, Tuple, Optional, Any
import warnings

TensorLike = Union[torch.Tensor, np.ndarray, List, Tuple, float, int]

def to_tensor(
        data:TensorLike, 
        dtype: torch.dtype = torch.float32, 
        device:Optional[Union[str, torch.device]]=None, 
        requires_grad:bool=False)->torch.Tensor:
    """Convert input data to a torch tensor."""
    if data is None: 
        raise ValueError("Cannot copnvert None to tensor.")
    if isinstance(data, torch.Tensor):
        tensor = data.to(dtype=dtype)
        if device is not None:
            tensor = tensor.to(device)
        if requires_grad and not tensor.requires_grad:
            tensor.requires_grad_(True)
        return tensor
    elif isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data).to(dtype=dtype)
        if device is not None:
            tensor = tensor.to(device)
        if requires_grad:
            tensor.requires_grad_(True)
        return tensor
    elif isinstance(data, (list, tuple)):
        tensor = torch.tensor(data, dtype=dtype)
        if device is not None:
            tensor = tensor.to(device)
        if requires_grad:
            tensor.requires_grad_(True)
        return tensor
    elif isinstance(data, (float, int)):
        tensor = torch.tensor(data, dtype=dtype)
        if device is not None:
            tensor = tensor.to(device)
        if requires_grad:
            tensor.requires_grad_(True)
        return tensor
    else:
        raise TypeError(f"Unsupported data type {type(data)} for tensor conversion, expected torch.Tensor, np.ndarray, list, tuple, float, or int.")
    
def ensure_tensor(
        data:TensorLike,
        name:str="data",
        dtype:torch.dtype=torch.float32,
        device:Optional[Union[str, torch.device]]=None,
        ndim:Optional[int]=None,
        shape:Optional[Tuple[Optional[int], ...]]=None
    )->torch.Tensor:
    # convert to tensor and validate properties
    
    tensor = to_tensor(data, dtype=dtype, device=device)
    # Check number of dimensions
    if ndim is not None and tensor.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D, got {tensor.ndim}D tensor with shape {tensor.shape}")
    
    # Check shape
    if shape is not None:
        if len(shape) != tensor.ndim:
            raise ValueError(f"{name} shape mismatch: expected {len(shape)}D, got {tensor.ndim}D")
        
        for i, (expected, actual) in enumerate(zip(shape, tensor.shape)):
            if expected is not None and expected != actual:
                raise ValueError(f"{name} shape mismatch at dimension {i}: "
                               f"expected {expected}, got {actual}")
    
    return tensor
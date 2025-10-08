"""
Historical Value at Risk (VaR) Module
======================================

GPU-accelerated historical VaR and related risk metrics using PyTorch.

This module provides functions for computing:
- Historical VaR 
- Expected Shortfall / CVaR
- Marginal VaR
- Component VaR
- Incremental VaR

All functions support GPU acceleration and flexible input formats.
"""
import torch 
import numpy as np
from typing import Union, Optional, Tuple, List
import warnings

from ..utils.tensors import to_tensor, TensorLike, clean_tensors_infnan

def historical_VaR(returns:TensorLike, confidence_level:float=0.95)->torch.Tensor:
    """
    Compute historical VaR at a specified confidence level.
    Args:
        returns: Tensor of shape (n_scenarios, n_assets) or (n_scenarios,) on the 
        confidence_level: Confidence level for VaR (e.g., 0.95 for 95% VaR)
    """
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")
    returns_tensor = clean_tensors_infnan(to_tensor(returns, dtype=torch.float32))
    if returns_tensor.numel()==0:
        raise ValueError("returns tensor is empty")
    alpha = 1.0 - confidence_level
    if returns_tensor.ndim == 1:
        var = torch.quantile(returns_tensor, alpha)
    else:
        var = torch.quantile(returns_tensor, alpha, dim=0)
    return torch.clamp(var, max=0.0)  # ensures VaR is non-positive

def historical_CVaR(returns:TensorLike, confidence_level:float=0.95)->torch.Tensor:
    """
    Compute historical Conditional VaR (CVaR) / Expected Shortfall at a specified confidence level.
    Args:
        returns: Tensor of shape (n_scenarios, n_assets) or (n_scenarios,)
        confidence_level: Confidence level for CVaR (e.g., 0.95 for 95% CVaR)
    """
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")
    returns_tensor = clean_tensors_infnan(to_tensor(returns, dtype=torch.float32))
    if returns_tensor.numel()==0:
        raise ValueError("returns tensor is empty")
    if returns_tensor.ndim == 1:
        var = historical_VaR(returns_tensor, confidence_level)
        tail_returns = returns_tensor[returns_tensor <= var]
        es = tail_returns.mean() if tail_returns.numel() > 0 else var
    else:
        var = historical_VaR(returns_tensor, confidence_level)
        mask = returns_tensor<=var.unsqueeze(0)
        masked_returns = torch.where(mask, returns_tensor, torch.tensor(0.0, device=returns_tensor.device))
        tail_counts = mask.sum(dim=0)
        tail_sums = masked_returns.sum(dim=0)
        es = torch.where(tail_counts > 0, tail_sums / tail_counts, var)
    return torch.clamp(es, max=0.0)  # ensures CVaR is non-positive

if __name__ == "__main__":
    # Simple test cases
    torch.manual_seed(42)
    log_returns = torch.randn((252,5)) * 0.2  # Simulated daily returns
    price_change = torch.exp(log_returns) - 1.0
    flat_prices = torch.ones((252,5)) * torch.tensor([100.0,200., 10000.,15.,75.])
    prices = flat_prices * (1.0 + price_change).cumprod(dim=0)
    returns = prices.diff(1,dim=0)

    var_95 = historical_VaR(returns, confidence_level=0.95)
    var_99 = historical_VaR(returns, confidence_level=0.99)
    cvar_95 = historical_CVaR(returns, confidence_level=0.95)
    cvar_99 = historical_CVaR(returns, confidence_level=0.99)
    print(f"95% Historical VaR: {np.round(var_95.numpy(),4)}")
    print(f"99% Historical VaR: {np.round(var_99.numpy(),4)}")
    print(f"95% Historical CVaR: {np.round(cvar_95.numpy(),4)}")
    print(f"99% Historical CVaR: {np.round(cvar_99.numpy(),4)}")
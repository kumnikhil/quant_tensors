"""
Module to collct all the risk metrics under quant_tensors.risk
"""

import torch
from ..risk.historical import historical_VaR, historical_CVaR
from ..utils.tensors import TensorLike

def VaR(returns: TensorLike, confidence_level: float = 0.95, method:str = "historical") -> torch.Tensor:
    if method.lower()=="historical":
        return historical_VaR(returns, confidence_level)
    else:
    
        raise ValueError(f"Unsupported VaR method {method}, currently only 'historical' is supported.")
    
def CVaR(returns: TensorLike, confidence_level: float = 0.95, method:str = "historical") -> torch.Tensor:
    if method.lower()=="historical":
        return historical_CVaR(returns, confidence_level)
    else:
        raise ValueError(f"Unsupported CVaR method {method}, currently only 'historical' is supported.")

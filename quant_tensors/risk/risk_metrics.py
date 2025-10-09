"""
Module to collct all the risk metrics under quant_tensors.risk
"""

import torch
from ..risk.historical import historical_VaR, historical_CVaR
from ..utils.tensors import TensorLike
class VaR(object):
    def __init__(self, method:str="historical", confidence_level:float=0.95):
        self.method = method
        self.confidence_level = confidence_level
    def _var_fn(self, pnl_vector: TensorLike) -> torch.Tensor:
        if self.method.lower()=="historical":
            return historical_VaR(pnl_vector, self.confidence_level)
        else:
            raise ValueError(f"Unsupported VaR method {self.method}, currently only 'historical' is supported.")
    def marginal_VaR(self, positions: TensorLike, pnl_vector: TensorLike) -> torch.Tensor:
        """
        Marginal VaR measures the senssitivity of the portfolio VaR to small changes in the position size of each asset.
        Args:

            pnl_vector: Tensor of shape (n_scenarios, n_assets) or (n_scenarios,)
        """
        pass
        return NotImplementedError("Marginal VaR is not implemented yet.")

    def __call__(self, pnl_vector: TensorLike) -> torch.Tensor:
        return self._var_fn(pnl_vector) 


class cVaR(object):
    def __init__(self, method:str="historical", confidence_level:float=0.95):
        self.method = method
        self.confidence_level = confidence_level
    def _cvar_fn(self, pnl_vector: TensorLike) -> torch.Tensor:
        if self.method.lower()=="historical":
            return historical_CVaR(pnl_vector, self.confidence_level)
        else:
            raise ValueError(f"Unsupported CVaR method {self.method}, currently only 'historical' is supported.")
    def __call__(self, pnl_vector: TensorLike) -> torch.Tensor:
        return self._cvar_fn(pnl_vector)
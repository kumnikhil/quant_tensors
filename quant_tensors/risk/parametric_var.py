""" module for implementing parametric VaR and CVaR calculations.
EWMA and GARCH models are supported for volatility estimation.
EWMA model implemented as per RiskMetrics 1996. and calibrate the decay factor lambda.
The model is assumed by generating the decay matrix and then multiplying with the returns to get weight adjusted volatility.

Generalized Garch model implementation, calibrated using MLE.
"""
import torch
from typing import Union, Optional, Tuple, List, Dict
import warnings

from ..utils.tensors import to_tensor, TensorLike, clean_tensors_infnan
class EWMA:
    @staticmethod
    def ewma_volatility(returns:TensorLike, lambda_decay:float=0.97, window:int=250,init_vol:Optional[Union[float, torch.Tensor]]=None, init_vol_window:int=20)->torch.Tensor:
        """
        Compute EWMA volatility for a series of returns.
        Args:
            returns: Tensor of shape (n_scenarios, n_assets) or (n_scenarios,)
            lambda: Decay factor for EWMA, typically between 0.94 and 0.97
            window: Lookback window size for volatility calculation
        """
        if not 0.0 < lambda_decay < 1.0:
            raise ValueError("lambda must be between 0 and 1")
        returns_tensor = clean_tensors_infnan(to_tensor(returns, dtype=torch.float32))
        if returns_tensor.numel()==0:
            raise ValueError("returns tensor is empty")
        if returns_tensor.ndim == 1:
            returns_tensor = returns_tensor.unsqueeze(1)  # Convert to (n_scenarios, 1)
        n_scenarios, n_assets = returns_tensor.shape
        if window > n_scenarios:
            warnings.warn("Window size is larger than number of scenarios. Using all available data.")
            window = n_scenarios
        if init_vol is None:
            init_var = returns_tensor[:init_vol_window].var(dim=0, unbiased=True).clamp(min=1e-6)
        elif isinstance(init_vol, torch.Tensor) and init_vol.shape[0]==n_assets:
            init_var = init_vol**2
        elif isinstance(init_vol, float):
            init_var = torch.tensor(init_vol**2, dtype=torch.float32).repeat(n_assets)
        else:
            raise RuntimeError("init_vol must be None, a float, or a tensor of shape (n_assets,)")
        r_squared = returns_tensor[-window:]**2
        decay_mat = torch.diag(torch.tensor([lambda_decay**(t-1)*(1.-lambda_decay) for t in range(window, 0,-1)]))
        vol = torch.sqrt((torch.matmul(decay_mat, r_squared)+init_var*(lambda_decay**window)).sum(dim=0))
        return vol  # shape (n_assets,)
    
    @staticmethod
    def calibrate_MLE(
        returns:TensorLike, 
        init_lambda:float=0.97,
        learn_rate:float=5e-3,
        lambda_range:Tuple[float, float] = (0.90, 0.9999), 
        max_iter:int=100, 
        init_window:int = 50)-> Dict[str, Union[float, torch.Tensor]]:
        """calibrate the decay factor lambda using gradient based optimization.
        Args:
            returns: Tensor of shape (n_scenarios, n_assets) or (n_scenarios,)
            init_lambda: Initial guess for lambda = 0.97
            lambda_range: Tuple specifying the range of lambda values to consider (0.90, 0.9999)
            max_iter: Maximum number of iterations for optimization
            init_window: Initial window size for volatility calculation
            n_grid: Number of grid points to search over in the specified lambda range
        """
        returns_tensor = clean_tensors_infnan(to_tensor(returns, dtype=torch.float32))
        device=returns.device
        if returns_tensor.numel()==0:
            raise ValueError("returns tensor is empty")
        if returns_tensor.ndim == 1:
            returns_tensor = returns_tensor.unsqueeze(1)
        # trainnable parameter constrianied to 0.90 , 0.9999
        lambda_logit = torch.tensor(
            torch.logit(torch.tensor((init_lambda - 0.90) / (0.99 - 0.9))),
            requires_grad=True,
            device=device
        )
        history = []
        optimizer = torch.optim.Adam([lambda_logit], lr=learn_rate)
        init_vol  = returns_tensor[:init_window].std(dim=0, unbiased=True).clamp(min=1e-6)
        for iter in range(max_iter):
            optimizer.zero_grad()
            logit_ = 0.85 +0.1*torch.sigmoid(lambda_logit) # constrain to (0.90, 0.99)
            vol  = EWMA.ewma_volatility(returns_tensor, lambda_decay=logit_, window=returns_tensor.shape[0], init_vol=init_vol)
            #NLL negative log likelihood
            nll = 0.5 * torch.sum(
                torch.log(2 * torch.pi * vol**2) + (returns_tensor/ vol)**2
            )
            nll.backward()
            optimizer.step()
            history.append(
                {
                    "iteration": iter,
                    "lambda": logit_.item(),
                    "nll": nll.item()
                }
            )
            if iter>10 and abs(history[-1]["nll"]-history[-2]["nll"])<1e-6:
                break
        optimal_lambda = 0.9 +0.1*torch.sigmoid(lambda_logit)
        return {
            "lambda": optimal_lambda.item(),
            "history": history,
            "converged": iter<max_iter-1
        }
if __name__ == "__main__":
    # Simple test cases
    torch.manual_seed(42)
    log_returns = torch.randn((252,5)) * 0.2  # Simulated daily returns
    price_change = torch.exp(log_returns) - 1.0
    flat_prices = torch.ones((252,5)) * torch.tensor([100.0,200., 10000.,15.,75.])
    prices = flat_prices * (1.0 + price_change).cumprod(dim=0)
    pnl_vec = prices.diff(1,dim=0)
    print("standard deviation:", pnl_vec.std(dim=0, unbiased=True))
    ewma_vol = EWMA.ewma_volatility(pnl_vec, lambda_decay=0.97, window=250)
    print("EWMA Volatility:", ewma_vol)

    calibration_result = EWMA.calibrate_MLE(pnl_vec, init_lambda=0.97, learn_rate=1e-2, max_iter=200)
    print("Calibrated Lambda:", calibration_result["lambda"])
    print("Converged:", calibration_result["converged"])
    print("Optimization History (last 5):", calibration_result["history"][-5:])
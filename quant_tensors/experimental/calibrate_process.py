import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from typing import Optional, Tuple, List, Dict, Any
import pandas as pd
import time 


class MomentBasedPINN(nn.Module):
    def __init__(self, input_dim:int=2, hidden_dim:int=64, output_dim:int=2):
        super(MomentBasedPINN, self).__init__()
        # network for drift \mu 
        self.drift = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1), 
            nn.Tanh() # ensures the drift is bounded
        )
        self. vol = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus() # ensures the volatility is positive
        )
        self.rf_rate = nn.Parameter(torch.tensor([0.04]), requires_grad=True)

    def forward(self, S:torch.Tensor, t:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = torch.cat([S, t], dim=1)
        mu = self.drift(inputs)
        sigma = self.vol(inputs)
        return mu, sigma
    
    def black76_price(self, S,K,T,r, sigma):
        d1 = (torch.log(S/K) + 0.5*sigma**2*T)/(sigma*torch.sqrt(T))
        d2 = d1 - sigma*torch.sqrt(T)
        call_price = torch.exp(-r*T) * (S*torch.distributions.Normal(0,1).cdf(d1) - K*torch.distributions.Normal(0,1).cdf(d2))
        return call_price
    
    def black_scholes_price(self, S,K,T,r,sigma):
        d1 = (torch.log(S/K) + (r + 0.5*sigma**2)*T)/(sigma*torch.sqrt(T))
        d2 = d1 - sigma*torch.sqrt(T)
        call_price = S*torch.distributions.Normal(0,1).cdf(d1) - K*torch.exp(-r*T)*torch.distributions.Normal(0,1).cdf(d2)
        return call_price
    
    def moment_conditions(self, S,t,dt=0.01):
        mu, sigma = self.forward(S,t)

        m1 = mu* dt
        m2 = sigma**2 * dt
        return m1, m2
    
    def option_price_pde(self, S, K, T):
        mu, sigma = self.forward(S, T)
        return self.black76_price(S, K, T, self.rf_rate, sigma)
        # return self.black_scholes_price(S, K, T, self.rf_rate, sigma)
    

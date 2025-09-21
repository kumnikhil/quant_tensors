import torch 
import torch.nn as nn
from dataclasses import dataclass
from ..utils.instruments import OptionData

class BlackScholes(nn.Module):
    def __init__(self):
        super(BlackScholes, self).__init__()
        self.normal_dist = torch.distributions.Normal(0,1)
    
    def d1(self, market_data:OptionData):
        return (torch.log(market_data.spot/market_data.strike) + market_data.rf_rate+ 0.5*market_data.volatility**2*market_data.tte) / (market_data.volatility*torch.sqrt(market_data.tte))
    
    def d2(self, market_data:OptionData):
        return self.d1(market_data) - market_data.volatility*torch.sqrt(market_data.tte)
    
    def call_price(self, market_data:OptionData):
        d1 = self.d1(market_data)
        d2 = self.d2(market_data)
        return market_data.spot*self.normal_dist.cdf(d1) - torch.exp(-market_data.rf_rate*market_data.tte) * market_data.strike*self.normal_dist.cdf(d2)
    
    def put_price(self, market_data:OptionData):
        d1 = self.d1(market_data)
        d2 = self.d2(market_data)
        return market_data.strike*torch.exp(-market_data.rf_rate*market_data.tte)*self.normal_dist.cdf(-d2) - market_data.spot*self.normal_dist.cdf(-d1)
    
    def forward(self,market_data:OptionData):
        call_prices = self.call_price(market_data=market_data)
        put_prices = self.put_price(market_data=market_data)
        return torch.where(market_data.is_call, call_prices, put_prices)
    
    def inverse_problem(self, market_data:OptionData, initial_volatility:float=0.2):
        if not isinstance(market_data.premium, torch.Tensor) :
            raise ValueError("Premium needed to compute the compute the implied volatility from the market price, and must of the same shape as strike tensor")
        def objective_function(volatility:torch.Tensor):
            market_data.volatility = volatility
            model_price = self.forward(market_data)
            return model_price - market_data.premium
        implied_vol = (torch.ones_like(market_data.strike)*initial_volatility).requires_grad_(True)
        optimizer = torch.optim.Adam([implied_vol], lr=0.05)
        for _ in range(100):
            optimizer.zero_grad()
            loss = torch.mean(objective_function(implied_vol)**2)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                implied_vol.clamp_(1e-6, 5.0) # ensure volatility stays within a reasonable range
        return_vol = implied_vol.detach()
        market_data.implied_volatility = return_vol
        return return_vol

    
if __name__ == "__main__":
    pricer = BlackScholes()
    market_data = OptionData(
        spot=torch.tensor([100.0, 100.0]),
        strike=torch.tensor([100.0, 90.0]),
        rf_rate=torch.tensor(0.05),
        volatility=torch.tensor([0.2, 0.25]),
        tte=torch.tensor([14/365, 44/365]),
        is_call=torch.tensor([True, False])
    )
    price = pricer.forward(market_data)
    market_data.premium = price
    print(f"Option Price: {price.numpy()}")
    implied_vol = pricer.inverse_problem(market_data, initial_volatility=0.2)
    print(f"Implied Volatility: {implied_vol.numpy()}")
    error = torch.abs(implied_vol - market_data.implied_volatility)
    print(f"Implied Volatility Error: {error.numpy()}")
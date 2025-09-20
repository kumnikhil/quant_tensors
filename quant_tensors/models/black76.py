import torch 
import torch.nn as nn

from ..utils.instruments import OptionData

class Black76(nn.Module):
    def __init__(self):
        super(Black76, self).__init__()
        self.normal_dist = torch.distributions.Normal(0,1)
    def d1(self, market_data:OptionData):
        return (torch.log(market_data.spot/market_data.strike) + 0.5*market_data.volatility**2*market_data.tte) / (market_data.volatility*torch.sqrt(market_data.tte))
    def d2(self, market_data:OptionData):
        return self.d1(market_data) - market_data.volatility*torch.sqrt(market_data.tte)
    def call_price(self, market_data:OptionData):
        d1 = self.d1(market_data=market_data)
        d2 = self.d2(market_data=market_data)
        return torch.exp(-market_data.rf_rate*market_data.tte) * (market_data.spot*self.normal_dist.cdf(d1) - market_data.strike*self.normal_dist.cdf(d2))
    def put_price(self, market_data:OptionData):
        d1 = self.d1(market_data)
        d2 = self.d2(market_data)
        return torch.exp(-market_data.rf_rate*market_data.tte) * (market_data.strike*self.normal_dist.cdf(-d2) - market_data.spot*self.normal_dist.cdf(-d1))
    def forward(self, market_data:OptionData):
        call_prices = self.call_price(market_data=market_data)
        put_prices = self.put_price(market_data=market_data)
        return torch.where(market_data.is_call, call_prices, put_prices)
    
if __name__ == "__main__":
    pricer = Black76()
    market_data = OptionData(
        spot=torch.tensor(100.0),
        strike=torch.tensor(100.0),
        rf_rate=torch.tensor(0.05),
        volatility=torch.tensor(0.2),
        tte=torch.tensor(1.0),
        is_call=torch.tensor(True)
    )
    price = pricer.forward(market_data)
    print(f"Option Price: {price.item()}")
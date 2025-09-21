import torch
from dataclasses import dataclass
from typing import Union, Optional

@dataclass()
class OptionData:
    spot: torch.Tensor
    strike: torch.Tensor
    rf_rate: Union[float, torch.Tensor]
    tte:torch.Tensor
    volatility: torch.Tensor
    is_call: torch.Tensor
    premium: Optional[torch.Tensor] = None
    implied_volatility: Optional[torch.Tensor] = None
    def __post_init__(self):
        if self.spot.shape != self.strike.shape or self.is_call.shape != self.spot.shape:
            raise ValueError("Inconsistent tensor shapes, tensors for spot,m strike and is_call should be the same")
        self.spot = self.spot.to(torch.float32)
        self.strike = self.strike.to(torch.float32)
        self.tte = self.tte.to(torch.float32)
        self.volatility = self.volatility.to(torch.float32)
        self.premium = self.premium.to(torch.float32) if isinstance(self.premium, torch.Tensor) else None
        self.implied_volatility = self.implied_volatility.to(torch.float32) if isinstance(self.implied_volatility, torch.Tensor) else None
        self.is_call = self.is_call.to(torch.bool)
        if isinstance(self.rf_rate, float):
            self.rf_rate = torch.tensor(self.rf_rate, dtype=torch.float32)*torch.ones_like(self.strike)
        else:
            self.rf_rate = self.rf_rate.expand(self.strike.shape).to(torch.float32)
        
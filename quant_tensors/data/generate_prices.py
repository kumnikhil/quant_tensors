import torch
from ..utils.instruments import OptionData

def generate_options_premium(pricer, spot:float, r_true:float, t_max:float=10.0, pts_spot:int= 50)->OptionData:
    spots = torch.linspace(0.5*spot, 1.5*spot, pts_spot)
    times = torch.linspace(1/365, t_max, pts_spot)
    S_grid, T_grid = torch.meshgrid(spots, times, indexing='ij')
    S_flat = S_grid.flatten().unsqueeze(1)
    T_flat = T_grid.flatten().unsqueeze(1)
    r_flat = r_true*torch.ones_like(S_flat)
    
    sigma_true = 0.2 + 0.1*torch.exp(-T_flat)*(S_flat/spot - 1)**2 ## Heston like volatility surface
    # genrating srtike around ATM 
    K_flat = S_flat*(0.8 + 0.2*torch.rand_like(S_flat)) 
    option_data = OptionData(
        spot=S_flat,
        strike=K_flat,
        rf_rate=r_flat,
        tte=T_flat,
        volatility=sigma_true,
        is_call=(torch.rand_like(S_flat) > 0.5)
    )
    premium  = pricer.forward(option_data)
    noise = 0.01*torch.randn_like(premium)
    option_data.premium = premium + noise
    return option_data
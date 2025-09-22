from pyexpat import model
import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from typing import Optional, Tuple, List, Dict, Any
import pandas as pd
import time 
from tqdm import tqdm
from ..utils.instruments import OptionData
from ..data.generate_prices import generate_options_premium
from ..models.black76 import Black76
from ..models.bsm import BlackScholes

class MomentBasedPINN(nn.Module):
    def __init__(self, input_dim:int=2, hidden_dim:int=64, initial_r:float= 0.04, output_dim:int=2, pricer_type:str='black76'):
        super(MomentBasedPINN, self).__init__()
        # network for drift \mu 
        if str(pricer_type).lower() == 'black76':
            self.pricer = Black76()
        elif str(pricer_type).lower() == 'bsm':
            self.pricer = BlackScholes()
        else:
            raise ValueError("Unsupported pricer type, choose either 'black76' or 'bsm'")
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
        self.r = nn.Parameter(torch.tensor([initial_r]), requires_grad=True)

    def forward(self, S:torch.Tensor, t:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = torch.cat([S, t], dim=1)
        mu = self.drift(inputs)
        sigma = self.vol(inputs)
        return mu, sigma
    
    # def black76_price(self, S,K,T,r, sigma):
    #     d1 = (torch.log(S/K) + 0.5*sigma**2*T)/(sigma*torch.sqrt(T))
    #     d2 = d1 - sigma*torch.sqrt(T)
    #     call_price = torch.exp(-r*T) * (S*torch.distributions.Normal(0,1).cdf(d1) - K*torch.distributions.Normal(0,1).cdf(d2))
    #     return call_price
    
    # def black_scholes_price(self, S,K,T,r,sigma):
    #     d1 = (torch.log(S/K) + (r + 0.5*sigma**2)*T)/(sigma*torch.sqrt(T))
    #     d2 = d1 - sigma*torch.sqrt(T)
    #     call_price = S*torch.distributions.Normal(0,1).cdf(d1) - K*torch.exp(-r*T)*torch.distributions.Normal(0,1).cdf(d2)
    #     return call_price
    
    def moment_conditions(self, S,t,dt=0.01):
        mu, sigma = self.forward(S,t)

        m1 = mu* dt
        m2 = sigma**2 * dt
        return m1, m2
    
    def option_price_pde(self, S, K, T):
        mu, sigma = self.forward(S, T)
        return self.black76_price(S, K, T, self.rf_rate, sigma)
        # return self.black_scholes_price(S, K, T, self.rf_rate, sigma)
    
    def get_risk_free_rate(self) -> float:
        """Get current risk-free rate value."""
        return self.r.item()
    
    
def train_model(n_epochs:int=2000,rf_rate:float=0.04,initial_vol =0.2, lambda_moment:float=1.0,lambda_price:float=10.0, t_max:float=2.0,pts_spot:int=50,pricer_type:str='black76'):
    model = MomentBasedPINN(pricer_type='black76', initial_r=rf_rate)
    data = generate_options_premium(pricer=model.pricer, spot=100.0, r_true=rf_rate,t_max=t_max , pts_spot=pts_spot)
    dt = t_max/pts_spot
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_history = []
     
    for epoch in tqdm(range(n_epochs), desc="Training Epochs"):
        optimizer.zero_grad()
        batch_size = 256
        idx = torch.randint(0, len(data.spot), (batch_size,))
        batch_data = OptionData(
            spot=data.spot[idx],
            strike=data.strike[idx],
            rf_rate=data.rf_rate[idx],
            tte=data.tte[idx],
            volatility=data.volatility[idx],
            is_call=data.is_call[idx],
            premium=data.premium[idx]
        )
        predicted_prices = model.pricer.forward(batch_data)
        price_loss = torch.mean((predicted_prices - batch_data.premium)**2) 
        m1, m2 = model.moment_conditions(batch_data.spot, batch_data.tte,dt)
        target_m1 = model.r * dt 
        target_m2 = initial_vol**2 * dt  
        # Moment constraints (risk-neutral drift should be r)
        moment_loss = torch.mean((m1 - target_m1)**2) + torch.mean((m2 - target_m2)**2)

        # Total loss
        total_loss = lambda_price * price_loss + lambda_moment * moment_loss
        total_loss.backward()
        optimizer.step()
        loss_history.append(total_loss.item())
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss = {total_loss.item():.6f}, "
                  f"Price Loss = {price_loss.item():.6f}, "
                  f"Moment Loss = {moment_loss.item():.6f}")
    return model, loss_history, data

def visualize_results(model:MomentBasedPINN, loss_history:List[float], data:OptionData):
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(loss_history)
    plt.yscale('log')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss History")
    
    plt.subplot(1,2,2)
    with torch.no_grad():
        predicted_prices = model.pricer.forward(data)
    plt.scatter(data.premium.numpy(), predicted_prices.numpy(), alpha=0.6)
    plt.plot([data.premium.min(), data.premium.max()], [data.premium.min(), data.premium.max()], 'r--')
    plt.xlabel("Market Premium")
    plt.ylabel("Model Predicted Price")
    plt.title("Model vs Market Prices")
    plt.tight_layout()
    plt.show()

def visualize_model(model:MomentBasedPINN, data:OptionData):
    model.eval()
    with torch.no_grad():
        # Create visualization grid
        S_vis = torch.linspace(80, 120, 30).unsqueeze(1)
        T_vis = torch.linspace(0.1, 1.0, 20).unsqueeze(1)
        
        S_grid, T_grid = torch.meshgrid(S_vis.squeeze(), T_vis.squeeze(), indexing='ij')
        S_flat = S_grid.flatten().unsqueeze(1)
        T_flat = T_grid.flatten().unsqueeze(1)
        
        # Get learned parameters
        mu_learned, sigma_learned = model.forward(S_flat, T_flat)    
        # Reshape for plotting
        sigma_surface = sigma_learned.reshape(S_grid.shape)
        # Plot volatility surface
        fig = plt.figure(figsize=(15, 5))
        # Learned volatility surface
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.plot_surface(S_grid.numpy(), T_grid.numpy(), sigma_surface.numpy(), 
                        cmap='viridis', alpha=0.8)
        ax1.set_xlabel('Spot Price')
        ax1.set_ylabel('Time to Maturity')
        ax1.set_zlabel('Volatility')
        ax1.set_title('Learned Volatility Surface')
        # True volatility surface
        ax2 = fig.add_subplot(132, projection='3d')
        sigma_true_vis = 0.2 + 0.1 * torch.exp(-T_flat) * (S_flat/100 - 1)**2
        sigma_true_surface = sigma_true_vis.reshape(S_grid.shape)
        ax2.plot_surface(S_grid.numpy(), T_grid.numpy(), sigma_true_surface.numpy(), 
                        cmap='viridis', alpha=0.8)
        ax2.set_xlabel('Spot Price')
        ax2.set_ylabel('Time to Maturity')
        ax2.set_zlabel('Volatility')
        ax2.set_title('True Volatility Surface')
        
        # Error plot
        ax3 = fig.add_subplot(133)
        error = torch.abs(sigma_surface - sigma_true_surface)
        im = ax3.contourf(S_grid.numpy(), T_grid.numpy(), error.numpy(), levels=20)
        ax3.set_xlabel('Spot Price')
        ax3.set_ylabel('Time to Maturity')
        ax3.set_title('Absolute Error')
        plt.colorbar(im, ax=ax3)
        
        plt.tight_layout()
        plt.show()
        
        print(f"Learned risk-free rate: {model.r.item():.4f}")
        print(f"Mean absolute error in volatility: {torch.mean(error).item():.4f}")

# Example usage
if __name__ == "__main__":
    print("Training Moment-Based PINN for Option Pricing...")
    
    model, losses, data = train_model()
    
    print("\nVisualizing results...")
    visualize_results(model, losses, data)
    visualize_model(model, data)

    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.yscale('log')
    plt.grid(True)
    plt.show()
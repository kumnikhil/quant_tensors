# quant_tensors

**GPU-Accelerated Quantitative Finance Library for Commodities Risk & Pricing**

A high-performance Python library built on PyTorch for market risk analytics, derivatives pricing, and risk management. Designed to address the needs of risk quants who need to leverage GPU acceleration for real-time risk calculations and portfolio analytics.

---

## 🎯 Overview

**quant_tensors** enables quantitative analysts to:
- **Price derivatives** 10-100x faster using GPU acceleration
- **Compute Greeks** automatically via PyTorch's autograd (no finite differences!)
- **Run Monte Carlo** simulations with millions of paths in seconds
- **Calculate VaR** across thousands of positions in real-time
- **Calibrate models** using GPU-optimized optimization algorithms

### Why quant_tensors?

Traditional quantitative libraries are CPU-bound and rely on slow finite-difference methods for Greeks. **quant_tensors** leverages PyTorch to:

✅ **Automatic Differentiation** - Compute exact Greeks without numerical approximations  
✅ **GPU Acceleration** - 10-100x speedup over CPU-based solutions  
✅ **Batch Processing** - Price thousands of instruments simultaneously  
✅ **End-to-End Differentiable** - Backpropagate through entire pricing pipelines  
✅ **Production-Ready** - Type-safe, validated, and tested  

---

## ✨ Features

### 📊 Volatility Models
- **SVI (Stochastic Volatility Inspired)** - Arbitrage-free volatility surface parameterization
- **SABR** - to be added
- **Local Volatility** - Dupire's local volatility with GPU calibration
- **Implied Volatility** - Fast Newton-Raphson on GPU

### 💰 Pricing Models
- **Black-76** - Industry standard for commodity options
  - European calls/puts
  - American options (Monte Carlo)
  - Asian options (arithmetic/geometric)
  - Barrier options
  - Spread options
- **Monte Carlo Engine**
  - Geometric Brownian Motion (GBM)
  - Mean reverting Ornstein-Uhlenbeck (OU) process
  - Jump diffusion (Merton, Kou)
  - Stochastic volatility (Heston)
  - Multi-asset with correlation
  - Variance reduction (antithetic, control variates)
  
### 📈 Risk Analytics
- **Value at Risk (VaR)**
  - Historical simulation
  - Filtered historical simulation
  - Monte Carlo VaR
  - Parametric VaR
- **Expected Shortfall (CVaR)**
- **Stress Testing** - Scenario analysis with GPU parallelization
- **Marginal VaR** - Component and incremental risk
- **Stressed VaR**
- **Greeks** - Delta, Gamma, Vega, Theta, Rho via autograd

### 📉 Forward Curve Models
- **Schwart-Smith (2000)** - two-factor models for forward curve
- **Nelson-Siegel** parameterization
- **Cubic Spline** interpolation
- **Arbitrage-Free** curve construction

### 💡 PnL Attribution
- **Detailed Decomposition**
  - Trading activity
  - Market changes
  - Carry / roll
  - Theta decay
  - Delta PnL
  - Gamma PnL
  - Vega PnL
  - Residual/unexplained
- **Position-Level** attribution
- **Portfolio-Level** hierarchical views

---

## 🚀 Quick Examples

### Price an Option with Automatic Greeks
```python
import quant_tensors as qt
import torch

# Enable gradient tracking
spot = torch.tensor(100.0, requires_grad=True)
vol = torch.tensor(0.25, requires_grad=True)
strike = 105.0
T = 0.5
r = 0.03

# Price European call
forward = spot * torch.exp(torch.tensor(r * T))
price = qt.black76.call(forward, strike, T, vol, r)

# Compute Greeks automatically (no finite differences!)
delta = torch.autograd.grad(price, spot, create_graph=True)[0]
gamma = torch.autograd.grad(delta, spot)[0]
vega = torch.autograd.grad(price, vol)[0]

print(f"Price: ${price:.4f}")
print(f"Delta: {delta:.4f}")
print(f"Gamma: {gamma:.4f}")
print(f"Vega:  {vega:.4f}")
Monte Carlo Simulation on GPU
pythonimport quant_tensors as qt
import torch

# Setup Monte Carlo engine
mc = qt.MonteCarlo(
    n_paths=1_000_000,
    n_steps=252,
    device='cuda'  # Run on GPU
)

# Asian call payoff
def asian_call(paths, strike):
    avg_price = paths.mean(dim=1)
    return torch.maximum(avg_price - strike, torch.tensor(0.0))

# Simulate paths
spot = 100.0
vol = 0.3
T = 1.0
strike = 100.0

paths = mc.simulate_gbm(spot, vol, T, r=0.05)
payoff = asian_call(paths, strike)
price = payoff.mean() * torch.exp(torch.tensor(-0.05 * T))

print(f"Asian Call: ${price:.4f}")
print(f"Std Error: ${payoff.std() / torch.sqrt(torch.tensor(1e6)):.4f}")
Portfolio VaR Calculation
pythonimport quant_tensors as qt
import torch

# Portfolio positions (WTI, Brent, Natural Gas)
positions = torch.tensor([1000., -500., 750.])
prices = torch.tensor([75.0, 80.0, 3.5])

# Historical returns (1000 days, 3 assets)
returns = torch.randn(1000, 3) * 0.02

# Calculate VaR and CVaR
portfolio_returns = (positions * prices).unsqueeze(0) * returns
var_95 = qt.risk.historical_var(portfolio_returns, confidence=0.95)
cvar_95 = qt.risk.expected_shortfall(portfolio_returns, confidence=0.95)

print(f"1-Day VaR (95%): ${var_95:,.2f}")
print(f"1-Day CVaR (95%): ${cvar_95:,.2f}")

# Marginal VaR for each position
marginal = qt.risk.marginal_var(positions, prices, returns, confidence=0.95)
print(f"Marginal VaR: {marginal}")
Calibrate SVI Volatility Surface
pythonimport quant_tensors as qt

# Market implied volatilities
strikes = [85., 90., 95., 100., 105., 110., 115.]
market_vols = [0.32, 0.29, 0.27, 0.26, 0.28, 0.31, 0.33]
spot = 100.0
T = 0.5

# Calibrate SVI model
svi = qt.volatility.SVI(device='cuda')
params, loss = svi.calibrate(
    strikes=strikes,
    implied_vols=market_vols,
    spot=spot,
    time_to_maturity=T,
    optimizer='adamw',
    weight_decay=0.001,
    n_restarts=3
)

# Generate smooth volatility curve
fine_strikes = torch.linspace(70., 130., 100)
smooth_vols = svi.implied_volatility(fine_strikes, params)

print(f"Calibration Loss: {loss:.6f}")
PnL Attribution
pythonimport quant_tensors as qt

# Define portfolio
portfolio = qt.Portfolio([
    {'type': 'call', 'strike': 100, 'quantity': 1000, 'T': 0.5},
    {'type': 'put', 'strike': 95, 'quantity': -500, 'T': 0.5},
])

# Market state at T0 and T1
market_t0 = {'spot': 100.0, 'vol': 0.25, 'rate': 0.03}
market_t1 = {'spot': 102.0, 'vol': 0.27, 'rate': 0.03}

# Compute PnL attribution
attribution = qt.pnl.attribute(
    portfolio, market_t0, market_t1,
    components=['carry', 'theta', 'delta', 'gamma', 'vega', 'residual']
)

for component, value in attribution.items():
    print(f"{component:10s}: ${value:>10,.2f}")

📊 Performance
Pricing Speed (10,000 Black-76 Options)
DeviceTimeSpeedupCPU (NumPy)2,450 ms1xCPU (PyTorch)1,830 ms1.3xGPU (CUDA)24 ms102x
Monte Carlo (1M paths, 252 steps)
DeviceTimeSpeedupCPU45.3 s1xGPU0.8 s57x
Greeks Calculation (10,000 options)
MethodTimeSpeedupFinite Differences (CPU)18.5 s1xAutograd (GPU)0.15 s123x
VaR (10,000 positions, 1,000 scenarios)
DeviceTimeSpeedupCPU8.2 s1xGPU0.3 s27x

🏗️ Architecture
quant_tensors/
├── black76/          # Black-76 pricing models
│   ├── european.py   # European options
│   ├── american.py   # American options
│   ├── asian.py      # Asian options
│   ├── barrier.py    # Barrier options
│   └── spread.py     # Spread options
│
├── montecarlo/       # Monte Carlo simulation
│   ├── engine.py     # Core MC engine
│   ├── processes.py  # Stochastic processes (GBM, Heston, etc.)
│   ├── variance.py   # Variance reduction
│   └── quasi.py      # Quasi-random sequences
│
├── volatility/       # Volatility models
│   ├── svi.py        # SVI model
│   ├── sabr.py       # SABR model
│   ├── local_vol.py  # Local volatility
│   └── surface.py    # Surface utilities
│
├── risk/             # Risk analytics
│   ├── var.py        # Value at Risk
│   ├── cvar.py       # Expected Shortfall
│   ├── stress.py     # Stress testing
│   └── greeks.py     # Greek calculations
│
├── curves/           # Forward curves
│   ├── forward.py    # Forward curve base
│   ├── nelson.py     # Nelson-Siegel
│   ├── spline.py     # Cubic splines
│   └── smooth.py     # Maximum smoothness
│
├── pnl/              # PnL attribution
│   ├── attribution.py # Attribution engine
│   ├── components.py  # Component calculations
│   └── portfolio.py   # Portfolio-level
│
└── utils/            # Utilities
    ├── tensors.py # utilities for taking arrays, lists and tuples to tensors
    ├── instruments.py  # define instruments objects

🎯 Design Principles

GPU-First Architecture - All operations optimized for GPU execution
Automatic Differentiation - Leverage PyTorch autograd for exact Greeks
Batch Everything - Vectorized operations for maximum throughput
Type Safety - Full type hints for IDE support and correctness
Flexible Inputs - Accept lists, NumPy arrays, or PyTorch tensors
Production-Ready - Comprehensive validation and error handling

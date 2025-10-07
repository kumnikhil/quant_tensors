"""
Historical Value at Risk (VaR) Module
======================================

GPU-accelerated historical VaR and related risk metrics using PyTorch.

This module provides functions for computing:
- Historical VaR 
- Expected Shortfall / CVaR
- Marginal VaR
- Component VaR
- Incremental VaR

All functions support GPU acceleration and flexible input formats.
"""
import torch 
from typing import Union, Optional, Tuple, List
import warnings

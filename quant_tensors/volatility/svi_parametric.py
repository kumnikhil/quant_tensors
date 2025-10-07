import torch
from typing import Optional, Tuple, Literal
import warnings
import numpy as np
from scipy.stats import norm

class SVI:
    """
    Stochastic Volatility Inspired (SVI) model for volatility surface parameterization.
    
    The SVI model parameterizes total variance as:
        w(k) = a + b * (ρ * (k - m) + sqrt((k - m)² + σ²))
    
    where:
        k: log-moneyness (log(K/S))
        w: total variance (IV² * T)
        a: overall level
        b: volatility of volatility
        ρ: skew parameter (-1 < ρ < 1)
        m: center of the smile
        σ: controls the smile curvature
    
    Key Features:
        - Automatic GPU/CPU device management
        - Robust loss functions (MSE, Huber, Relative)
        - AdamW optimizer with decoupled weight decay
        - Multiple random restarts for global optimization
        - Automatic outlier detection
        - Direct calibration from option prices
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize SVI model.
        
        Args:
            device: 'cuda', 'cpu', or None. If None, automatically uses GPU if available.
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"SVI Model initialized on device: {self.device}")
    
    def raw(self, k: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Compute SVI total variance using the raw parameterization.
        
        Formula: w(k) = a + b * (ρ * (k - m) + sqrt((k - m)² + σ²))
        
        Args:
            k: Log-moneyness tensor of shape (N,)
            params: Parameter tensor of shape (5,) containing [a, b, ρ, m, σ]
        
        Returns:
            Total variance tensor of shape (N,)
        
        Example:
            >>> svi = SVI()
            >>> k = torch.tensor([0.0, 0.1, -0.1])
            >>> params = torch.tensor([0.04, 0.1, -0.3, 0.0, 0.15])
            >>> w = svi.raw(k, params)
        """
        # Ensure tensors are on correct device
        k = k.to(self.device)
        params = params.to(self.device)
        
        # Extract parameters
        a = params[0]      # overall level
        b = params[1]      # vol of vol
        rho = params[2]    # skew
        m = params[3]      # center
        sigma = params[4]  # curvature
        
        # Apply constraints to ensure valid parameters
        b = torch.clamp(b, min=1.0e-8)
        sigma = torch.clamp(sigma, min=1.0e-4)
        rho = torch.clamp(rho, min=-0.999999, max=0.99999)
        
        # Compute SVI formula
        k_centered = k - m
        sqrt_term = torch.sqrt(k_centered**2 + sigma**2)
        total_var = a + b * (rho * k_centered + sqrt_term)
        
        return torch.clamp(total_var, min=1.0e-8)
    
    def objective(
        self,
        params: torch.Tensor,
        k: torch.Tensor,
        market_w: torch.Tensor,
        weights: torch.Tensor,
        regularization: float = 0.0,
        loss_type: str = 'mse'
    ) -> torch.Tensor:
        """
        Compute weighted loss between model and market with optional regularization.
        
        Args:
            params: Parameter tensor of shape (5,)
            k: Log-moneyness tensor of shape (N,)
            market_w: Market total variance tensor of shape (N,)
            weights: Weight tensor of shape (N,), should sum to 1
            regularization: L2 regularization strength (for Adam optimizer)
            loss_type: 'mse' (standard), 'huber' (robust), or 'relative' (percentage-based)
        
        Returns:
            Scalar loss value
        """
        model_w = self.raw(k, params)
        
        # Choose loss function
        if loss_type == 'mse':
            # Standard mean squared error
            errors = (model_w - market_w)**2
        elif loss_type == 'huber':
            # Robust Huber loss (less sensitive to outliers)
            delta = 0.1
            abs_error = torch.abs(model_w - market_w)
            errors = torch.where(
                abs_error <= delta,
                0.5 * abs_error**2,
                delta * (abs_error - 0.5 * delta)
            )
        elif loss_type == 'relative':
            # Relative error (percentage-based)
            errors = ((model_w - market_w) / (market_w + 1e-8))**2
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}. Use 'mse', 'huber', or 'relative'")
        
        # Weighted loss
        data_loss = torch.sum(weights * errors)
        
        # Add L2 regularization (for Adam only; AdamW uses weight_decay)
        if regularization > 0:
            # Penalize b and sigma to prevent overfitting
            reg_loss = regularization * (params[1]**2 + params[4]**2)
            return data_loss + reg_loss
        
        return data_loss
    
    def calibrate(
        self,
        k: torch.Tensor,
        market_w: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        initial_params: Optional[torch.Tensor] = None,
        lr: float = 0.005,
        tol: float = 1e-6,
        early_stopping_rounds: int = 50,
        max_iter: int = 5000,
        verbose: bool = False,
        regularization: float = 0.0,
        loss_type: str = 'mse',
        n_restarts: int = 2,
        use_scheduler: bool = True,
        optimizer_type: str = 'adamw',
        weight_decay: float = 0.001
    ) -> Tuple[torch.Tensor, float]:
        """
        Calibrate SVI parameters to market data with robust optimization.
        
        Args:
            k: Log-moneyness tensor
            market_w: Market total variance tensor
            weights: Weights for each point (optional, defaults to equal weights)
            initial_params: Starting parameters (optional, uses smart initialization if None)
            lr: Initial learning rate
            tol: Convergence tolerance
            early_stopping_rounds: Patience for early stopping
            max_iter: Maximum iterations per restart
            verbose: Print progress if True
            regularization: L2 regularization strength (for Adam only, use 0 for AdamW)
            loss_type: 'mse', 'huber' (robust to outliers), or 'relative'
            n_restarts: Number of random restarts (>1 for robustness)
            use_scheduler: Use learning rate scheduler if True
            optimizer_type: 'adamw' (recommended), 'adam', or 'sgd'
            weight_decay: Weight decay for AdamW (alternative to regularization)
        
        Returns:
            best_params: Calibrated parameters [a, b, ρ, m, σ]
            best_loss: Final loss value
        
        Example:
            >>> svi = SVI()
            >>> k = torch.linspace(-0.3, 0.3, 30)
            >>> market_w = torch.rand(30) * 0.05 + 0.03
            >>> params, loss = svi.calibrate(k, market_w, optimizer_type='adamw', 
            ...                               weight_decay=0.001, loss_type='huber')
        """
        # Move data to device
        k = k.to(self.device)
        market_w = market_w.to(self.device)
        
        # Initialize weights
        if weights is None:
            weights = torch.ones_like(k)
        weights = weights.to(self.device)
        weights = weights / torch.sum(weights)
        
        # Detect and handle outliers (except when using Huber loss)
        if loss_type != 'huber':
            k, market_w, weights = self._handle_outliers(k, market_w, weights)
        
        best_overall_loss = float('inf')
        best_overall_params = None
        
        # Multiple random restarts for robustness
        for restart in range(n_restarts):
            if verbose and n_restarts > 1:
                print(f"\n--- Restart {restart + 1}/{n_restarts} ---")
            
            # Initialize parameters
            if initial_params is None or restart > 0:
                init_params = self._initialize_params(k, market_w, restart)
            else:
                init_params = initial_params
            
            if not isinstance(init_params, torch.Tensor):
                init_params = torch.tensor(init_params, dtype=torch.float32)
            
            # Run optimization
            params, loss = self._optimize(
                k, market_w, weights, init_params,
                lr, tol, early_stopping_rounds, max_iter,
                verbose, regularization, loss_type, use_scheduler,
                optimizer_type, weight_decay
            )
            
            if loss < best_overall_loss:
                best_overall_loss = loss
                best_overall_params = params
                if verbose and n_restarts > 1:
                    print(f"  → New best loss: {loss:.8f}")
        
        return best_overall_params, best_overall_loss
    
    def _optimize(
        self,
        k: torch.Tensor,
        market_w: torch.Tensor,
        weights: torch.Tensor,
        initial_params: torch.Tensor,
        lr: float,
        tol: float,
        early_stopping_rounds: int,
        max_iter: int,
        verbose: bool,
        regularization: float,
        loss_type: str,
        use_scheduler: bool,
        optimizer_type: str,
        weight_decay: float
    ) -> Tuple[torch.Tensor, float]:
        """Single optimization run with specified optimizer."""
        
        params = initial_params.clone().detach().to(self.device).requires_grad_(True)
        
        # Choose optimizer
        if optimizer_type.lower() == 'adamw':
            optimizer = torch.optim.AdamW([params], lr=lr, weight_decay=weight_decay)
        elif optimizer_type.lower() == 'adam':
            optimizer = torch.optim.Adam([params], lr=lr)
        elif optimizer_type.lower() == 'sgd':
            optimizer = torch.optim.SGD([params], lr=lr, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}. Use 'adamw', 'adam', or 'sgd'")
        
        # Learning rate scheduler for better convergence
        if use_scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=20, verbose=False
            )
        
        # Compute bounds for clamping
        k_min, k_max = torch.min(k), torch.max(k)
        k_range = k_max - k_min
        max_w = torch.max(market_w)
        
        best_loss = float('inf')
        best_params = params.clone().detach()
        no_improve_count = 0
        
        for iteration in range(max_iter):
            optimizer.zero_grad()
            
            # Compute loss
            # Note: Use regularization only with Adam (AdamW uses weight_decay instead)
            reg = regularization if optimizer_type.lower() == 'adam' else 0.0
            loss = self.objective(params, k, market_w, weights, reg, loss_type)
            loss.backward()
            
            # Gradient clipping for numerical stability
            torch.nn.utils.clip_grad_norm_([params], max_norm=1.0)
            
            optimizer.step()
            
            # Apply parameter constraints after optimization step
            with torch.no_grad():
                params[0].clamp_(min=1.0e-6, max=max_w * 2.0)  # a
                params[1].clamp_(min=1.0e-6, max=max_w * 2.0)  # b
                params[2].clamp_(min=-0.99999, max=0.99999)    # rho
                params[3].clamp_(min=k_min - 2.0 * k_range, 
                                max=k_max + 2.0 * k_range)     # m
                params[4].clamp_(min=1.0e-4, max=k_range * 2.0) # sigma
            
            current_loss = loss.item()
            
            # Update learning rate scheduler
            if use_scheduler:
                scheduler.step(current_loss)
            
            # Check for improvement
            if abs(best_loss - current_loss) < tol:
                no_improve_count += 1
            else:
                no_improve_count = 0
            
            if current_loss < best_loss:
                best_loss = current_loss
                best_params = params.clone().detach()
            
            # Verbose output
            if verbose and (iteration % 100 == 0 or iteration == max_iter - 1):
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Iter {iteration:5d}: loss={current_loss:.8f}, lr={current_lr:.6f}")
            
            # Early stopping
            if no_improve_count >= early_stopping_rounds:
                if verbose:
                    print(f"Early stopping at iteration {iteration}")
                break
        
        return best_params, best_loss
    
    def _handle_outliers(
        self, 
        k: torch.Tensor, 
        market_w: torch.Tensor, 
        weights: torch.Tensor,
        threshold: float = 3.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Detect and downweight outliers using z-score method."""
        
        # Compute z-scores
        mean_w = torch.mean(market_w)
        std_w = torch.std(market_w)
        z_scores = torch.abs((market_w - mean_w) / (std_w + 1e-8))
        
        # Identify outliers
        outlier_mask = z_scores > threshold
        if torch.any(outlier_mask):
            n_outliers = torch.sum(outlier_mask).item()
            weights = weights.clone()
            weights[outlier_mask] *= 0.1  # Reduce weight of outliers
            weights = weights / torch.sum(weights)  # Renormalize
            
            warnings.warn(f"Detected {n_outliers} outliers, downweighting them.")
        
        return k, market_w, weights
    
    def _initialize_params(
        self, 
        k: torch.Tensor, 
        market_w: torch.Tensor,
        restart: int = 0
    ) -> torch.Tensor:
        """
        Smart parameter initialization with optional randomization for restarts.
        
        Uses heuristics based on market data characteristics:
        - a: initialized to minimum variance
        - b: based on variance range
        - rho: based on skew direction
        - m: location of minimum variance
        - sigma: based on strike range
        """
        
        min_w = torch.min(market_w)
        max_w = torch.max(market_w)
        k_min, k_max = torch.min(k), torch.max(k)
        k_range = k_max - k_min
        
        # Base initialization
        sigma_guess = torch.maximum(k_range / 4.0, torch.tensor(0.05))
        b_guess = torch.maximum((max_w - min_w) / (2.0 * sigma_guess), torch.tensor(1.0e-3))
        rho_guess = torch.clip(torch.sign(market_w[-1] - market_w[0]) * 0.5, -0.99, 0.99)
        m_guess = k[torch.argmin(market_w)]
        
        # Add randomization for restarts > 0
        if restart > 0:
            torch.manual_seed(restart * 42)
            a_init = min_w.item() * (0.8 + 0.4 * torch.rand(1).item())
            b_init = b_guess.item() * (0.8 + 0.4 * torch.rand(1).item())
            rho_init = torch.clip(rho_guess + 0.3 * (torch.rand(1) - 0.5), -0.95, 0.95).item()
            m_init = m_guess.item() + 0.2 * k_range.item() * (torch.rand(1).item() - 0.5)
            sigma_init = sigma_guess.item() * (0.8 + 0.4 * torch.rand(1).item())
        else:
            a_init = min_w.item()
            b_init = b_guess.item()
            rho_init = rho_guess.item()
            m_init = m_guess.item()
            sigma_init = sigma_guess.item()
        
        return torch.tensor([a_init, b_init, rho_init, m_init, sigma_init], 
                          dtype=torch.float32, device=self.device)
    
    @staticmethod
    def option_premiums_to_variance(
        option_prices: torch.Tensor,
        strikes: torch.Tensor,
        spot: float,
        time_to_maturity: float,
        rate: float = 0.0,
        option_type: str = 'call',
        method: str = 'newton'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert option premiums to implied volatilities and then to total variance.
        
        Args:
            option_prices: Market option prices
            strikes: Strike prices
            spot: Current spot price
            time_to_maturity: Time to expiry in years
            rate: Risk-free rate
            option_type: 'call' or 'put'
            method: 'newton' (accurate) or 'brenner-subrahmanyam' (fast approximation)
        
        Returns:
            k: Log-moneyness
            w: Total variance (IV² * T)
        
        Raises:
            ImportError: If scipy is not available (required for Newton method)
        
        Example:
            >>> option_prices = torch.tensor([15.5, 11.2, 7.8, 5.1])
            >>> strikes = torch.tensor([85., 90., 95., 100.])
            >>> k, w = SVI.option_premiums_to_variance(option_prices, strikes, 
            ...                                         spot=100.0, time_to_maturity=0.25)
        """
        # Compute log-moneyness
        k = torch.log(strikes / spot)
        
        # Convert to numpy for scipy
        option_prices_np = option_prices.cpu().numpy()
        strikes_np = strikes.cpu().numpy()
        
        ivs = []
        for i, (price, strike) in enumerate(zip(option_prices_np, strikes_np)):
            try:
                iv = SVI._implied_volatility(
                    price, spot, strike, time_to_maturity, rate, option_type, method
                )
                ivs.append(iv)
            except Exception as e:
                warnings.warn(f"Failed to compute IV for strike {strike}: {e}")
                ivs.append(np.nan)
        
        ivs = torch.tensor(ivs, dtype=torch.float32)
        
        # Remove NaN values
        valid_mask = ~torch.isnan(ivs)
        k = k[valid_mask]
        ivs = ivs[valid_mask]
        
        # Compute total variance w = IV² * T
        w = ivs**2 * time_to_maturity
        
        return k, w
    
    @staticmethod
    def _implied_volatility(
        option_price: float,
        spot: float,
        strike: float,
        T: float,
        r: float,
        option_type: str,
        method: str
    ) -> float:
        """Compute implied volatility from option price using specified method."""
        
        if method == 'brenner-subrahmanyam':
            # Fast approximation for ATM options
            return np.sqrt(2 * np.pi / T) * (option_price / spot)
        
        elif method == 'newton':
            # Newton-Raphson method (accurate)
            def bs_price(vol):
                d1 = (np.log(spot / strike) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
                d2 = d1 - vol * np.sqrt(T)
                
                if option_type == 'call':
                    return spot * norm.cdf(d1) - strike * np.exp(-r * T) * norm.cdf(d2)
                else:
                    return strike * np.exp(-r * T) * norm.cdf(-d2) - spot * norm.cdf(-d1)
            
            def vega(vol):
                d1 = (np.log(spot / strike) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
                return spot * norm.pdf(d1) * np.sqrt(T)
            
            # Initial guess
            vol = 0.2
            
            # Newton-Raphson iterations
            for _ in range(50):
                price = bs_price(vol)
                diff = option_price - price
                
                if abs(diff) < 1e-6:
                    break
                
                v = vega(vol)
                if v < 1e-10:
                    break
                
                vol = vol + diff / v
                vol = max(0.001, min(vol, 5.0))  # Constrain to reasonable range
            
            return vol
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def calibrate_from_options(
        self,
        option_prices: torch.Tensor,
        strikes: torch.Tensor,
        spot: float,
        time_to_maturity: float,
        rate: float = 0.0,
        option_type: str = 'call',
        **calibrate_kwargs
    ) -> Tuple[torch.Tensor, float]:
        """
        Calibrate SVI model directly from option premiums.
        
        Args:
            option_prices: Market option prices
            strikes: Strike prices  
            spot: Current spot price
            time_to_maturity: Time to expiry in years
            rate: Risk-free rate
            option_type: 'call' or 'put'
            **calibrate_kwargs: Additional arguments passed to calibrate()
        
        Returns:
            best_params: Calibrated parameters [a, b, ρ, m, σ]
            best_loss: Final loss value
        
        Example:
            >>> option_prices = torch.tensor([15.5, 11.2, 7.8, 5.1, 3.2, 2.1, 1.4])
            >>> strikes = torch.tensor([85., 90., 95., 100., 105., 110., 115.])
            >>> params, loss = svi.calibrate_from_options(option_prices, strikes,
            ...                                            spot=100.0, time_to_maturity=0.25)
        """
        # Convert option prices to total variance
        k, market_w = self.option_premiums_to_variance(
            option_prices, strikes, spot, time_to_maturity, rate, option_type
        )
        
        # Calibrate using total variance
        return self.calibrate(k, market_w, **calibrate_kwargs)


def test_SVI():
    """Demonstrate basic usage of the Enhanced SVI model."""
    
    print("=" * 80)
    print("SVI Model - Example Usage")
    print("=" * 80)
    
    # Initialize model (automatically uses GPU if available)
    svi = SVI(device='cpu')
    
    print("\n1. Default Calibration- AdamW")
    print("-" * 80)
    
    torch.manual_seed(42)
    k = torch.linspace(-0.3, 0.3, 30)
    true_params = torch.tensor([0.04, 0.1, -0.35, 0.0, 0.15])
    market_w = svi.raw(k, true_params) + torch.randn_like(k) * 0.002
    
    # Calibrate with AdamW (recommended)
    params, loss = svi.calibrate(
        k, market_w,
        optimizer_type='adamw',
        weight_decay=0.001,
        loss_type='huber',
        n_restarts=3,
        lr=0.005,
        max_iter=2000,
        verbose=True
    )
    
    print("\nCalibrated parameters:")
    param_names = ['a', 'b', 'rho', 'm', 'sigma']
    true_params_np = true_params.numpy()
    for i, name in enumerate(param_names):
        print(f"  {name}: {params[i]:.6f} (true: {true_params_np[i]:.6f})")
    print(f"Final loss: {loss:.8f}")
    
    # Example 2: Calibration from option prices
    print("\n\n2. Calibration from Option Prices")
    print("-" * 80)
    
    spot = 100.0
    strikes = torch.tensor([85., 90., 95., 100., 105., 110., 115.])
    time_to_maturity = 0.25
    
    # Simulate option prices
    k_opt = torch.log(strikes / spot)
    params_opt = torch.tensor([0.035, 0.12, -0.45, 0.0, 0.18])
    total_var = svi.raw(k_opt, params_opt)
    implied_vols = torch.sqrt(total_var / time_to_maturity)
    
    # Approximate call prices
    intrinsic = torch.maximum(torch.tensor(spot) - strikes, torch.tensor(0.))
    time_value = 0.4 * implied_vols * spot * torch.sqrt(torch.tensor(time_to_maturity))
    option_prices = intrinsic + time_value + torch.randn_like(intrinsic) * 0.1
    
    print(f"Spot: ${spot:.2f}")
    print(f"Time to maturity: {time_to_maturity} years")
    print(f"Strikes: {strikes.numpy()}")
    print(f"Option prices: {option_prices.numpy()}")
    
    # Calibrate
    params_from_options, loss_options = svi.calibrate_from_options(
        option_prices,
        strikes,
        spot,
        time_to_maturity,
        rate=0.02,
        option_type='call',
        optimizer_type='adamw',
        weight_decay=0.001,
        loss_type='huber',
        n_restarts=3,
        lr=0.01,
        max_iter=2000
    )
    
    print("\nCalibrated from options:")
    for i, name in enumerate(param_names):
        print(f"  {name}: {params_from_options[i]:.6f}")
    print(f"Loss: {loss_options:.8f}")
    
    # Example 3: Compare optimizers
    print("\n\n3. Optimizer Comparison")
    print("-" * 80)
    
    # Adam
    params_adam, loss_adam = svi.calibrate(
        k, market_w,
        optimizer_type='adam',
        regularization=0.001,
        max_iter=1500,
        verbose=False
    )
    
    # AdamW
    params_adamw, loss_adamw = svi.calibrate(
        k, market_w,
        optimizer_type='adamw',
        weight_decay=0.001,
        max_iter=1500,
        verbose=False
    )
    
    print(f"Adam loss:   {loss_adam:.8f}")
    print(f"AdamW loss:  {loss_adamw:.8f}")
    improvement = (loss_adam - loss_adamw) / loss_adam * 100
    if improvement > 0:
        print(f"AdamW is {improvement:.1f}% better")
    
    print("\n" + "=" * 80)
    print("✓ Examples complete!")
    print("=" * 80)


if __name__ == "__main__":
    test_SVI()
"""
Black-Scholes model and implied volatility calculations
"""
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar
from voldiscount.config.config import DEFAULT_PARAMS
from typing import Dict, Any

def black_scholes(
    S: float, 
    K: float, 
    T: float, 
    r: float, 
    sigma: float, 
    **kwargs
) -> float:
    """
    Calculate option price using Black-Scholes model.
    
    Parameters:
    -----------
    S : float
        Underlying price
    K : float
        Strike price
    T : float
        Time to expiry in years
    r : float
        Risk-free interest rate (annualized)
    sigma : float
        Implied volatility (annualized)
    option_type : str
        'call' or 'put'
    q : float
        Dividend/repo rate (annualized)
        
    Returns:
    --------
    float : Option price
    """
    params: Dict[str, Any] = DEFAULT_PARAMS.copy()
    params.update(kwargs)

    # Input validation
    if not all(isinstance(param, (int, float)) for param in [S, K, T, r, sigma, params['q']]):
        return np.nan
        
    if sigma <= 0 or T <= 0 or S <= 0 or K <= 0:
        return np.nan
    
    # Handle potential numerical issues
    try:
        d1 = (np.log(S / K) + (r - params['q'] + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if params['option_type'].lower() == 'call':
            price = S * np.exp(-params['q'] * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:  # put
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-params['q'] * T) * norm.cdf(-d1)
        
        return price
    except (ValueError, ZeroDivisionError, OverflowError):
        return np.nan


def implied_volatility(
    price: float, 
    S: float, 
    K: float, 
    T: float, 
    r: float, 
    **kwargs
) -> float:
    """
    Calculate implied volatility using numerical optimization.
    """
    params: Dict[str, Any] = DEFAULT_PARAMS.copy()
    params.update(kwargs)

    # Define objective function for optimization
    def objective(sigma):
        bs_price = black_scholes(
            S=S, 
            K=K, 
            T=T, 
            r=r, 
            sigma=sigma, 
            option_type=params['option_type'], 
            q=params['q']
            )
        return (bs_price - price) ** 2
    
    try:
        # Use bisection for initial guess
        low, high = 0.05, 1.0  # Keep original range for efficient convergence
        for _ in range(10):
            mid = (low + high) / 2
            mid_price = black_scholes(
                S=S, 
                K=K, 
                T=T, 
                r=r, 
                sigma=mid, 
                option_type=params['option_type'], 
                q=params['q']
                )
            if mid_price > price:
                high = mid
            else:
                low = mid
                
        # Optimize with efficient initial guess but wider acceptance bounds
        result = minimize_scalar(
            objective,
            bounds=(
                max(
                    params['volatility_lower_bound'], 
                    low * params['vol_lb_scalar']
                    ), 
                min(params['volatility_upper_bound'], 
                    high * params['vol_ub_scalar'])
                    ),
            method='bounded',
            options={'xatol': 1e-5, 'maxiter': params['max_iterations']}
        )
        
        # Accept results up to 1000% volatility
        if (result.success and params['volatility_lower_bound'] <= result.x <= params['volatility_upper_bound']): #type: ignore
            return result.x #type: ignore
        else:
            return np.nan
        
    except Exception:
        return np.nan
    
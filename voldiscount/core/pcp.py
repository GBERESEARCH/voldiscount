"""
Put-call parity related functions
"""
import numpy as np
import pandas as pd
from voldiscount.config.config import DEFAULT_PARAMS
from voldiscount.core.black_scholes import implied_volatility
from typing import Dict, Any

def calculate_q_from_pcp(
    put_price: float, 
    call_price: float, 
    strike: float, 
    S: float, 
    T: float, 
    r: float
) -> float:
    """
    Calculate dividend/repo rate from put-call parity.
    
    Parameters:
    -----------
    put_price : float
        Put option price
    call_price : float
        Call option price
    strike : float
        Strike price (same for both options)
    S : float
        Underlying price
    T : float
        Time to expiry in years
    r : float
        Risk-free interest rate
        
    Returns:
    --------
    float : Calculated dividend/repo rate q
    """
    # C - P = S * e^(-q*T) - K * e^(-r*T)
    discount_factor = np.exp(-r * T)
    forward = (call_price - put_price) / discount_factor + strike
    q = -np.log(forward / S) / T
   
    return q


def calculate_implied_volatilities(
    df: pd.DataFrame, 
    S: float, 
    term_structure: pd.DataFrame, 
    **kwargs
) -> pd.DataFrame:
    """
    Calculate implied volatilities for all options using the calibrated discount rates.
    
    Parameters:
    -----------
    df : DataFrame
        Options data for all expiry dates
    S : float
        Underlying price
    term_structure : DataFrame
        Term structure of discount rates
        
    Returns:
    --------
    DataFrame : Options data with implied volatilities
    """
    params: Dict[str, Any] = DEFAULT_PARAMS.copy()
    params.update(kwargs)

    # Create a list to store implied volatilities
    iv_data = []
    
    # Create a lookup dictionary for faster rate retrieval
    rate_lookup = {row['expiry_date']: row['discount_rate'] 
                  for _, row in term_structure.iterrows()}
    
    for _, row in df.iterrows():
        # Find matching discount rate
        expiry = row['Expiry']
        if expiry not in rate_lookup:
            continue
            
        discount_rate = rate_lookup[expiry]
        if np.isnan(discount_rate):
            continue
        
        # Extract parameters
        T = row['Years_To_Expiry']
        K = row['Strike']
        option_type = row['Option Type'].lower()
        price = row['Last Price']
        
        # Calculate moneyness
        moneyness = K / S
        
        # Calculate implied volatility using the discount rate directly with q=0
        try:
            iv = implied_volatility(
                price=price, 
                S=S, 
                K=K, 
                T=T, 
                r=discount_rate,  # Use the discount rate directly
                option_type=option_type, 
                q=0  # No separate dividend/repo rate
            )
            
            if not np.isnan(iv) and params['volatility_lower_bound'] <= iv <= params['volatility_upper_bound']:  # Reasonable IV range: 0.1% to 1000%
                iv_data.append({
                    'Expiry': expiry,
                    'Strike': K,
                    'Moneyness': moneyness,
                    'Days': row['Days_To_Expiry'],
                    'Years': T,
                    'Option_Type': option_type,
                    'IV': iv
                })
        except:
            continue
    
    # Create dataframe of implied volatilities
    iv_df = pd.DataFrame(iv_data)
    
    return iv_df

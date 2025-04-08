import pandas as pd
import numpy as np
from voldiscount.config.config import DEFAULT_PARAMS
from typing import Dict, Any

def calculate_forward_prices(
    df: pd.DataFrame, 
    S: float, 
    **kwargs
) -> Dict[pd.Timestamp, float]:
    """
    Calculate forward prices for each expiry date based on put-call parity.
    
    Parameters:
    -----------
    df : DataFrame
        Options data with put and call prices
    S : float
        Current spot price
    initial_rate : float
        Initial guess for discount rate
    fallback_growth : float
        Annual growth rate to use when good option pairs aren't available
    min_price : float
        Minimum acceptable option price (default 0.0)
    debug : bool
        Whether to print debug information
    debug_threshold : float
        Only print debug info for expiries longer than this many years
    min_forward_ratio : float
        Minimum acceptable forward/spot ratio
    max_forward_ratio : float
        Maximum acceptable forward/spot ratio
        
    Returns:
    --------
    dict : Dictionary mapping expiry dates to forward prices
    """
    
    params: Dict[str, Any] = DEFAULT_PARAMS.copy()
    params.update(kwargs)

    print("Calculating forward prices for each expiry date...")
    
    # Calculate forward prices for each expiry
    forward_prices = {}
    
    # Process each expiry date separately
    for expiry, expiry_df in df.groupby('Expiry'):
        years = expiry_df['Years To Expiry'].iloc[0]
        
        # Get puts and calls for this expiry
        puts = expiry_df[expiry_df['Option Type'].str.lower() == 'put']
        calls = expiry_df[expiry_df['Option Type'].str.lower() == 'call']
        
        if puts.empty or calls.empty:
            # Use fallback growth rate if no pairs available
            forward = S * (1 + params['fallback_growth']) ** years
            forward_prices[expiry] = forward
            continue
        
        # Find strike pairs for exact matches with high liquidity
        exact_pairs = []
        
        # Check if we have Volume or Open Interest columns
        has_volume = 'Volume' in puts.columns and 'Volume' in calls.columns
        has_open_interest = 'Open Interest' in puts.columns and 'Open Interest' in calls.columns
        
        # Find exact strike matches
        common_strikes = set(puts['Strike']).intersection(set(calls['Strike']))
        
        for strike in common_strikes:
            put_options = puts[puts['Strike'] == strike]
            call_options = calls[calls['Strike'] == strike]
            
            if put_options.empty or call_options.empty:
                continue
                
            put_row = put_options.iloc[0]
            call_row = call_options.iloc[0]
            
            # Only consider options with valid prices
            if put_row['Last Price'] <= params['min_price'] or call_row['Last Price'] <= params['min_price']:
                continue
                
            # Calculate moneyness
            strike_float = float(strike)
            moneyness = abs(strike_float / S - 1.0)
            
            # Calculate liquidity score based on volume and open interest if available
            liquidity_score = 0
            
            if has_volume:
                put_vol = float(put_row['Volume']) if not pd.isna(put_row['Volume']) else 0
                call_vol = float(call_row['Volume']) if not pd.isna(call_row['Volume']) else 0
                liquidity_score += put_vol + call_vol
                
            if has_open_interest:
                put_oi = float(put_row['Open Interest']) if not pd.isna(put_row['Open Interest']) else 0
                call_oi = float(call_row['Open Interest']) if not pd.isna(call_row['Open Interest']) else 0
                liquidity_score += put_oi + call_oi
                
            # If no volume/OI data, use inverse of moneyness as proxy for liquidity
            if liquidity_score == 0:
                liquidity_score = 1.0 / (moneyness + 0.01)
            
            exact_pairs.append({
                'strike': strike_float,
                'put_price': float(put_row['Last Price']),
                'call_price': float(call_row['Last Price']),
                'moneyness': moneyness,
                'liquidity_score': liquidity_score
            })
        
        # Calculate forward using the best available pairs
        if exact_pairs:
            # Sort by liquidity score (descending) then by moneyness (ascending)
            exact_pairs.sort(key=lambda x: (-x['liquidity_score'], x['moneyness']))
            
            # Calculate forward estimates from top 3 pairs or all if fewer
            top_pairs = exact_pairs[:min(3, len(exact_pairs))]
            forward_estimates = []
            
            for pair in top_pairs:
                strike = pair['strike']
                put_price = pair['put_price']
                call_price = pair['call_price']
                
                # Simple estimate using initial rate
                discount_factor = np.exp(-params['initial_rate'] * years)
                forward_est = strike + (call_price - put_price) / discount_factor
                
                # Add to estimates if within acceptable bounds
                if params['min_forward_ratio'] * S < forward_est < params['max_forward_ratio'] * S:
                    forward_estimates.append(forward_est)
                    
            # Calculate weighted average if we have estimates
            if forward_estimates:
                forward = sum(forward_estimates) / len(forward_estimates)
            else:
                # Fallback to simple growth model
                forward = S * (1 + params['fallback_growth']) ** years
        else:
            # No exact pairs, use simple growth model
            forward = S * (1 + params['fallback_growth']) ** years
        
        forward_prices[expiry] = forward
        
        print(f"Expiry: {expiry} ({years:.2f} years)")
        print(f"  Spot: {S:.2f}, Forward: {forward:.2f}, Ratio: {forward/S:.4f}")
    
    return forward_prices
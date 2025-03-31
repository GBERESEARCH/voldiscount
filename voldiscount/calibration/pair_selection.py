import pandas as pd
from config.config import DEFAULT_PARAMS
from typing import Dict, Any

def select_option_pairs(df, S, **kwargs):
    """
    Find options with matching or nearly matching strikes for each expiry.
    
    Parameters:
    -----------
    df : DataFrame
        Options data for all expiry dates
    S : float
        Underlying price
    forward_prices : dict or None
        Dictionary mapping expiry dates to forward prices, if None uses spot
    max_strike_diff_pct : float
        Maximum allowed difference between put and call strikes as percentage of S
    min_option_price : float
        Minimum price for valid options
    consider_volume : bool
        Whether to consider volume/open interest in pair selection
    min_pair_volume : int
        Minimum combined volume for a pair to be considered
    debug : bool
        Whether to print debug information
    best_pair_only : bool
        Whether to keep only the best pair for each expiry
        
    Returns:
    --------
    dict : Dictionary mapping expiry dates to lists of put-call pairs
    """
    params: Dict[str, Any] = DEFAULT_PARAMS.copy()
    params.update(kwargs)

    price_ref = "forwards" if params['forward_prices'] else "spot price"
    print(f"Finding strike-matched pairs with max diff: {params['max_strike_diff_pct']*100:.1f}% of {price_ref}")
    
    # Check required columns
    required_cols = ['Expiry', 'Strike', 'Option Type', 'Last Price']
    for col in required_cols:
        if col not in df.columns:
            print(f"ERROR: Missing required column '{col}' in options data")
            return {}
    
    # Ensure Option Type is properly formatted
    if not df['Option Type'].str.lower().isin(['call', 'put']).all():
        df['Option Type'] = df['Option Type'].str.lower()
    
    # Add diagnostic counters
    total_expiries = 0
    expiries_with_options = 0
    expiries_with_min_options = 0
    expiries_with_common_strikes = 0
    expiries_with_valid_pairs = 0
    
    # Process each expiry
    pairs_by_expiry = {}
    for expiry, expiry_df in df.groupby('Expiry'):
        total_expiries += 1
        # Get reference price (spot or forward)
        reference_price = params['forward_prices'].get(expiry, S) if params['forward_prices'] else S
        
        # Get put and call data
        puts = expiry_df[(expiry_df['Option Type'].str.lower() == 'put') & 
                         (expiry_df['Last Price'] > params['min_option_price'])]
        calls = expiry_df[(expiry_df['Option Type'].str.lower() == 'call') & 
                         (expiry_df['Last Price'] > params['min_option_price'])]
        
        if puts.empty or calls.empty:
            print(f"Skipping expiry {expiry} - missing puts or calls")
            continue
        
        expiries_with_options += 1

        # Check for minimum number of options per type (NEW CHECK)
        if len(puts) < params['min_options_per_type'] or len(calls) < params['min_options_per_type']:
            print(f"Skipping expiry {expiry} - insufficient options (need {params['min_options_per_type']} of each type, found {len(puts)} puts, {len(calls)} calls)")
            continue
            
        expiries_with_min_options += 1

        pairs = []
        
        # 1. First try exact strike matches
        common_strikes = set(puts['Strike']).intersection(set(calls['Strike']))

        print(f"Expiry {expiry}: {len(puts)} puts, {len(calls)} calls, {len(common_strikes)} common strikes")
        
        if common_strikes:
            expiries_with_common_strikes += 1

        for strike in common_strikes:
            try:
                put_row = puts[puts['Strike'] == strike].iloc[0]
                call_row = calls[calls['Strike'] == strike].iloc[0]
                
                strike_float = float(strike)
                put_price = float(put_row['Last Price'])
                call_price = float(call_row['Last Price'])
                moneyness = abs(strike_float / reference_price - 1.0)
                                                
                pairs.append({
                    'put_strike': strike_float,
                    'call_strike': strike_float,
                    'put_price': put_price,
                    'call_price': call_price,
                    'strike_diff': 0.0,
                    'strike_diff_pct': 0.0,
                    'moneyness': moneyness,
                    'is_exact_match': True,
                    'years': float(expiry_df['Years To Expiry'].iloc[0]),
                    'days': int(expiry_df['Days To Expiry'].iloc[0])
                })
            except Exception as e:
                print(f"  Error processing strike {strike}: {e}")
        
        # 2. If we don't have enough exact matches, look for close strikes
        if len(pairs) < params['close_strike_min_pairs']:
            max_diff = reference_price * params['max_strike_diff_pct']
            
            for _, put_row in puts.iterrows():
                put_strike = float(put_row['Strike'])
                put_price = float(put_row['Last Price'])
                put_moneyness = abs(put_strike / reference_price - 1.0)
                
                for _, call_row in calls.iterrows():
                    call_strike = float(call_row['Strike'])
                    call_price = float(call_row['Last Price'])
                    call_moneyness = abs(call_strike / reference_price - 1.0)
                    
                    # Check if strikes are close enough
                    strike_diff = abs(put_strike - call_strike)
                    if strike_diff > max_diff:
                        continue
                        
                    # Skip if we already have an exact match for this strike
                    if any(p['put_strike'] == put_strike and p['call_strike'] == put_strike for p in pairs):
                        continue
                    
                    # Calculate average moneyness
                    avg_moneyness = (put_moneyness + call_moneyness) / 2
                    
                    pairs.append({
                        'put_strike': put_strike,
                        'call_strike': call_strike,
                        'put_price': put_price,
                        'call_price': call_price,
                        'strike_diff': strike_diff,
                        'strike_diff_pct': strike_diff / reference_price,
                        'moneyness': avg_moneyness,  # Use avg_moneyness for consistency
                        'is_exact_match': False,
                        'years': float(put_row['Years To Expiry']),
                        'days': int(put_row['Days To Expiry'])
                    })
        
        if pairs:
            # Sort pairs: exact matches first, then by moneyness
            expiries_with_valid_pairs += 1
            def pair_sort_key(p):
                exact_match = p.get('is_exact_match', False)
                moneyness = p.get('moneyness', 1.0)
                strike_diff = p.get('strike_diff_pct', 0.0)
                
                return (-1 if exact_match else 0, moneyness, strike_diff)
            
            pairs.sort(key=pair_sort_key)
            
            # Keep only the best pair or all pairs
            if params['best_pair_only']:
                pairs_by_expiry[expiry] = [pairs[0]]
            else:
                pairs_by_expiry[expiry] = pairs
   

    print(f"Diagnostics:")
    print(f"  Total expiries: {total_expiries}")
    print(f"  Expiries with both puts and calls: {expiries_with_options}")
    print(f"  Expiries with min. {params['min_options_per_type']} of each option type: {expiries_with_min_options}")
    print(f"  Expiries with common strikes: {expiries_with_common_strikes}")
    print(f"  Expiries with valid pairs after all filtering: {expiries_with_valid_pairs}")
        
    return pairs_by_expiry
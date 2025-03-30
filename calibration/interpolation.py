import pandas as pd
from config.config import DEFAULT_PARAMS
from typing import Dict, Any

def interpolate_rate(df, expiry_date, days, years):
    """
    Interpolate a discount rate for a specific expiry date
    
    Parameters:
    -----------
    df : DataFrame
        Term structure DataFrame
    expiry_date : datetime
        Expiry date to interpolate for
    days : int
        Days to expiry
    years : float
        Years to expiry
    debug : bool
        Whether to print debug info
        
    Returns:
    --------
    DataFrame : Updated term structure with interpolated value
    """
    
    
    # Find the closest dates before and after
    before_df = df[df['days'] < days].sort_values('days', ascending=False)
    after_df = df[df['days'] > days].sort_values('days')
    
    if before_df.empty or after_df.empty:
        print(f"Cannot interpolate for {expiry_date}: insufficient data points")
        return df
    
    before = before_df.iloc[0]
    after = after_df.iloc[0]
    
    # Linear interpolation
    days_range = after['days'] - before['days']
    days_frac = (days - before['days']) / days_range
    rate = before['discount_rate'] + days_frac * (after['discount_rate'] - before['discount_rate'])
    
    # Also interpolate reference_price if available
    reference_price = None
    forward_ratio = None
    if 'reference_price' in before and 'reference_price' in after:
        reference_price = before['reference_price'] + days_frac * (after['reference_price'] - before['reference_price'])
        
    if 'forward_ratio' in before and 'forward_ratio' in after:
        forward_ratio = before['forward_ratio'] + days_frac * (after['forward_ratio'] - before['forward_ratio'])
    
    print(f"Interpolated rate for {expiry_date} ({days} days): {rate:.6f}")
    print(f"  Between: {before['expiry_date']} ({before['days']} days): {before['discount_rate']:.6f}")
    print(f"  And: {after['expiry_date']} ({after['days']} days): {after['discount_rate']:.6f}")
    
    # Create new row for the dataframe
    new_row = {
        'expiry_date': expiry_date,
        'days': days,
        'years': years,
        'discount_rate': rate,
        'method': 'interpolated',
        'put_strike': None,
        'call_strike': None,
        'put_price': None,
        'call_price': None,
        'put_iv': None,
        'call_iv': None,
        'iv_diff': None
    }
    
    # Add reference price and forward ratio if available
    if reference_price is not None:
        new_row['reference_price'] = reference_price
    if forward_ratio is not None:
        new_row['forward_ratio'] = forward_ratio
    
    # Append the new row and return
    return pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

def extrapolate_early(df, expiry_date, days, years, **kwargs):
    """
    Extrapolate a discount rate for an early expiry date
    
    Parameters:
    -----------
    df : DataFrame
        Term structure DataFrame
    expiry_date : datetime
        Expiry date to extrapolate for
    days : int
        Days to expiry
    years : float
        Years to expiry
    debug : bool
        Whether to print debug info
        
    Returns:
    --------
    DataFrame : Updated term structure with extrapolated value
    """
    params: Dict[str, Any] = DEFAULT_PARAMS.copy()
    params.update(kwargs)

    if len(df) < params['min_options_per_expiry']:
        print(f"Cannot extrapolate for {expiry_date}: insufficient data points")
        return df
    
    # Use the first two points for early extrapolation
    first = df.sort_values('days').iloc[0]
    second = df.sort_values('days').iloc[1]
    
    # Simple linear extrapolation
    days_diff = second['days'] - first['days']
    rate_diff = second['discount_rate'] - first['discount_rate']
    daily_rate_change = rate_diff / days_diff
    
    extrapolated_rate = first['discount_rate'] - (first['days'] - days) * daily_rate_change
    extrapolated_rate = max(0.0, extrapolated_rate)  # Ensure non-negative
    
    # Also extrapolate reference_price if available
    reference_price = None
    forward_ratio = None
    if 'reference_price' in first and 'reference_price' in second:
        price_diff = second['reference_price'] - first['reference_price']
        daily_price_change = price_diff / days_diff
        reference_price = first['reference_price'] - (first['days'] - days) * daily_price_change
        reference_price = max(0.0, reference_price)  # Ensure non-negative
        
    if 'forward_ratio' in first and 'forward_ratio' in second:
        ratio_diff = second['forward_ratio'] - first['forward_ratio']
        daily_ratio_change = ratio_diff / days_diff
        forward_ratio = first['forward_ratio'] - (first['days'] - days) * daily_ratio_change
        forward_ratio = max(0.0, forward_ratio)  # Ensure non-negative
    
    print(f"Extrapolated early rate for {expiry_date} ({days} days): {extrapolated_rate:.6f}")
    print(f"  Using: {first['expiry_date']} ({first['days']} days): {first['discount_rate']:.6f}")
    print(f"  And: {second['expiry_date']} ({second['days']} days): {second['discount_rate']:.6f}")
    
    # Create new row for the dataframe
    new_row = {
        'expiry_date': expiry_date,
        'days': days,
        'years': years,
        'discount_rate': extrapolated_rate,
        'method': 'extrapolated',
        'put_strike': None,
        'call_strike': None,
        'put_price': None,
        'call_price': None,
        'put_iv': None,
        'call_iv': None,
        'iv_diff': None
    }
    
    # Add reference price and forward ratio if available
    if reference_price is not None:
        new_row['reference_price'] = reference_price
    if forward_ratio is not None:
        new_row['forward_ratio'] = forward_ratio
    
    # Append the new row and return
    return pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

def extrapolate_late(df, expiry_date, days, years, **kwargs):
    """
    Extrapolate a discount rate for a late expiry date
    
    Parameters:
    -----------
    df : DataFrame
        Term structure DataFrame
    expiry_date : datetime
        Expiry date to extrapolate for
    days : int
        Days to expiry
    years : float
        Years to expiry
    debug : bool
        Whether to print debug info
        
    Returns:
    --------
    DataFrame : Updated term structure with extrapolated value
    """
    
    params: Dict[str, Any] = DEFAULT_PARAMS.copy()
    params.update(kwargs)

    if len(df) < params['min_options_per_expiry']:
        print(f"Cannot extrapolate for {expiry_date}: insufficient data points")
        return df
    
    # Use the last two points for late extrapolation
    last = df.sort_values('days', ascending=False).iloc[0]
    second_last = df.sort_values('days', ascending=False).iloc[1]
    
    # Simple linear extrapolation
    days_diff = last['days'] - second_last['days']
    rate_diff = last['discount_rate'] - second_last['discount_rate']
    daily_rate_change = rate_diff / days_diff
    
    extrapolated_rate = last['discount_rate'] + (days - last['days']) * daily_rate_change
    extrapolated_rate = max(0.0, min(0.2, extrapolated_rate))  # Ensure reasonable bounds
    
    # Also extrapolate reference_price if available
    reference_price = None
    forward_ratio = None
    if 'reference_price' in last and 'reference_price' in second_last:
        price_diff = last['reference_price'] - second_last['reference_price']
        daily_price_change = price_diff / days_diff
        reference_price = last['reference_price'] + (days - last['days']) * daily_price_change
        # Ensure reasonable bounds - allow significant growth for long-dated forward prices
        reference_price = max(last['reference_price'], reference_price)
        
    if 'forward_ratio' in last and 'forward_ratio' in second_last:
        ratio_diff = last['forward_ratio'] - second_last['forward_ratio']
        daily_ratio_change = ratio_diff / days_diff
        forward_ratio = last['forward_ratio'] + (days - last['days']) * daily_ratio_change
        forward_ratio = max(last['forward_ratio'], forward_ratio)
    
    print(f"Extrapolated late rate for {expiry_date} ({days} days): {extrapolated_rate:.6f}")
    print(f"  Using: {second_last['expiry_date']} ({second_last['days']} days): {second_last['discount_rate']:.6f}")
    print(f"  And: {last['expiry_date']} ({last['days']} days): {last['discount_rate']:.6f}")
    
    # Create new row for the dataframe
    new_row = {
        'expiry_date': expiry_date,
        'days': days,
        'years': years,
        'discount_rate': extrapolated_rate,
        'method': 'extrapolated',
        'put_strike': None,
        'call_strike': None,
        'put_price': None,
        'call_price': None,
        'put_iv': None,
        'call_iv': None,
        'iv_diff': None
    }
    
    # Add reference price and forward ratio if available
    if reference_price is not None:
        new_row['reference_price'] = reference_price
    if forward_ratio is not None:
        new_row['forward_ratio'] = forward_ratio
    
    # Append the new row and return
    return pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
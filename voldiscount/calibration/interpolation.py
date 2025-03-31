import pandas as pd
from voldiscount.config.config import DEFAULT_PARAMS
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
        
    Returns:
    --------
    DataFrame : Updated term structure with interpolated value
    """
    
    # Find the closest dates before and after
    before_df = df[df['Days'] < days].sort_values('Days', ascending=False)
    after_df = df[df['Days'] > days].sort_values('Days')
    
    if before_df.empty or after_df.empty:
        print(f"Cannot interpolate for {expiry_date}: insufficient data points")
        return df
    
    before = before_df.iloc[0]
    after = after_df.iloc[0]
    
    # Linear interpolation
    days_range = after['Days'] - before['Days']
    days_frac = (days - before['Days']) / days_range
    rate = before['Discount Rate'] + days_frac * (after['Discount Rate'] - before['Discount Rate'])
    
    # Also interpolate reference_price if available
    reference_price = None
    forward_ratio = None
    if 'Forward Price' in before and 'Forward Price' in after:
        reference_price = before['Forward Price'] + days_frac * (after['Forward Price'] - before['Forward Price'])
        
    if 'Forward Ratio' in before and 'Forward Ratio' in after:
        forward_ratio = before['Forward Ratio'] + days_frac * (after['Forward Ratio'] - before['Forward Ratio'])
    
    print(f"Interpolated rate for {expiry_date} ({days} days): {rate:.6f}")
    print(f"  Between: {before['Expiry']} ({before['Days']} days): {before['Discount Rate']:.6f}")
    print(f"  And: {after['Expiry']} ({after['Days']} days): {after['Discount Rate']:.6f}")
    
    # Create new row for the dataframe
    new_row = {
        'Expiry': expiry_date,
        'Days': days,
        'Years': years,
        'Discount Rate': rate,
        'Method': 'interpolated',
        'Put Strike': None,
        'Call Strike': None,
        'Put Price': None,
        'Call Price': None,
        'Put Implied Vol': None,
        'Call Implied Vol': None,
        'Implied Vol Diff': None
    }
    
    # Add reference price and forward ratio if available
    if reference_price is not None:
        new_row['Forward Price'] = reference_price
    if forward_ratio is not None:
        new_row['Forward Ratio'] = forward_ratio
    
    # Filter out None values
    filtered_row = {k: v for k, v in new_row.items() if v is not None}

    # Only concatenate if there's actual data to add
    if filtered_row:  # Check if dictionary contains any entries
        return pd.concat([df, pd.DataFrame([filtered_row])], ignore_index=True)
    else:
        # If all values were None, just return the original DataFrame unchanged
        return df

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
    first = df.sort_values('Days').iloc[0]
    second = df.sort_values('Days').iloc[1]
    
    # Simple linear extrapolation
    days_diff = second['Days'] - first['Days']
    rate_diff = second['Discount Rate'] - first['Discount Rate']
    daily_rate_change = rate_diff / days_diff
    
    extrapolated_rate = first['Discount Rate'] - (first['Days'] - days) * daily_rate_change
    extrapolated_rate = max(0.0, extrapolated_rate)  # Ensure non-negative
    
    # Also extrapolate reference_price if available
    reference_price = None
    forward_ratio = None
    if 'Forward Price' in first and 'Forward Price' in second:
        price_diff = second['Forward Price'] - first['Forward Price']
        daily_price_change = price_diff / days_diff
        reference_price = first['Forward Price'] - (first['Days'] - days) * daily_price_change
        reference_price = max(0.0, reference_price)  # Ensure non-negative
        
    if 'Forward Ratio' in first and 'Forward Ratio' in second:
        ratio_diff = second['Forward Ratio'] - first['Forward Ratio']
        daily_ratio_change = ratio_diff / days_diff
        forward_ratio = first['Forward Ratio'] - (first['Days'] - days) * daily_ratio_change
        forward_ratio = max(0.0, forward_ratio)  # Ensure non-negative
    
    print(f"Extrapolated early rate for {expiry_date} ({days} days): {extrapolated_rate:.6f}")
    print(f"  Using: {first['Expiry']} ({first['Days']} days): {first['Discount Rate']:.6f}")
    print(f"  And: {second['Expiry']} ({second['Days']} days): {second['Discount Rate']:.6f}")
    
    # Create new row for the dataframe
    new_row = {
        'Expiry': expiry_date,
        'Days': days,
        'Years': years,
        'Discount Rate': extrapolated_rate,
        'Method': 'extrapolated',
        'Put Strike': None,
        'Call Strike': None,
        'Put Price': None,
        'Call Price': None,
        'Put Implied Vol': None,
        'Call Implied Vol': None,
        'Implied Vol Diff': None
    }
    
    # Add reference price and forward ratio if available
    if reference_price is not None:
        new_row['Forward Price'] = reference_price
    if forward_ratio is not None:
        new_row['Forward Ratio'] = forward_ratio
    
    # Filter out None values
    filtered_row = {k: v for k, v in new_row.items() if v is not None}

    # Only concatenate if there's actual data to add
    if filtered_row:  # Check if dictionary contains any entries
        return pd.concat([df, pd.DataFrame([filtered_row])], ignore_index=True)
    else:
        # If all values were None, just return the original DataFrame unchanged
        return df

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
    last = df.sort_values('Days', ascending=False).iloc[0]
    second_last = df.sort_values('Days', ascending=False).iloc[1]
    
    # Simple linear extrapolation
    days_diff = last['Days'] - second_last['Days']
    rate_diff = last['Discount Rate'] - second_last['Discount Rate']
    daily_rate_change = rate_diff / days_diff
    
    extrapolated_rate = last['Discount Rate'] + (days - last['Days']) * daily_rate_change
    extrapolated_rate = max(0.0, min(0.2, extrapolated_rate))  # Ensure reasonable bounds
    
    # Also extrapolate reference_price if available
    reference_price = None
    forward_ratio = None
    if 'Forward Price' in last and 'Forward Price' in second_last:
        price_diff = last['Forward Price'] - second_last['Forward Price']
        daily_price_change = price_diff / days_diff
        reference_price = last['Forward Price'] + (days - last['Days']) * daily_price_change
        # Ensure reasonable bounds - allow significant growth for long-dated forward prices
        reference_price = max(last['Forward Price'], reference_price)
        
    if 'Forward Ratio' in last and 'Forward Ratio' in second_last:
        ratio_diff = last['Forward Ratio'] - second_last['Forward Ratio']
        daily_ratio_change = ratio_diff / days_diff
        forward_ratio = last['Forward Ratio'] + (days - last['Days']) * daily_ratio_change
        forward_ratio = max(last['Forward Ratio'], forward_ratio)
    
    print(f"Extrapolated late rate for {expiry_date} ({days} days): {extrapolated_rate:.6f}")
    print(f"  Using: {second_last['Expiry']} ({second_last['Days']} days): {second_last['Discount Rate']:.6f}")
    print(f"  And: {last['Expiry']} ({last['Days']} days): {last['Discount Rate']:.6f}")
    
    # Create new row for the dataframe
    new_row = {
        'Expiry': expiry_date,
        'Days': days,
        'Years': years,
        'Discount Rate': extrapolated_rate,
        'Method': 'extrapolated',
        'Put Strike': None,
        'Call Strike': None,
        'Put Price': None,
        'Call Price': None,
        'Put Implied Vol': None,
        'Call Implied Vol': None,
        'Implied Vol Diff': None
    }
    
    # Add reference price and forward ratio if available
    if reference_price is not None:
        new_row['Forward Price'] = reference_price
    if forward_ratio is not None:
        new_row['Forward Ratio'] = forward_ratio
    
    # Filter out None values
    filtered_row = {k: v for k, v in new_row.items() if v is not None}

    # Only concatenate if there's actual data to add
    if filtered_row:  # Check if dictionary contains any entries
        return pd.concat([df, pd.DataFrame([filtered_row])], ignore_index=True)
    else:
        # If all values were None, just return the original DataFrame unchanged
        return df
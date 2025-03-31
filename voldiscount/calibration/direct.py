import pandas as pd
import numpy as np
import calendar
from scipy.optimize import minimize_scalar
from core.black_scholes import implied_volatility

from calibration.interpolation import interpolate_rate, extrapolate_early, extrapolate_late
from calibration.pair_selection import select_option_pairs
from config.config import DEFAULT_PARAMS
from typing import Dict, Any

def direct_discount_rate_calibration(df, S, **kwargs):
    """
    Calculate discount rates directly from put-call parity for each expiry using an ATM representative pair.
    
    Parameters:
    -----------
    df : DataFrame
        Options data with put and call prices
    S : float
        Underlying price
    max_strike_diff_pct : float
        Maximum allowed difference between put and call strikes as percentage of S
    debug : bool
        Whether to print debug information
    min_option_price : float
        Minimum acceptable option price
    consider_volume : bool
        Whether to consider volume/open interest in pair selection
    min_pair_volume : int
        Minimum combined volume for a pair to be considered
    min_options_per_expiry : int
        Minimum number of valid option pairs required per expiry
    reference_date : str or datetime or None
        Reference date for filtering options (format: 'YYYY-MM-DD'). 
        If None, uses the maximum trade date in the dataset.
    monthlies : bool
        If True, only include standard monthly expiries (3rd Friday of each month)
    """

    params: Dict[str, Any] = DEFAULT_PARAMS.copy()
    params.update(kwargs)

    print("Performing direct discount rate calibration with ATM representative pairs")
    
    print(df.columns)
    print(params['reference_date'])
    # Filter by reference date if specified
    if 'Last Trade Date' in df.columns and params['reference_date'] is not None:
        # Convert reference_date to datetime if it's a string
        if isinstance(params['reference_date'], str):
            params['reference_date'] = pd.to_datetime(params['reference_date'])
        
        # Ensure reference_date is datetime
        params['reference_date'] = pd.to_datetime(params['reference_date'])
        
        # Filter to options traded on or after the reference date
        filtered_df = df[df['Last Trade Date'] >= params['reference_date']].copy()

        filtered_count = len(df) - len(filtered_df)
        if filtered_count > 0:
            print(f"Filtered out {filtered_count} options with trade dates before {params['reference_date']}")
        print(f"Using {len(filtered_df)} options traded on or after {params['reference_date']}")
        
        # Use filtered dataframe for further processing
        df = filtered_df
    
    # Filter for monthly options (3rd Friday) if requested
    if params['monthlies'] and 'Expiry' in df.columns:
        # Define a function to check if a date is the 3rd Friday of the month
        def is_standard_monthly_expiry(date):
            """
            Check if a date is a standard monthly option expiration (normally 3rd Friday,
            but Thursday when the Friday is a holiday).
            
            Parameters:
            -----------
            date : datetime or pandas.Timestamp
                The date to check
            
            Returns:
            --------
            bool : True if the date is a standard monthly expiration
            """
            date = pd.to_datetime(date)
            
            # Calculate the date of the 3rd Friday
            c = calendar.monthcalendar(date.year, date.month)
            # Find all Fridays (weekday 4 = Friday)
            fridays = [week[4] for week in c if week[4] != 0]
            third_friday = pd.Timestamp(date.year, date.month, fridays[2])
            
            # Check if the date is the 3rd Friday
            if date.date() == third_friday.date():
                return True
            
            # Check if the date is the Thursday before the 3rd Friday (potential holiday adjustment)
            if date.weekday() == 3:  # Thursday
                next_day = date + pd.Timedelta(days=1)
                if next_day.date() == third_friday.date():
                    # This is the Thursday before the 3rd Friday
                    # For longer-dated options, this is likely a holiday-adjusted expiry
                    return True
            
            return False
        
        # Apply the filter using our more sophisticated function
        monthly_df = df[df['Expiry'].apply(is_standard_monthly_expiry)].copy()
        

        monthly_count = len(monthly_df)
        filter_count = len(df) - monthly_count
        print(f"Filtered to {monthly_count} standard monthly expiries (removed {filter_count} non-monthly expiries)")
        
        # Print the retained expiries for verification
        unique_expiries = sorted(monthly_df['Expiry'].unique())
        print(f"Retained {len(unique_expiries)} unique monthly expiry dates:")
        for exp in unique_expiries:
            print(f"  - {exp}")
        
        # Use monthly dataframe for further processing
        df = monthly_df
    
    # Find all valid pairs
    pairs_by_expiry = select_option_pairs(
        df=df, 
        S=S, 
        forward_prices=None, 
        max_strike_diff_pct=params['max_strike_diff_pct'], 
        min_option_price=params['min_option_price'], 
        consider_volume=params['consider_volume'], 
        min_pair_volume=params['min_pair_volume'], 
        debug=params['debug'], 
        best_pair_only=params['best_pair_only']
    )
    
    if not pairs_by_expiry:
        print("ERROR: No valid put-call pairs found. Cannot calibrate.")
        return pd.DataFrame()
    
    term_structure = []
    
    for expiry, pairs in pairs_by_expiry.items():
        if len(pairs) < params['min_options_per_expiry']:
            print(f"Skipping expiry {expiry}: only {len(pairs)} pairs, need at least {params['min_options_per_expiry']}")
            continue
            
        # Find the most ATM pair
        atm_idx = min(range(len(pairs)), key=lambda i: 
            abs((pairs[i]['put_strike'] + pairs[i]['call_strike']) / (2 * S) - 1.0))
        atm_pair = pairs[atm_idx]
        
        # Extract parameters
        put_strike = atm_pair['put_strike']
        call_strike = atm_pair['call_strike']
        put_price = atm_pair['put_price']
        call_price = atm_pair['call_price']
        T = atm_pair['years']
        days = atm_pair['days']
        
        # Calculate rate for this pair
        try:
            optimal_rate, put_iv, call_iv, iv_diff = optimize_discount_rate(
                put_price, call_price, put_strike, call_strike, 
                S, S, T, abs(put_strike - call_strike) < 0.01
            )
            
            # Calculate forward price directly from put-call parity
            # Forward = Strike + (Call - Put) * e^(r*T)
            avg_strike = (put_strike + call_strike) / 2
            discount_factor = np.exp(-optimal_rate * T)
            forward_price = avg_strike + (call_price - put_price) / discount_factor
            forward_ratio = forward_price / S
            
            term_structure.append({
                'Expiry': expiry,
                'Days': days,
                'Years': round(T, 4),
                'Discount Rate': round(optimal_rate, 4),
                'Put Strike': put_strike,
                'Call Strike': call_strike,
                'Put Price': put_price,
                'Call Price': call_price,
                'Put Implied Vol': round(put_iv, 4),
                'Call Implied Vol': round(call_iv, 4),
                'Implied Vol Diff': round(iv_diff, 4),
                'Forward Price': round(forward_price, 4),
                'Forward Ratio': round(forward_ratio, 4)
            })
            
        except Exception as e:
            print(f"ERROR calculating rate for expiry {expiry}: {e}")
    
    # Convert to DataFrame and sort
    df_term_structure = pd.DataFrame(term_structure).sort_values('Days')

    print(f"Direct calibration created term structure with {len(df_term_structure)} expiries")

    # After creating the initial term_structure
    if not df_term_structure.empty:
        # Find all unique expiries in original data
        all_expiries = set(df['Expiry'].unique())
        
        # Find expiries we have rates for
        calculated_expiries = set(df_term_structure['Expiry'].unique())
        
        # Determine missing expiries
        missing_expiries = all_expiries - calculated_expiries
        
        if missing_expiries:
            print(f"Interpolating rates for {len(missing_expiries)} missing expiries")
            df_term_structure = apply_interpolation(df_term_structure, df, missing_expiries)
    
    return df_term_structure

def optimize_discount_rate(put_price, call_price, put_strike, call_strike, 
                          S, reference_price, T, strikes_equal, **kwargs):
    """
    Optimize the discount rate to match implied volatilities or satisfy put-call parity.
    """
   
    params: Dict[str, Any] = DEFAULT_PARAMS.copy()
    params.update(kwargs)

    # Define objective functions with unique names
    def objective_equal_strikes(rate):
        # Calculate IVs with the given discount rate for equal strikes
        put_iv = implied_volatility(
            price=put_price, 
            S=S, 
            K=put_strike, 
            T=T, 
            r=rate, 
            option_type='put', 
            q=0
            )
        call_iv = implied_volatility(
            price=call_price, 
            S=S, 
            K=call_strike, 
            T=T, 
            r=rate, 
            option_type='call', 
            q=0
            )
        
        # Return the absolute difference between IVs
        if np.isnan(put_iv) or np.isnan(call_iv):
            return 1.0  # Penalize invalid results
        
        return abs(put_iv - call_iv)
        
    def objective_different_strikes(rate):
        # For different strikes, use put-call parity with the midpoint strike
        K_avg = (put_strike + call_strike) / 2
        
        # Calculate what the call price should be using put-call parity
        synthetic_call = put_price + reference_price - K_avg * np.exp(-rate * T)
        
        # Calculate what the put price should be using put-call parity
        synthetic_put = call_price - reference_price + K_avg * np.exp(-rate * T)
        
        # We want to minimize the relative pricing error
        call_error = abs(synthetic_call - call_price) / call_price
        put_error = abs(synthetic_put - put_price) / put_price
        
        return call_error + put_error
        
    # Select the appropriate objective function based on whether strikes are equal
    objective_function = objective_equal_strikes if strikes_equal else objective_different_strikes
        
    # Initial guess based on direct calculation from put-call parity
    try:
        K_avg = (put_strike + call_strike) / 2
        forward_price = call_price - put_price + K_avg
        initial_rate = -np.log(forward_price / reference_price) / T
        
        # Ensure initial_rate is reasonable but allow negative rates
        initial_rate = max(min(initial_rate, params['max_int_rate']), params['min_int_rate'])  # Allow rates from -10% to 20%
    except:
        # Fallback to a reasonable initial guess
        initial_rate = 0.05
        
    # Optimize the discount rate using the selected objective function
    result = minimize_scalar(
        fun=objective_function,
        bounds=(params['min_int_rate'], params['max_int_rate']),  # Allow rates from -10% to 20%
        method='bounded',
        options={'xatol': 1e-8}
    )
    
    optimal_rate = result.x #type: ignore
    
    # Calculate final IVs with the optimal rate
    put_iv = implied_volatility(
        price=put_price, 
        S=S, 
        K=put_strike, 
        T=T, 
        r=optimal_rate, 
        option_type='put', 
        q=0
        )
    call_iv = implied_volatility(
        price=call_price, 
        S=S, 
        K=call_strike, 
        T=T, 
        r=optimal_rate, 
        option_type='call', 
        q=0
        )
    iv_diff = abs(put_iv - call_iv) if not np.isnan(put_iv) and not np.isnan(call_iv) else np.nan
    
    return optimal_rate, put_iv, call_iv, iv_diff

def apply_interpolation(df_term_structure, df_original, missing_expiries):
    """
    Apply interpolation for missing expiries.
    
    Parameters:
    -----------
    df_term_structure : DataFrame
        Term structure DataFrame with already calculated rates
    df_original : DataFrame
        Original options DataFrame with all expiry information
    missing_expiries : list or set
        Expiry dates that need interpolation
    debug : bool
        Whether to print debug information
    
    Returns:
    --------
    DataFrame : Updated term structure with interpolated values
    """
    
    if not df_term_structure.empty and missing_expiries:
        # Get days and years for each missing expiry
        days_lookup = {}
        years_lookup = {}
        
        for expiry in missing_expiries:
            # Get expiry info from original data
            expiry_days = df_original[df_original['Expiry'] == expiry]['Days To Expiry'].iloc[0]
            expiry_years = df_original[df_original['Expiry'] == expiry]['Years To Expiry'].iloc[0]
            days_lookup[expiry] = expiry_days
            years_lookup[expiry] = expiry_years
        
        # Apply appropriate interpolation/extrapolation for each missing expiry
        for expiry in missing_expiries:
            days = days_lookup[expiry]
            years = years_lookup[expiry]
            
            if days < df_term_structure['Days'].min():
                # Extrapolate for early dates
                df_term_structure = extrapolate_early(df_term_structure, expiry, days, years)
            elif days > df_term_structure['Days'].max():
                # Extrapolate for later dates
                df_term_structure = extrapolate_late(df_term_structure, expiry, days, years)
            else:
                # Interpolate for middle dates
                df_term_structure = interpolate_rate(df_term_structure, expiry, days, years)
    
    return df_term_structure
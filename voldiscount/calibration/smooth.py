"""
Smooth curve calibration for discount rates using a parametric model.

This module implements a Nelson-Siegel curve-fitting approach to create
a consistent term structure of discount rates across multiple expiries.
"""
import numpy as np
import pandas as pd
import time
import calendar
from scipy.optimize import minimize, minimize_scalar, OptimizeResult
from voldiscount.calibration.pair_selection import select_option_pairs
from voldiscount.config.config import DEFAULT_PARAMS
from voldiscount.core.black_scholes import implied_volatility
from typing import Dict, Any, List, Tuple


def nelson_siegel(
    t: float, 
    beta0: float, 
    beta1: float, 
    beta2: float, 
    tau: float
) -> float:
    """
    Nelson-Siegel parametric model for yield curves.
    
    Parameters:
    -----------
    t : float
        Time to maturity in years
    beta0 : float
        Long-term level parameter
    beta1 : float
        Short-term component parameter
    beta2 : float
        Medium-term component parameter
    tau : float
        Decay factor parameter
        
    Returns:
    --------
    float : Interest rate at time t
    """
    if tau <= 0:
        # Avoid division by zero
        tau = 0.0001
        
    # Calculate Nelson-Siegel factors
    factor = (1 - np.exp(-t / tau)) / (t / tau) if t > 0 else 1.0
    factor2 = factor - np.exp(-t / tau)
    
    return beta0 + beta1 * factor + beta2 * factor2


def ns_objective_function(
    params: List[float], 
    pairs_by_expiry: Dict, 
    S: float
) -> float:
    """
    Objective function for global optimization of the Nelson-Siegel curve.
    
    Parameters:
    -----------
    params : List[float]
        Parameters of the Nelson-Siegel model [beta0, beta1, beta2, tau]
    pairs_by_expiry : Dict
        Dictionary mapping expiry dates to lists of option pairs
    S : float
        Underlying price
        
    Returns:
    --------
    float : Sum of squared implied volatility differences
    """
    beta0, beta1, beta2, tau = params
    
    total_error = 0.0
    pair_weights = []
    pair_errors = []
    
    for expiry, pairs in pairs_by_expiry.items():
        for pair in pairs:
            t = pair['years']
            put_strike = pair['put_strike']
            call_strike = pair['call_strike']
            put_price = pair['put_price']
            call_price = pair['call_price']
            
            # Calculate moneyness - weight ATM pairs more heavily
            strike_mid = (put_strike + call_strike) / 2
            moneyness_weight = 1.0 / (1.0 + abs(strike_mid / S - 1.0) * 10)
            
            # Weight exact strike matches more heavily
            strike_match_weight = 5.0 if abs(put_strike - call_strike) < 0.01 else 1.0
            
            # Calculate rate from Nelson-Siegel model
            rate = nelson_siegel(t, beta0, beta1, beta2, tau)
            
            # Skip extreme rates
            if rate < -0.1 or rate > 0.3:
                continue
                
            # Calculate implied volatilities using this rate
            try:
                put_iv = implied_volatility(
                    price=put_price, 
                    S=S, 
                    K=put_strike, 
                    T=t, 
                    r=rate, 
                    option_type='put', 
                    q=0
                )
                
                call_iv = implied_volatility(
                    price=call_price, 
                    S=S, 
                    K=call_strike, 
                    T=t, 
                    r=rate, 
                    option_type='call', 
                    q=0
                )
                
                # Skip invalid IVs
                if np.isnan(put_iv) or np.isnan(call_iv):
                    continue
                    
                # Calculate weighted error
                iv_diff = put_iv - call_iv
                weight = moneyness_weight * strike_match_weight
                error = (iv_diff ** 2) * weight
                
                pair_weights.append(weight)
                pair_errors.append(error)
                
            except Exception:
                continue
    
    # Calculate weighted average error
    if len(pair_weights) > 0:
        total_error = sum(pair_errors) / sum(pair_weights)
    else:
        total_error = 1.0  # Penalty for no valid pairs
        
    return total_error


def smooth_curve_calibration(
    df: pd.DataFrame, 
    S: float, 
    **kwargs
) -> pd.DataFrame:
    """
    Smooth curve calibration using a two-step approach:
    
    1. Calculate optimal discount rate for top 5 ATM pairs per tenor and average them
    2. Fit Nelson-Siegel curve to these robust tenor-specific rates
    """
    
    # Standard parameter handling
    params = DEFAULT_PARAMS.copy()
    params.update(kwargs)
    
    print("Performing smooth curve calibration using two-step Nelson-Siegel approach")
    
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
    
    # Find all valid pairs (don't limit to best_pair_only)
    pairs_by_expiry = select_option_pairs(
        df=df, 
        S=S, 
        forward_prices=None, 
        max_strike_diff_pct=params['max_strike_diff_pct'], 
        min_option_price=params['min_option_price'], 
        consider_volume=params['consider_volume'], 
        min_pair_volume=params['min_pair_volume'], 
        debug=params['debug'], 
        best_pair_only=False  # Use multiple pairs per expiry for better fitting
    )
    
    if not pairs_by_expiry:
        print("ERROR: No valid put-call pairs found. Cannot calibrate.")
        return pd.DataFrame()
    
    # Count total pairs for diagnostics
    total_pairs = sum(len(pairs) for pairs in pairs_by_expiry.values())
    print(f"Found {total_pairs} valid option pairs across {len(pairs_by_expiry)} expiries")
    
    # Step 1: Calculate optimal discount rate for each tenor
    start_time = time.time()
    tenor_rates = []
    
    print("Step 1: Calculating optimal discount rates per tenor...")
    
    for expiry, pairs in pairs_by_expiry.items():
        # Sort pairs by ATM-ness
        sorted_pairs = sorted(pairs, key=lambda p: abs((p['put_strike'] + p['call_strike'])/(2*S) - 1.0))
        
        # Take top 5 most ATM pairs (or fewer if not available)
        top_pairs = sorted_pairs[:min(5, len(sorted_pairs))]
        
        # Skip if no valid pairs
        if not top_pairs:
            continue
        
        # Calculate optimal discount rate for each pair
        pair_rates = []
        
        for pair in top_pairs:
            years: float = pair['years']
            put_price: float = pair['put_price']
            call_price: float = pair['call_price']
            put_strike: float = pair['put_strike']
            call_strike: float = pair['call_strike']
           
            # Define objective function that finds rate where IV_put = IV_call
            def iv_diff_objective(rate):
                try:
                    put_iv = implied_volatility(
                        price=put_price, S=S, K=put_strike, T=years, r=rate,
                        option_type='put', q=0
                    ) 
                    
                    call_iv = implied_volatility(
                        price=call_price, S=S, K=call_strike, T=years, r=rate, option_type='call', q=0
                    )
                    
                    if np.isnan(put_iv) or np.isnan(call_iv):
                        return 1.0  # Penalty for invalid IVs
                    
                    return (put_iv - call_iv) ** 2
                    
                except Exception:
                    return 1.0  # Penalty for calculation errors
            
            # Find optimal rate for this pair using minimize_scalar
            try:                 
                result = minimize_scalar(
                    iv_diff_objective,
                    bounds=(-0.1, 0.15),
                    method='bounded',
                    options={'xatol': 1e-5}
                )
                
                if result.success and -0.1 <= result.x <= 0.15: #type: ignore
                    pair_rates.append(result.x) #type: ignore 
            except Exception as e:
                print(f"  Error calculating rate for pair {put_strike}/{call_strike}: {e}")
                continue
        
        # Calculate average rate for this tenor if we have valid rates
        if pair_rates:
            avg_rate = sum(pair_rates) / len(pair_rates)
            days = top_pairs[0]['days']
            years = top_pairs[0]['years']
            
            tenor_rates.append({
                'expiry': expiry,
                'days': days,
                'years': years,
                'rate': avg_rate
            })
            
            print(f"  Tenor {expiry} ({days} days): Averaged {len(pair_rates)} rates to {avg_rate:.6f}")
    
    step1_time = time.time() - start_time
    print(f"Step 1 completed in {step1_time:.2f} seconds, found {len(tenor_rates)} valid tenor rates")
    
    # Check if we have enough tenor rates for curve fitting
    if len(tenor_rates) < 4:
        print(f"ERROR: Not enough valid tenor rates ({len(tenor_rates)}) for curve fitting")
        print("Need at least 4 points to fit Nelson-Siegel curve")
        return pd.DataFrame()
    
    # Step 2: Fit Nelson-Siegel curve to tenor rates
    print("Step 2: Fitting Nelson-Siegel curve to tenor rates...")
    start_time = time.time()
    
    # Convert to arrays for fitting
    years_array = np.array([t['years'] for t in tenor_rates])
    rates_array = np.array([t['rate'] for t in tenor_rates])
    days_array = np.array([t['days'] for t in tenor_rates])
    expiries = [t['expiry'] for t in tenor_rates]
    
    # Print tenor data before fitting
    print("Tenor data for curve fitting:")
    for i, tenor in enumerate(tenor_rates):
        print(f"  {i+1}. {tenor['expiry']} ({tenor['days']} days): {tenor['rate']:.6f}")
    
    # Define curve fitting objective function
    def curve_fit_objective(params):
        beta0, beta1, beta2, tau = params
        
        if tau <= 0:
            return 1.0e10  # Large penalty for invalid tau
        
        predicted_rates = np.array([nelson_siegel(t, beta0, beta1, beta2, tau) for t in years_array])
        
        # Mean squared error between predicted and observed rates
        mse = np.mean((predicted_rates - rates_array) ** 2)
        return mse
    
    # Initial parameters and bounds
    initial_params = [np.mean(rates_array), 0.0, 0.0, 1.0]
    bounds = [(0.0, 0.1), (-0.05, 0.05), (-0.05, 0.05), (0.5, 3.0)]
    
    print(f"Initial parameters: beta0={initial_params[0]:.6f}, beta1={initial_params[1]:.6f}, " + 
          f"beta2={initial_params[2]:.6f}, tau={initial_params[3]:.6f}")
    
    # Run optimization to fit curve
    result = minimize(
        curve_fit_objective,
        initial_params,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 100}
    )
    
    step2_time = time.time() - start_time
    
    # Extract fitted parameters
    if result.success:
        beta0, beta1, beta2, tau = result.x
        print(f"Curve fitting successful in {step2_time:.2f} seconds")
        print(f"Fitted NS parameters: beta0={beta0:.6f}, beta1={beta1:.6f}, beta2={beta2:.6f}, tau={tau:.6f}")
    else:
        print(f"WARNING: Curve fitting failed: {result.message}")
        print("Using average rate as fallback solution")
        # Simple fallback: flat curve at average rate
        beta0 = np.mean(rates_array)
        beta1, beta2, tau = 0.0, 0.0, 1.0
    
    # Generate term structure for ALL expiries in the original dataset
    term_structure = []

    # Get all unique expiries from original dataset
    all_expiries = sorted(df['Expiry'].unique())
    print(f"Generating term structure for all {len(all_expiries)} expiries in dataset")

    for expiry in all_expiries:
        # Get expiry details from original data
        expiry_df = df[df['Expiry'] == expiry]
        if expiry_df.empty:
            continue
            
        days = expiry_df['Days To Expiry'].iloc[0]
        years = expiry_df['Years To Expiry'].iloc[0]
        
        # Calculate rate using fitted Nelson-Siegel parameters
        rate = nelson_siegel(years, beta0, beta1, beta2, tau) #type: ignore
        
        # Check if we have valid pairs for this expiry
        if expiry in pairs_by_expiry and pairs_by_expiry[expiry]:
            # Find most ATM pair
            pairs = pairs_by_expiry[expiry]
            atm_idx = min(range(len(pairs)), key=lambda i: 
                abs((pairs[i]['put_strike'] + pairs[i]['call_strike']) / (2 * S) - 1.0))
            atm_pair = pairs[atm_idx]
            
            # Extract parameters for term structure
            put_strike = atm_pair['put_strike']
            call_strike = atm_pair['call_strike']
            put_price = atm_pair['put_price']
            call_price = atm_pair['call_price']
            
            # Calculate IVs and forward price using put-call parity
            try:
                put_iv = implied_volatility(
                    price=put_price, S=S, K=put_strike, T=years, 
                    r=rate, option_type='put', q=0
                )
                call_iv = implied_volatility(
                    price=call_price, S=S, K=call_strike, T=years, 
                    r=rate, option_type='call', q=0
                )
                iv_diff = abs(put_iv - call_iv) if not np.isnan(put_iv) and not np.isnan(call_iv) else np.nan
                
                # Calculate forward price from put-call parity
                avg_strike = (put_strike + call_strike) / 2
                discount_factor = np.exp(-rate * years)
                forward_price = avg_strike + (call_price - put_price) / discount_factor
                forward_ratio = forward_price / S
            except Exception as e:
                # If IV calculation fails, use theoretical forward price
                print(f"Error calculating IVs for {expiry}: {e}")
                put_iv, call_iv, iv_diff = None, None, None
                forward_price = S * np.exp(rate * years)
                forward_ratio = forward_price / S
        else:
            # No valid pairs for this expiry, use theoretical calculations
            put_strike, call_strike, put_price, call_price = None, None, None, None #type: ignore
            put_iv, call_iv, iv_diff = None, None, None
            
            # Calculate theoretical forward price using risk-neutral pricing
            forward_price = S * np.exp(rate * years)
            forward_ratio = forward_price / S
            print(f"Expiry {expiry}: Using theoretical forward price (no valid pairs)")
        
        # Add to term structure
        term_structure.append({
            'Expiry': expiry,
            'Days': days,
            'Years': round(years, 4),
            'Discount Rate': round(rate, 4),
            'Put Strike': put_strike,
            'Call Strike': call_strike,
            'Put Price': put_price,
            'Call Price': call_price,
            'Put Implied Volatility': round(put_iv, 4) if put_iv is not None and not np.isnan(put_iv) else None,
            'Call Implied Volatility': round(call_iv, 4) if call_iv is not None and not np.isnan(call_iv) else None,
            'Implied Volatility Diff': round(iv_diff, 4) if iv_diff is not None and not np.isnan(iv_diff) else None,
            'Forward Price': round(forward_price, 4),
            'Forward Ratio': round(forward_ratio, 4),
            'Method': 'smooth_curve'
        })
    
    # Convert to DataFrame and sort
    df_term_structure = pd.DataFrame(term_structure).sort_values('Days')
    
    print(f"Smooth curve calibration created term structure with {len(df_term_structure)} expiries")
    
    return df_term_structure


def vasicek_model(
    t: float, 
    r0: float, 
    k: float, 
    theta: float, 
    sigma: float
) -> float:
    """
    Vasicek short-rate model for yield curves.
    
    Parameters:
    -----------
    t : float
        Time to maturity in years
    r0 : float
        Current short rate
    k : float
        Mean reversion speed
    theta : float
        Long-term mean rate
    sigma : float
        Volatility of the short rate
        
    Returns:
    --------
    float : Zero-coupon rate at time t
    """
    if t <= 0:
        return r0
        
    B = (1 - np.exp(-k * t)) / k
    A = np.exp((theta - sigma**2/(2*k**2)) * (B - t) - (sigma**2 * B**2) / (4 * k))
    
    # Zero-coupon bond price
    P = A * np.exp(-B * r0)
    
    # Convert to continuously compounded rate
    return -np.log(P) / t

"""
Black-Scholes model and implied volatility calculations
"""

from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize_scalar

from voldiscount.vol_params import DEFAULT_PARAMS

# pylint: disable=invalid-name
# pylint: disable=R0913, R0917

class Models():
    """
    Black-Scholes model and implied volatility calculations
    """
    @staticmethod
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
        if not all(isinstance(param, (int, float))
                   for param in [S, K, T, r, sigma, params['q']]):
            return np.nan

        if sigma <= 0 or T <= 0 or S <= 0 or K <= 0:
            return np.nan

        # Handle potential numerical issues
        try:
            d1 = ((np.log(S / K) + (r - params['q'] + 0.5 * sigma**2) * T)
                  / (sigma * np.sqrt(T)))
            d2 = d1 - sigma * np.sqrt(T)

            if params['option_type'].lower() == 'call':
                price = (S * np.exp(-params['q'] * T) * norm.cdf(d1)
                         - K * np.exp(-r * T) * norm.cdf(d2))
            else:  # put
                price = (K * np.exp(-r * T) * norm.cdf(-d2) -
                         S * np.exp(-params['q'] * T) * norm.cdf(-d1))

            return price
        except (TypeError, ValueError, ZeroDivisionError, OverflowError, RuntimeWarning) as e:
            print(f"Returning NaN due to: {str(e)}")
            return np.nan


    @classmethod
    def implied_volatility(
        cls,
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
            bs_price = cls.black_scholes(
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
                mid_price = cls.black_scholes(
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
            if (result.success and params['volatility_lower_bound'] #type: ignore
                <= result.x <= params['volatility_upper_bound']): #type: ignore
                return result.x #type: ignore
            return np.nan

        except (TypeError, ValueError, ZeroDivisionError, OverflowError, RuntimeWarning) as e:
            print(f"Returning NaN due to: {str(e)}")
            return np.nan


    @classmethod
    def calculate_forward_prices(
        cls,
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

            exact_pairs = cls._calc_strike_matches(
                puts=puts,
                calls=calls,
                params=params,
                S=S)

            # Calculate forward using the best available pairs
            if exact_pairs:
                forward = cls._calc_forward(
                    exact_pairs=exact_pairs,
                    params=params,
                    years=years,
                    S=S)
            else:
                # No exact pairs, use simple growth model
                forward = S * (1 + params['fallback_growth']) ** years

            forward_prices[expiry] = forward

            print(f"Expiry: {expiry} ({years:.2f} years)")
            print(f"  Spot: {S:.2f}, Forward: {forward:.2f}, Ratio: {forward/S:.4f}")

        return forward_prices


    @staticmethod
    def _calc_strike_matches(
        puts,
        calls,
        params,
        S) -> List:

        # Find strike pairs for exact matches with high liquidity
        exact_pairs = []

        # Check if we have Volume or Open Interest columns
        has_volume = 'Volume' in puts.columns and 'Volume' in calls.columns
        has_open_interest = ('Open Interest' in puts.columns
                             and 'Open Interest' in calls.columns)

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
            if (put_row['Last Price'] <= params['min_price']
                or call_row['Last Price'] <= params['min_price']):
                continue


            liquidity_params = {}

            # Calculate moneyness
            liquidity_params['strike_float'] = float(strike)
            liquidity_params['moneyness'] = abs(
                liquidity_params['strike_float'] / S - 1.0)


            # Calculate liquidity score based on volume and open interest if available
            liquidity_params['liquidity_score'] = 0

            if has_volume:
                liquidity_params['put_vol'] = (float(put_row['Volume'])
                           if not pd.isna(put_row['Volume']) else 0)
                liquidity_params['call_vol'] = (float(call_row['Volume'])
                            if not pd.isna(call_row['Volume']) else 0)
                liquidity_params['liquidity_score'] += (liquidity_params['put_vol']
                                                        + liquidity_params['call_vol'])

            if has_open_interest:
                liquidity_params['put_oi'] = (float(put_row['Open Interest'])
                          if not pd.isna(put_row['Open Interest']) else 0)
                liquidity_params['call_oi'] = (float(call_row['Open Interest'])
                           if not pd.isna(call_row['Open Interest']) else 0)
                liquidity_params['liquidity_score'] += (liquidity_params['put_oi']
                                                        + liquidity_params['call_oi'])

            # If no volume/OI data, use inverse of moneyness as proxy for liquidity
            if liquidity_params['liquidity_score'] == 0:
                liquidity_params['liquidity_score'] = 1.0 / (
                    liquidity_params['moneyness'] + 0.01)

            exact_pairs.append({
                'strike': liquidity_params['strike_float'],
                'put_price': float(put_row['Last Price']),
                'call_price': float(call_row['Last Price']),
                'moneyness': liquidity_params['moneyness'],
                'liquidity_score': liquidity_params['liquidity_score']
            })

        return exact_pairs


    @staticmethod
    def _calc_forward(exact_pairs, params, years, S):
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
            if (params['min_forward_ratio'] * S <
                forward_est < params['max_forward_ratio'] * S):
                forward_estimates.append(forward_est)

        # Calculate weighted average if we have estimates
        if forward_estimates:
            forward = sum(forward_estimates) / len(forward_estimates)
        else:
            # Fallback to simple growth model
            forward = S * (1 + params['fallback_growth']) ** years

        return forward


    @staticmethod
    def standardize_datetime(
        df: pd.DataFrame,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Standardize datetime columns in a DataFrame to be timezone-naive.

        Parameters:
        -----------
        df : pandas.DataFrame
            Input DataFrame
        columns : list, optional
            List of column names to standardize.
            If None, attempts to standardize all datetime columns.

        Returns:
        --------
        pandas.DataFrame
            DataFrame with standardized datetime columns
        """
        # Create a copy to avoid modifying the original DataFrame
        df = df.copy()

        # If no columns specified, find datetime columns
        if columns is None:
            columns = df.select_dtypes(include=['datetime64']).columns #type: ignore

        for col in columns: #type: ignore
            if col in df.columns:
                try:
                    # Convert to timezone-naive, preserving local time
                    df[col] = pd.to_datetime(df[col], utc=False).dt.tz_localize(None)
                except TypeError:
                    # Handle columns that might already be timezone-naive
                    df[col] = pd.to_datetime(df[col], utc=False)

        return df

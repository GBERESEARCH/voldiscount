"""
Direct: Calculate discount rates directly from put-call parity for each
expiry using an ATM representative pair.

Smooth: Curve calibration for discount rates using a parametric model.
This module implements a Nelson-Siegel curve-fitting approach to create
a consistent term structure of discount rates across multiple expiries.

"""
import copy
import time
from typing import Dict, Any, Tuple, Set

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

from voldiscount.interpolation import Interpolation
from voldiscount.models import Models
from voldiscount.pair_selection import PairSelection

standardize_datetime = Models.standardize_datetime

interpolate_rate = Interpolation.interpolate_rate
extrapolate_early = Interpolation.extrapolate_early
extrapolate_late = Interpolation.extrapolate_late

implied_volatility = Models.implied_volatility

# pylint: disable=invalid-name


class Calibration():
    """
    Direct: Calculate discount rates directly from put-call parity for each
    expiry using an ATM representative pair.

    Smooth: Curve calibration for discount rates using a parametric model.
    This module implements a Nelson-Siegel curve-fitting approach to create
    a consistent term structure of discount rates across multiple expiries.

    """

    @classmethod
    def calibrate_rates(
        cls,
        tables: Dict[str, Any],
        params: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Calibrate direct and smooth discount rates.

        Parameters:
        -----------
        tables : dict
            Tables dictionary
        params : dict
            Configuration parameters

        Returns:
        --------
        tuple : (dict, dict)
            Updated tables and params
        """
        calibration_start = time.time()

        print("Finding option pairs for calibration...")
        df = copy.deepcopy(tables['source_data'])
        tables['filtered_df'] = cls._filter_df(df=df, params=params)

        params, tables = PairSelection.select_option_pairs_enhanced(
            df=tables['filtered_df'], params=params, tables=tables
        )

        # Run direct calibration
        print("Running direct discount rate calibration...")
        direct_term_structure, params = cls.direct_curve_calibration(
            params=params, tables=tables)

        # Run smooth curve calibration
        print("\nRunning smooth curve calibration...")
        smooth_term_structure, params = cls.smooth_curve_calibration(
            params=params, tables=tables)

        # Standardize datetime in term structures
        if direct_term_structure is not None and not direct_term_structure.empty:
            direct_term_structure = standardize_datetime(
                direct_term_structure, columns=['Expiry'])
            direct_forwards = {
                row['Expiry']: row['Forward Price']
                for _, row in direct_term_structure.iterrows()
                if 'Forward Price' in direct_term_structure.columns
            }
            tables['direct_term_structure'] = direct_term_structure
            tables['direct_forwards'] = direct_forwards

        if smooth_term_structure is not None and not smooth_term_structure.empty:
            smooth_term_structure = standardize_datetime(
                smooth_term_structure, columns=['Expiry'])
            smooth_forwards = {
                row['Expiry']: row['Forward Price']
                for _, row in smooth_term_structure.iterrows()
                if 'Forward Price' in smooth_term_structure.columns
            }
            tables['smooth_term_structure'] = smooth_term_structure
            tables['smooth_forwards'] = smooth_forwards

        params['timings']['calibration'] = time.time() - calibration_start

        # Print calibration results
        cls._print_calibration_results(tables)

        return tables, params


    @classmethod
    def direct_curve_calibration(
        cls,
        params: Dict[str, Any],
        tables: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Calculate discount rates directly from put-call parity for each expiry
        using an ATM representative pair.

        Parameters:
        -----------
        df : DataFrame
            Options data with put and call prices
        S : float
            Underlying price
        max_strike_diff_pct : float
            Maximum allowed difference between put and call strikes as
            percentage of S
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
            If True, only include standard monthly expiries (3rd Friday
                                                             of each month)
        """

        print("Performing direct discount rate calibration with ATM representative pairs")

        if not tables['pairs_by_expiry']:
            print("ERROR: No valid put-call pairs found. Cannot calibrate.")
            return pd.DataFrame(), params

        term_structure = cls._calc_direct_term_structure(
            pairs_by_expiry=tables['pairs_by_expiry'], params=params)

        # Convert to DataFrame and sort
        df_term_structure = pd.DataFrame(term_structure).sort_values('Days')

        print(f"Direct calibration created term structure with "
              f"{len(df_term_structure)} expiries")

        # After creating the initial term_structure
        if not df_term_structure.empty:
            # Find all unique expiries in original data
            all_expiries = set(tables['filtered_df']['Expiry'].unique())

            # Find expiries we have rates for
            calculated_expiries = set(df_term_structure['Expiry'].unique())

            # Determine missing expiries
            missing_expiries = all_expiries - calculated_expiries

            if missing_expiries:
                print(f"Interpolating rates for {len(missing_expiries)} "
                      f"missing expiries")
                df_term_structure, params = cls._apply_interpolation(
                    df_term_structure=df_term_structure, df_original=tables['filtered_df'],
                    missing_expiries=missing_expiries, params=params)

        return df_term_structure, params


    @classmethod
    def _calc_direct_term_structure(
            cls,
            pairs_by_expiry,
            params: Dict[str, Any]
            ):
        term_structure = []

        for expiry, pairs in pairs_by_expiry.items():
            if len(pairs) < params['min_options_per_expiry']:
                print(f"Skipping expiry {expiry}: only {len(pairs)} pairs, "
                      f"need at least {params['min_options_per_expiry']}")
                continue

            # First try exact matches only
            exact_pairs = [p for p in pairs if p['is_exact_match']]
            
            if exact_pairs:
                # Find most ATM among exact matches
                atm_idx = min(range(len(exact_pairs)), key=lambda i:
                    abs(exact_pairs[i]['put_strike'] / params['underlying_price'] - 1.0))
                atm_pair = exact_pairs[atm_idx]
            else:
                # No exact matches - skip this tenor for direct method
                print(f"No exact strike matches for {expiry} - will interpolate")
                continue    

            # Calculate rate for this pair
            try:
                opt_params = cls._optimize_discount_rate(
                    atm_pair=atm_pair,S=params['underlying_price'], params=params)

                # Calculate forward price directly from put-call parity
                # Forward = Strike + (Call - Put) * e^(r*T)
                avg_strike = (atm_pair['put_strike'] + atm_pair['call_strike']) / 2
                discount_factor = np.exp(-opt_params['optimal_rate'] * atm_pair['years'])
                forward_price = (avg_strike + (atm_pair['call_price']
                                               - atm_pair['put_price']) / discount_factor)
                forward_ratio = forward_price / params['underlying_price']

                term_structure.append({
                    'Expiry': expiry,
                    'Days': atm_pair['days'],
                    'Years': round(atm_pair['years'], 4),
                    'Discount Rate': round(opt_params['optimal_rate'], 4),
                    'Put Strike': atm_pair['put_strike'],
                    'Call Strike': atm_pair['call_strike'],
                    'Put Price': atm_pair['put_price'],
                    'Call Price': atm_pair['call_price'],
                    'Put Implied Vol': round(opt_params['put_iv'], 4),
                    'Call Implied Vol': round(opt_params['call_iv'], 4),
                    'Implied Vol Diff': round(opt_params['iv_diff'], 4),
                    'Forward Price': round(forward_price, 4),
                    'Forward Ratio': round(forward_ratio, 4)
                })

            except (ValueError, ZeroDivisionError, OverflowError, RuntimeWarning) as e:
                print(f"ERROR calculating rate for expiry {expiry}: {e}")

        return term_structure


    @staticmethod
    def _optimize_discount_rate(
        atm_pair,
        S: float,
        params: Dict[str, Any]
    ) -> Dict:
        """
        Optimize the discount rate to match implied volatilities or satisfy
        put-call parity.
        """
        opt_params = {}
        opt_params['reference_price'] = S
        opt_params['strikes_equal'] = abs(
            atm_pair['put_strike'] - atm_pair['call_strike']) < 0.01

        # Define objective functions with unique names
        def objective_equal_strikes(rate):
            # Calculate IVs with the given discount rate for equal strikes
            put_iv = implied_volatility(
                price=atm_pair['put_price'],
                S=S,
                K=atm_pair['put_strike'],
                T=atm_pair['years'],
                r=rate,
                option_type='put',
                q=0
                )
            call_iv = implied_volatility(
                price=atm_pair['call_price'],
                S=S,
                K=atm_pair['call_strike'],
                T=atm_pair['years'],
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
            K_avg = (atm_pair['put_strike'] + atm_pair['call_strike']) / 2

            # Calculate what the call price should be using put-call parity
            synthetic_call = (atm_pair['put_price'] + opt_params['reference_price']
                              - K_avg * np.exp(-rate * atm_pair['years']))

            # Calculate what the put price should be using put-call parity
            synthetic_put = (atm_pair['call_price'] - opt_params['reference_price']
                             + K_avg * np.exp(-rate * atm_pair['years']))

            # We want to minimize the relative pricing error
            call_error = abs(
                synthetic_call - atm_pair['call_price']) / atm_pair['call_price']
            put_error = abs(
                synthetic_put - atm_pair['put_price']) / atm_pair['put_price']

            return call_error + put_error

        # Select the appropriate objective function based on whether strikes are equal
        objective_function = (
            objective_equal_strikes
            if opt_params['strikes_equal']
            else objective_different_strikes
            )

        # Initial guess based on direct calculation from put-call parity
        try:
            K_avg = (atm_pair['put_strike'] + atm_pair['call_strike']) / 2
            forward_price = atm_pair['call_price'] - atm_pair['put_price'] + K_avg
            initial_rate = (-np.log(forward_price / opt_params['reference_price'])
                            / atm_pair['years'])

            # Ensure initial_rate is reasonable but allow negative rates
            initial_rate = max(min(initial_rate, params['max_int_rate']),
                               params['min_int_rate'])  # Allow rates from -10% to 20%
        except (ValueError, ZeroDivisionError, OverflowError, RuntimeWarning):
            # Fallback to a reasonable initial guess
            initial_rate = 0.05

        # Optimize the discount rate using the selected objective function
        result = minimize_scalar(
            fun=objective_function,
            bounds=(params['min_int_rate'], params['max_int_rate']),  # Allow rates from -10% to 20%
            method='bounded',
            options={'xatol': 1e-8}
        )

        opt_params['optimal_rate'] = result.x #type: ignore

        # Calculate final IVs with the optimal rate
        opt_params['put_iv'] = implied_volatility(
            price=atm_pair['put_price'],
            S=S,
            K=atm_pair['put_strike'],
            T=atm_pair['years'],
            r=opt_params['optimal_rate'],
            option_type='put',
            q=0
            )
        opt_params['call_iv'] = implied_volatility(
            price=atm_pair['call_price'],
            S=S,
            K=atm_pair['call_strike'],
            T=atm_pair['years'],
            r=opt_params['optimal_rate'],
            option_type='call',
            q=0
            )
        opt_params['iv_diff'] = (
            abs(opt_params['put_iv'] - opt_params['call_iv'])
            if not np.isnan(opt_params['put_iv'])
            and not np.isnan(opt_params['call_iv']) else np.nan
            )

        return opt_params


    @staticmethod
    def _apply_interpolation(
        df_term_structure: pd.DataFrame,
        df_original: pd.DataFrame,
        missing_expiries: Set[pd.Timestamp],
        params: Dict[str, Any]
    ) -> tuple[pd.DataFrame, Dict[str, Any]]:
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
        params : Dict
            Parameter dictionary
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
                expiry_days = df_original[
                    df_original['Expiry'] == expiry]['Days To Expiry'].iloc[0]
                expiry_years = df_original[
                    df_original['Expiry'] == expiry]['Years To Expiry'].iloc[0]
                days_lookup[expiry] = expiry_days
                years_lookup[expiry] = expiry_years

            # Apply appropriate interpolation/extrapolation for each missing expiry
            for expiry in missing_expiries:
                days = days_lookup[expiry]
                years = years_lookup[expiry]

                if days < df_term_structure['Days'].min():
                    # Extrapolate for early dates
                    df_term_structure, params = extrapolate_early(
                        df_term_structure, expiry, days, years, params=params)
                elif days > df_term_structure['Days'].max():
                    # Extrapolate for later dates
                    df_term_structure, params = extrapolate_late(
                        df_term_structure, expiry, days, years, params=params)
                else:
                    # Interpolate for middle dates
                    df_term_structure, params = interpolate_rate(
                        df_term_structure, expiry, days, years, params=params)

            df_term_structure = df_term_structure.sort_values('Days').reset_index(drop=True)

        return df_term_structure, params


    @classmethod
    def smooth_curve_calibration(
        cls,
        params: Dict[str, Any],
        tables: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Smooth curve calibration using a two-step approach:

        1. Calculate optimal discount rate for top 5 ATM pairs per tenor and average them
        2. Fit Nelson-Siegel curve to these robust tenor-specific rates
        """

        print("Performing smooth curve calibration using two-step Nelson-Siegel approach")

        if not tables['pairs_by_expiry']:
            print("ERROR: No valid put-call pairs found. Cannot calibrate.")
            return pd.DataFrame(), params

        # Count total pairs for diagnostics
        total_pairs = sum(len(pairs) for pairs in tables['pairs_by_expiry'].values())
        print(f"Found {total_pairs} valid option pairs across "
              f"{len(tables['pairs_by_expiry'])} expiries")

        time_data = {}
        # Step 1: Calculate optimal discount rate for each tenor
        time_data['start_time'] = time.time()

        print("Step 1: Calculating optimal discount rates per tenor...")

        tenor_rates = cls._calc_smooth_rates(
            pairs_by_expiry=tables['pairs_by_expiry'], 
            S=params['underlying_price'], 
            params=params
            )

        time_data['step1_time'] = time.time() - time_data['start_time']
        print(f"Step 1 completed in {time_data['step1_time']:.2f} seconds, "
              f"found {len(tenor_rates)} valid tenor rates")
        
        if params['exact_strikes']:
            total_expiries = len(tables['pairs_by_expiry'])
            print(f"Exact strikes mode: Found rates for {len(tenor_rates)} of "
                f"{total_expiries} tenors (others will use fitted curve)")

        # Check if we have enough tenor rates for curve fitting
        if len(tenor_rates) < 4:
            print(f"ERROR: Not enough valid tenor rates "
                  f"({len(tenor_rates)}) for curve fitting")
            print("Need at least 4 points to fit Nelson-Siegel curve")
            return pd.DataFrame(), params

        # Step 2: Fit Nelson-Siegel curve to tenor rates
        print("Step 2: Fitting Nelson-Siegel curve to tenor rates...")
        time_data['start_time'] = time.time()

        ns_params = cls._ns_fitting(tenor_rates=tenor_rates, time_data=time_data)

        ns_params['S'] = params['underlying_price']

        # Get all unique expiries from original dataset
        all_expiries = sorted(tables['filtered_df']['Expiry'].unique())
        print(f"Generating term structure for all {len(all_expiries)} "
              f"expiries in dataset")

        term_structure = cls._calc_smooth_term_structure(
            all_expiries=all_expiries,
            pairs_by_expiry=tables['pairs_by_expiry'],
            df=tables['filtered_df'],
            ns_params=ns_params
            )

        # Convert to DataFrame and sort
        df_term_structure = pd.DataFrame(term_structure).sort_values('Days')

        print(f"Smooth curve calibration created term structure with "
              f"{len(df_term_structure)} expiries")

        return df_term_structure, params


    @staticmethod
    def _filter_df(df, params):
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
                print(f"Filtered out {filtered_count} options with trade dates "
                      f"before {params['reference_date']}")
            print(f"Using {len(filtered_df)} options traded on or after "
                  f"{params['reference_date']}")

            # Use filtered dataframe for further processing
            df = filtered_df

        return df


    @staticmethod
    def _calc_smooth_rates(pairs_by_expiry, S, params):
        tenor_rates = []

        for expiry, pairs in pairs_by_expiry.items():
            # Filter for exact matches only if required
            if params['exact_strikes']:
                exact_pairs = [p for p in pairs if p['is_exact_match']]
                if not exact_pairs:
                    continue  # Skip this tenor if no exact matches
                included_pairs = exact_pairs
            else:
                included_pairs = pairs
            # Sort pairs by ATM-ness
            sorted_pairs = sorted(included_pairs, key=lambda p: abs(
                (p['put_strike'] + p['call_strike'])/(2*S) - 1.0))

            # Take top 5 most ATM pairs (or fewer if not available)
            top_pairs = sorted_pairs[:min(5, len(sorted_pairs))]

            # Skip if no valid pairs
            if not top_pairs:
                continue

            # Calculate optimal discount rate for each pair
            pair_rates = []

            for pair in top_pairs:
                # Define objective function that finds rate where IV_put = IV_call
                def iv_diff_objective(
                        rate,
                        _years=pair['years'],
                        _put_price=pair['put_price'],
                        _call_price=pair['call_price'],
                        _put_strike=pair['put_strike'],
                        _call_strike=pair['call_strike']
                        ):
                    try:
                        put_iv = implied_volatility(
                            price=_put_price, S=S, K=_put_strike, T=_years, r=rate,
                            option_type='put', q=0
                        )

                        call_iv = implied_volatility(
                            price=_call_price, S=S, K=_call_strike, T=_years,
                            r=rate, option_type='call', q=0
                        )

                        if np.isnan(put_iv) or np.isnan(call_iv):
                            return 1.0  # Penalty for invalid IVs

                        return (put_iv - call_iv) ** 2

                    except (ValueError, ZeroDivisionError, OverflowError,
                            RuntimeWarning):
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
                except (ValueError, ZeroDivisionError, OverflowError,
                        RuntimeWarning) as e:
                    print(f"  Error calculating rate for pair "
                          f"{pair['put_strike']}/{pair['call_strike']}: {e}")
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

                print(f"  Tenor {expiry} ({days} days): Averaged "
                      f"{len(pair_rates)} rates to {avg_rate:.6f}")

        return tenor_rates


    @classmethod
    def _ns_fitting(cls, tenor_rates, time_data):
        # Convert to arrays for fitting
        arrays = {}
        arrays['years_array'] = np.array([t['years'] for t in tenor_rates])
        arrays['rates_array'] = np.array([t['rate'] for t in tenor_rates])

        # Print tenor data before fitting
        print("Tenor data for curve fitting:")
        for i, tenor in enumerate(tenor_rates):
            print(f"  {i+1}. {tenor['expiry']} ({tenor['days']} days): "
                  f"{tenor['rate']:.6f}")

        # Define curve fitting objective function
        def curve_fit_objective(params):
            beta0, beta1, beta2, tau = params

            if tau <= 0:
                return 1.0e10  # Large penalty for invalid tau

            predicted_rates = np.array([cls._nelson_siegel(
                t, beta0, beta1, beta2, tau) for t in arrays['years_array']])

            # Mean squared error between predicted and observed rates
            mse = np.mean((predicted_rates - arrays['rates_array']) ** 2)
            return mse

        # Initial parameters and bounds
        initial_params = [np.mean(arrays['rates_array']), 0.0, 0.0, 1.0]
        bounds = [(0.0, 0.1), (-0.05, 0.05), (-0.05, 0.05), (0.5, 3.0)]

        print(f"Initial parameters: beta0={initial_params[0]:.6f}, "
              f"beta1={initial_params[1]:.6f}, " +
            f"beta2={initial_params[2]:.6f}, tau={initial_params[3]:.6f}")

        # Run optimization to fit curve
        result = minimize(
            curve_fit_objective,
            initial_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100}
        )

        time_data['step2_time'] = time.time() - time_data['start_time']

        # Extract fitted parameters
        if result.success:
            beta0, beta1, beta2, tau = result.x
            print(f"Curve fitting successful in {time_data['step2_time']:.2f} seconds")
            print(f"Fitted NS parameters: beta0={beta0:.6f}, "
                  f"beta1={beta1:.6f}, beta2={beta2:.6f}, tau={tau:.6f}")
        else:
            print(f"WARNING: Curve fitting failed: {result.message}")
            print("Using average rate as fallback solution")
            # Simple fallback: flat curve at average rate
            beta0 = np.mean(arrays['rates_array'])
            beta1, beta2, tau = 0.0, 0.0, 1.0

        ns_params = {}
        ns_params['beta0'] = beta0
        ns_params['beta1'] = beta1
        ns_params['beta2'] = beta2
        ns_params['tau'] = tau

        return ns_params


    @classmethod
    def _calc_smooth_term_structure(
            cls, all_expiries, pairs_by_expiry, df, ns_params):
        # Generate term structure for ALL expiries in the original dataset
        term_structure = []

        for expiry in all_expiries:
            # Get expiry details from original data
            expiry_df = df[df['Expiry'] == expiry]
            if expiry_df.empty:
                continue

            years = expiry_df['Years To Expiry'].iloc[0]

            # Calculate rate using fitted Nelson-Siegel parameters
            rate = cls._nelson_siegel(
                t=years,
                beta0=ns_params['beta0'],
                beta1=ns_params['beta1'],
                beta2=ns_params['beta2'],
                tau=ns_params['tau'])

            # Check if we have valid pairs for this expiry
            if expiry in pairs_by_expiry and pairs_by_expiry[expiry]:
                # Find most ATM pair
                pairs = pairs_by_expiry[expiry]
                iv_params= {}

                iv_params['atm_idx'] = min(
                    range(len(pairs)), key=lambda i, current_pairs=pairs:
                    abs((current_pairs[i]['put_strike']
                         + current_pairs[i]['call_strike'])
                        / (2 * ns_params['S']) - 1.0))
                atm_pair = pairs[iv_params['atm_idx']]

                # Calculate IVs and forward price using put-call parity
                try:
                    iv_params['put_iv'] = implied_volatility(
                        price=atm_pair['put_price'], S=ns_params['S'],
                        K=atm_pair['put_strike'], T=years,
                        r=rate, option_type='put', q=0
                    )
                    iv_params['call_iv'] = implied_volatility(
                        price=atm_pair['call_price'], S=ns_params['S'],
                        K=atm_pair['call_strike'], T=years,
                        r=rate, option_type='call', q=0
                    )
                    iv_params['iv_diff'] = (
                        abs(iv_params['put_iv'] - iv_params['call_iv'])
                        if not np.isnan(iv_params['put_iv'])
                        and not np.isnan(iv_params['call_iv'])
                        else np.nan)

                    # Calculate forward price from put-call parity
                    iv_params['forward_price'] = (
                        (atm_pair['put_strike'] + atm_pair['call_strike']) / 2
                        + (atm_pair['call_price']
                           - atm_pair['put_price'])
                        / np.exp(-rate * years))
                    iv_params['forward_ratio'] = iv_params['forward_price'] / ns_params['S']
                except (ValueError, ZeroDivisionError, OverflowError,
                        RuntimeWarning) as e:
                    # If IV calculation fails, use theoretical forward price
                    print(f"Error calculating IVs for {expiry}: {e}")
                    iv_params['put_iv'] = None
                    iv_params['call_iv'] = None
                    iv_params['iv_diff'] = None
                    iv_params['forward_price'] = ns_params['S'] * np.exp(rate * years)
                    iv_params['forward_ratio'] = iv_params['forward_price'] / ns_params['S']
            else:
                # No valid pairs for this expiry, use theoretical calculations
                atm_pair['put_strike'] = None
                atm_pair['call_strike'] = None
                atm_pair['put_price'] = None
                atm_pair['call_price'] = None
                iv_params['put_iv'] = None
                iv_params['call_iv'] = None
                iv_params['iv_diff'] = None

                # Calculate theoretical forward price using risk-neutral pricing
                iv_params['forward_price'] = ns_params['S'] * np.exp(rate * years)
                iv_params['forward_ratio'] = iv_params['forward_price'] / ns_params['S']
                print(f"Expiry {expiry}: Using theoretical forward price "
                      f"(no valid pairs)")

            # Add to term structure
            term_structure.append({
                'Expiry': expiry,
                'Days': expiry_df['Days To Expiry'].iloc[0],
                'Years': round(years, 4),
                'Discount Rate': round(rate, 4),
                'Put Strike': atm_pair['put_strike'],
                'Call Strike': atm_pair['call_strike'],
                'Put Price': atm_pair['put_price'],
                'Call Price': atm_pair['call_price'],
                'Put Implied Volatility': (
                    round(iv_params['put_iv'], 4)
                    if iv_params['put_iv'] is not None
                    and not np.isnan(iv_params['put_iv']) else None),
                'Call Implied Volatility': (
                    round(iv_params['call_iv'], 4)
                    if iv_params['call_iv'] is not None
                    and not np.isnan(iv_params['call_iv']) else None),
                'Implied Volatility Diff': (
                    round(iv_params['iv_diff'], 4)
                    if iv_params['iv_diff'] is not None
                    and not np.isnan(iv_params['iv_diff']) else None),
                'Forward Price': round(iv_params['forward_price'], 4),
                'Forward Ratio': round(iv_params['forward_ratio'], 4),
                'Method': 'smooth_curve'
            })

        return term_structure


    @staticmethod
    def _nelson_siegel(
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


    @staticmethod
    def _print_calibration_results(
        tables: Dict[str, Any]
    ) -> None:
        """
        Print calibration results.

        Parameters:
        -----------
        tables : dict
            Tables dictionary containing term structures
        """
        # Print direct term structure
        print("\nDirect Calibration Term Structure:")
        direct_ts = tables.get('direct_term_structure')
        if direct_ts is not None and not direct_ts.empty:
            cols_to_print = [
                'Expiry', 'Days', 'Years', 'Discount Rate', 'Forward Price',
                'Forward Ratio']
            cols_available = [col for col in cols_to_print if col in direct_ts.columns]
            print(direct_ts[cols_available])

            # Also print calibration details
            print("\nOptions Used for Direct Calibration:")
            detail_cols = ['Expiry', 'Put Strike', 'Call Strike', 'Put Price',
                           'Call Price', 'Put Implied Volatility',
                           'Call Implied Volatility', 'Implied Volatility Diff']
            valid_detail_cols = [col for col in detail_cols if col in direct_ts.columns]
            print(direct_ts[valid_detail_cols])
        else:
            print("No valid term structure from direct calibration.")

        # Print smooth term structure
        print("\nSmooth Curve Term Structure:")
        smooth_ts = tables.get('smooth_term_structure')
        if smooth_ts is not None and not smooth_ts.empty:
            cols_to_print = ['Expiry', 'Days', 'Years', 'Discount Rate',
                             'Forward Price', 'Forward Ratio']
            cols_available = [col for col in cols_to_print if col in smooth_ts.columns]
            print(smooth_ts[cols_available])
        else:
            print("No valid term structure from smooth curve calibration.")

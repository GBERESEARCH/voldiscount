"""
Find Put-Call pairs to use for extracting discount rates
"""
import math
from typing import Dict, Any, List, Tuple

import pandas as pd

# pylint: disable=invalid-name

class PairSelection():
    """
    Find Put-Call pairs to use for extracting discount rates
    """

    @classmethod
    def select_option_pairs_enhanced(
        cls,
        df: pd.DataFrame,
        tables: Dict[str, Any],
        params: Dict[str, Any],
    ) -> Tuple[Dict[pd.Timestamp, List[Dict[str, Any]]], Dict[str, Any], Dict[str, Any]]:
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
        pair_selection_method : str
            Method to use for pair selection: 'original' or 'enhanced'
        min_absolute_volume : int
            Minimum acceptable volume for any option
        min_relative_volume_factor : float
            Minimum as fraction of median volume
        max_trade_age_minutes : int
            Maximum age of trades in minutes
        prioritize_recent_trades : bool
            Whether to prioritize recent trades
        prioritize_round_strikes : bool
            Whether to prioritize round strikes
        atm_priority_factor : float
            Weight given to ATM-ness (0.0-1.0)
        secondary_factor_weight : float
            Weight given to secondary factors (liquidity, recency, roundness)
        filter_low_liquidity_pairs : bool
            Remove pairs with insufficient volume
        exclude_stale_options : bool
            Exclude options with trades older than max_trade_age_minutes

        Returns:
        --------
        dict : Dictionary mapping expiry dates to lists of put-call pairs
        """

        price_ref = "forwards" if params['forward_prices'] else "spot price"
        print(f"Finding strike-matched pairs with max diff: "
                f"{params['max_strike_diff_pct']*100:.1f}% of {price_ref}")

        # Check required columns
        required_cols = ['Expiry', 'Strike', 'Option Type', 'Last Price']
        for col in required_cols:
            if col not in df.columns:
                print(f"ERROR: Missing required column '{col}' in options data")
                return {}, params, tables

        # Ensure Option Type is properly formatted
        if not df['Option Type'].str.lower().isin(['call', 'put']).all():
            df['Option Type'] = df['Option Type'].str.lower()

        params['diagnostics'] = {}
        # Add diagnostic counters
        params['diagnostics']['total_expiries'] = 0
        params['diagnostics']['expiries_with_options'] = 0
        params['diagnostics']['expiries_with_min_options'] = 0
        params['diagnostics']['expiries_with_common_strikes'] = 0
        params['diagnostics']['expiries_with_valid_pairs'] = 0

        # Calculate volume statistics if we're considering volume
        params['volume_stats'] = {}
        params['has_volume'] = 'Volume' in df.columns

        if params['consider_volume'] and params['has_volume']:
            params['volume_stats'] = {
                'median': df['Volume'].median(),
                'mean': df['Volume'].mean(),
                'std': df['Volume'].std()
            }
            print(f"Volume statistics - Median: {params['volume_stats']['median']}, "
                    f"Mean: {params['volume_stats']['mean']:.2f}")

        params['reference_time'] = cls._get_reference_time(df=df, params=params)

        # Process each expiry
        pairs_by_expiry = {}

        for expiry, expiry_df in df.groupby('Expiry'):
            puts, calls = cls._enhanced_diagnostics(
                expiry=expiry,
                expiry_df=expiry_df,
                params=params
                )

            # Apply option filtering based on selection method
            all_pairs = cls._get_all_pairs(
                puts=puts,
                calls=calls,
                expiry=expiry,
                params=params
                )

            # Check for common strikes (for diagnostic purposes)
            if any(pair['is_exact_match'] for pair in all_pairs):
                params['diagnostics']['expiries_with_common_strikes'] += 1

            # If we have pairs, add to the results
            if all_pairs:
                # Keep only the best pair or all pairs
                if params['best_pair_only']:
                    pairs_by_expiry[expiry] = [all_pairs[0]]
                else:
                    pairs_by_expiry[expiry] = all_pairs

                params['diagnostics']['expiries_with_valid_pairs'] += 1

                exact_matches = sum(1 for p in all_pairs if p['is_exact_match'])
                print(f"  Found {len(all_pairs)} valid pairs ({exact_matches} exact matches)")
            else:
                print(f"No valid pairs found for expiry {expiry}")

        print("Diagnostics:")
        print(f"  Total expiries: {params['diagnostics']['total_expiries']}")
        print(f"  Expiries with both puts and calls: "
            f"{params['diagnostics']['expiries_with_options']}")
        print(f"  Expiries with min. {params['min_options_per_type']} of each option type: "
            f"{params['diagnostics']['expiries_with_min_options']}")
        print(f"  Expiries with common strikes: "
            f"{params['diagnostics']['expiries_with_common_strikes']}")
        print(f"  Expiries with valid pairs after all filtering: "
            f"{params['diagnostics']['expiries_with_valid_pairs']}")
        print(f"  Selection method: {params['pair_selection_method']}")

        return pairs_by_expiry, params, tables


    @staticmethod
    def _enhanced_diagnostics(expiry, expiry_df, params):
        params['diagnostics']['total_expiries'] += 1
        # Get reference price (spot or forward)
        params['reference_price'] = (
            params['forward_prices'].get(expiry, params['underlying_price'])
            if params['forward_prices'] else params['underlying_price'])

        # Get put and call data
        puts = expiry_df[(expiry_df['Option Type'].str.lower() == 'put') &
                        (expiry_df['Last Price'] > params['min_option_price'])]
        calls = expiry_df[(expiry_df['Option Type'].str.lower() == 'call') &
                        (expiry_df['Last Price'] > params['min_option_price'])]

        print(f"Expiry {expiry}: Initial counts - {len(puts)} puts, "
                f"{len(calls)} calls")

        if puts.empty or calls.empty:
            print(f"Skipping expiry {expiry} - missing puts or calls")

        params['diagnostics']['expiries_with_options'] += 1

        # Check for minimum number of options per type
        if (len(puts) < params['min_options_per_type']
            or len(calls) < params['min_options_per_type']):
            print(f"Skipping expiry {expiry} - insufficient options "
                    f"(need {params['min_options_per_type']} of each type, "
                    f"found {len(puts)} puts, {len(calls)} calls)")


        params['diagnostics']['expiries_with_min_options'] += 1

        return puts, calls


    @classmethod
    def _get_all_pairs(cls, puts, calls, expiry, params):
        if params['pair_selection_method'] == 'enhanced':
            puts, calls = cls._filter_options(
                puts=puts,
                calls=calls,
                params=params
                )

            if puts.empty or calls.empty:
                print(f"Skipping expiry {expiry} after filtering "
                        f"- missing puts or calls")

            if (len(puts) < params['min_options_per_type']
                or len(calls) < params['min_options_per_type']):
                print(f"Skipping expiry {expiry} after filtering "
                        f"- insufficient options")


        if (len(puts) < params['min_options_per_type']
            or len(calls) < params['min_options_per_type']):
            print(f"  FAILURE: Insufficient options after filtering "
                    f"(need {params['min_options_per_type']} of each type)")
        else:
            print(f"  SUCCESS: {len(puts)} puts and {len(calls)} calls "
                    f"remain after filtering")

        # Find option pairs according to the selected method
        if params['pair_selection_method'] == 'enhanced':
            all_pairs, params = cls._find_enhanced_pairs(
                puts=puts,
                calls=calls,
                expiry=expiry,
                params=params
            )
        else:
            all_pairs = cls._find_original_pairs(
                puts=puts,
                calls=calls,
                expiry=expiry,
                params=params
            )

        return all_pairs


    @staticmethod
    def _get_reference_time(df: pd.DataFrame, params: Dict) -> pd.Timestamp:
        """
        Get average last trade time of highest volume strikes.

        Parameters:
        -----------
        df : DataFrame
            Options data with 'Last Trade Date', 'Strike', and 'Volume' columns
        params : dict
            Configuration parameters from DEFAULT_PARAMS

        Returns:
        --------
        pd.Timestamp : Reference time for recency comparisons
        """
        # # Ensure required columns exist
        # if not all(col in df.columns for col in ['Last Trade Date', 'Strike', 'Volume']):
        #     # Fall back to maximum time if columns missing
        #     return pd.to_datetime(df['Last Trade Date']).max()

        # # Number of high volume strikes to consider
        # top_n_strikes = params['reference_time_strikes']

        # # Group by strike and sum volumes
        # strike_volumes = df.groupby('Strike')['Volume'].sum().reset_index()

        # # Get the top N strikes by volume
        # top_strikes = strike_volumes.nlargest(top_n_strikes, 'Volume')['Strike'].values

        # # Filter dataframe to these strikes
        # top_strike_data = df[df['Strike'].isin(top_strikes)]

        # # Get the latest trade for each of these strikes
        # latest_trades = top_strike_data.groupby('Strike')['Last Trade Date'].max()

        # # If we have valid timestamps, calculate average
        # if not latest_trades.empty and all(isinstance(t, pd.Timestamp) for t in latest_trades):
        #     # Convert all timestamps to UNIX timestamp (seconds since epoch)
        #     timestamps = [t.timestamp() for t in latest_trades]

        #     # Calculate average timestamp
        #     avg_timestamp = sum(timestamps) / len(timestamps)

        #     # Convert back to pandas Timestamp
        #     return pd.Timestamp.fromtimestamp(avg_timestamp)

        # # Fallback to maximum time if calculation fails
        # return pd.to_datetime(df['Last Trade Date']).max()

        print("\n_get_reference_time diagnostics:")
        print(f"Total options in dataset: {len(df)}")

        # Number of high volume strikes to consider
        print(f"Using top {params['reference_time_strikes']} strikes by volume")

        # Group by strike and sum volumes
        strike_volumes = df.groupby('Strike')['Volume'].sum().reset_index()
        print(f"Unique strikes in dataset: {len(strike_volumes)}")

        # Get the top N strikes by volume
        top_volume_strikes = strike_volumes.nlargest(
            params['reference_time_strikes'], 'Volume')
        print(f"Top {len(top_volume_strikes)} strikes by volume:")
        for _, row in top_volume_strikes.iterrows():
            print(f"  Strike: {row['Strike']}, Volume: {row['Volume']}")

        # Filter dataframe to these strikes
        top_strike_data = df[df['Strike'].isin(top_volume_strikes['Strike'].values)]
        print(f"Options for top strikes: {len(top_strike_data)}")

        # Get the latest trade for each of these strikes
        latest_trades = top_strike_data.groupby('Strike')['Last Trade Date'].max()
        print("Latest trade times for each top strike:")
        for strike, trade_time in latest_trades.items():
            print(f"  Strike: {strike}, Latest trade: {trade_time}")

        # If we have valid timestamps, calculate average
        if (not latest_trades.empty
            and all(isinstance(t, pd.Timestamp) for t in latest_trades)):
            # Convert all timestamps to UNIX timestamp (seconds since epoch)
            timestamps = [t.timestamp() for t in latest_trades]

            # Calculate metrics
            max_time = pd.Timestamp.fromtimestamp(
                max(timestamps), tz='UTC').tz_localize(None)
            min_time = pd.Timestamp.fromtimestamp(
                min(timestamps), tz='UTC').tz_localize(None)
            avg_time = pd.Timestamp.fromtimestamp(
                (sum(timestamps) / len(timestamps)), tz='UTC').tz_localize(None)

            print("Timestamp analysis:")
            print(f"  Earliest of latest trades: {min_time}")
            print(f"  Latest of latest trades: {max_time}")
            print(f"  Time range (max-min): {max_time - min_time}")
            print(f"  Average timestamp: {avg_time}")
            print(f"  Time delta from latest: {avg_time - max_time}")

            # Check if average is in the future
            latest_overall = pd.to_datetime(df['Last Trade Date']).max()
            print(f"  Latest trade in entire dataset: {latest_overall}")
            print(f"  Average timestamp vs. latest overall: {avg_time - latest_overall}")

        return avg_time


    @staticmethod
    def _filter_options(
        puts: pd.DataFrame,
        calls: pd.DataFrame,
        params: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Filter options based on volume and recency criteria.

        Returns:
        --------
        Tuple[DataFrame, DataFrame]: Filtered puts and calls
        """
        if params['filter_low_liquidity_pairs'] and params['has_volume'] and params['volume_stats']:
            min_abs_volume = params['min_absolute_volume']
            min_rel_factor = params['min_relative_volume_factor']
            median_volume = params['volume_stats']['median']

            rel_threshold = median_volume * min_rel_factor if median_volume > 0 else 0
            volume_threshold = max(min_abs_volume, rel_threshold)

            print(f"  Volume filtering (threshold: {volume_threshold:.2f}):")
            puts_before = len(puts)
            calls_before = len(calls)

            if volume_threshold > 0:
                puts = puts[puts['Volume'] >= volume_threshold]
                calls = calls[calls['Volume'] >= volume_threshold]

                puts_removed = puts_before - len(puts)
                calls_removed = calls_before - len(calls)
                print(f"    Removed {puts_removed}/{puts_before} puts "
                        f"({puts_removed/puts_before:.1%})")
                print(f"    Removed {calls_removed}/{calls_before} calls "
                        f"({calls_removed/calls_before:.1%})")
                print(f"    Remaining: {len(puts)} puts, {len(calls)} calls")

        if (params['exclude_stale_options']
            and 'Last Trade Date' in puts.columns
            and 'Last Trade Date' in calls.columns):
            # Check if Last Trade Date has time information
            if (isinstance(puts['Last Trade Date'].iloc[0]
                           if not puts.empty else None, pd.Timestamp)):
                max_age = pd.Timedelta(minutes=params['max_trade_age_minutes'])

                print(f"  Recency filtering (max age: {max_age} minutes):")
                puts_before = len(puts)
                calls_before = len(calls)

                print(f"    Reference time: {params['reference_time']}")
                print(f"    Earliest put trade time: {puts['Last Trade Date'].min()}")
                print(f"    Earliest call trade time: {calls['Last Trade Date'].min()}")
                print(f"    Last put trade time: {puts['Last Trade Date'].max()}")
                print(f"    Last call trade time: {calls['Last Trade Date'].max()}")
                print(f"    Time window: {params['reference_time'] - max_age} "
                        f"to {params['reference_time']}")

                # Print timezone info for reference time
                print(f"Reference time tzinfo: {params['reference_time'].tzinfo}")

                # Print timezone info for option timestamps
                print(f"Option timestamp tzinfo: "
                f"{puts['Last Trade Date'].iloc[0].tzinfo if not puts.empty else 'N/A'}")

                puts = puts[puts['Last Trade Date'] >= params['reference_time'] - max_age]
                calls = calls[calls['Last Trade Date'] >= params['reference_time'] - max_age]

                puts_removed = puts_before - len(puts)
                calls_removed = calls_before - len(calls)
                print(f"    Removed {puts_removed}/{puts_before} puts "
                        f"({puts_removed/puts_before:.1%})")

                print(f"    Removed {calls_removed}/{calls_before} calls "
                        f"({calls_removed/calls_before:.1%})")
                print(f"    Remaining: {len(puts)} puts, {len(calls)} calls")

        return puts, calls


    @classmethod
    def _find_original_pairs(
        cls,
        puts: pd.DataFrame,
        calls: pd.DataFrame,
        expiry: object,
        params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Find option pairs using the original selection method.

        Returns:
        --------
        List[Dict[str, Any]]: Sorted list of option pairs
        """

        # First, collect exact strike matches
        exact_pairs = cls._exact_strike_matches_orig(
            puts=puts,
            calls=calls,
            expiry=expiry,
            params=params
            )

        # Look for close strikes if needed
        close_pairs = cls._close_strike_matches_orig(
            exact_pairs=exact_pairs,
            puts=puts,
            calls=calls,
            params=params
            )

        # Combine and sort pairs
        all_pairs = exact_pairs + close_pairs

        # Sort by: exact match first, then moneyness, then strike diff
        all_pairs.sort(key=lambda p: (
            -1 if p['is_exact_match'] else 0,
            p['moneyness'],
            p.get('strike_diff_pct', 0.0)
        ))

        return all_pairs


    @classmethod
    def _exact_strike_matches_orig(cls, puts, calls, expiry, params):
        exact_pairs = []

        common_strikes = set(puts['Strike']).intersection(set(calls['Strike']))

        print(f"Expiry {expiry}: {len(puts)} puts, {len(calls)} calls, "
                f"{len(common_strikes)} common strikes")

        for strike in common_strikes:
            try:
                put_row = puts[puts['Strike'] == strike].iloc[0]
                call_row = calls[calls['Strike'] == strike].iloc[0]

                strike_float = float(strike)
                put_price = float(put_row['Last Price'])
                call_price = float(call_row['Last Price'])
                moneyness = abs(strike_float / params['reference_price'] - 1.0)

                exact_pairs.append({
                    'put_strike': strike_float,
                    'call_strike': strike_float,
                    'put_price': put_price,
                    'call_price': call_price,
                    'strike_diff': 0.0,
                    'strike_diff_pct': 0.0,
                    'moneyness': moneyness,
                    'is_exact_match': True,
                    'years': float(put_row['Years To Expiry']),
                    'days': int(put_row['Days To Expiry'])
                })
            except (ValueError, ZeroDivisionError, OverflowError,
                    RuntimeWarning) as e:
                print(f"  Error processing strike {strike}: {e}")

        return exact_pairs


    @classmethod
    def _close_strike_matches_orig(cls, exact_pairs, puts, calls, params):
        close_pairs = []

        max_diff = params['reference_price'] * params['max_strike_diff_pct']


        if len(exact_pairs) < params['close_strike_min_pairs']:
            for _, put_row in puts.iterrows():
                put_strike = float(put_row['Strike'])
                put_moneyness = abs(put_strike / params['reference_price'] - 1.0)

                for _, call_row in calls.iterrows():
                    call_strike = float(call_row['Strike'])
                    call_moneyness = abs(call_strike / params['reference_price'] - 1.0)

                    # Check if strikes are close enough
                    strike_diff = abs(put_strike - call_strike)
                    if strike_diff > max_diff:
                        continue

                    # Skip if we already have an exact match for this strike
                    if any(p['put_strike'] == put_strike
                           and p['call_strike'] == put_strike for p in exact_pairs):
                        continue

                    # Calculate average moneyness
                    avg_moneyness = (put_moneyness + call_moneyness) / 2

                    close_pairs.append({
                        'put_strike': put_strike,
                        'call_strike': call_strike,
                        'put_price': float(put_row['Last Price']),
                        'call_price': float(call_row['Last Price']),
                        'strike_diff': strike_diff,
                        'strike_diff_pct': strike_diff / params['reference_price'],
                        'moneyness': avg_moneyness,
                        'is_exact_match': False,
                        'years': float(put_row['Years To Expiry']),
                        'days': int(put_row['Days To Expiry'])
                    })

        return close_pairs


    @classmethod
    def _find_enhanced_pairs(
        cls,
        puts: pd.DataFrame,
        calls: pd.DataFrame,
        expiry: object,
        params: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Find option pairs using the enhanced selection method.

        Returns:
        --------
        List[Dict[str, Any]]: Sorted list of option pairs
        """

        # Determine if we have time information
        params['has_time'] = ('Last Trade Date' in puts.columns
                    and 'Last Trade Date' in calls.columns
                    and isinstance(puts['Last Trade Date'].iloc[0]
                                   if not puts.empty else None, pd.Timestamp))

        # Process exact strike matches
        pairs = cls._exact_strike_matches_enh(
            puts=puts,
            calls=calls,
            params=params,
            expiry=expiry
            )

        # Look for close strikes
        pairs = cls._close_strike_matches_enh(
            pairs=pairs,
            puts=puts,
            calls=calls,
            params=params
            )

        # Sort pairs based on the enhanced criteria
        pairs = cls._sort_enhanced_pairs(pairs, params)

        return pairs, params


    @classmethod
    def _exact_strike_matches_enh(cls, puts, calls, params, expiry):
        pairs = []

        # First, collect exact strike matches
        common_strikes = set(puts['Strike']).intersection(set(calls['Strike']))

        print(f"Expiry {expiry} (enhanced): {len(puts)} puts, {len(calls)} "
              f"calls, {len(common_strikes)} common strikes")

        for strike in common_strikes:
            try:
                put_row = puts[puts['Strike'] == strike].iloc[0]
                call_row = calls[calls['Strike'] == strike].iloc[0]

                put_price = float(put_row['Last Price'])
                call_price = float(call_row['Last Price'])
                moneyness = abs(float(strike) / params['reference_price'] - 1.0)

                # Create base pair info
                pair = {
                    'put_strike': float(strike),
                    'call_strike': float(strike),
                    'put_price': put_price,
                    'call_price': call_price,
                    'strike_diff': 0.0,
                    'strike_diff_pct': 0.0,
                    'moneyness': moneyness,
                    'is_exact_match': True,
                    'years': float(put_row['Years To Expiry']),
                    'days': int(put_row['Days To Expiry']),
                }

                # Add scoring metrics
                pair = cls._add_enhanced_metrics(
                    pair=pair,
                    put_row=put_row,
                    call_row=call_row,
                    params=params
                )

                pairs.append(pair)
            except (ValueError, ZeroDivisionError, OverflowError,
                    RuntimeWarning) as e:
                print(f"  Error processing strike {strike}: {e}")

        return pairs


    @classmethod
    def _close_strike_matches_enh(cls, pairs, puts, calls, params):
        max_diff = params['reference_price'] * params['max_strike_diff_pct']

        for _, put_row in puts.iterrows():
            put_strike = float(put_row['Strike'])
            put_moneyness = abs(put_strike / params['reference_price'] - 1.0)

            for _, call_row in calls.iterrows():
                call_strike = float(call_row['Strike'])
                call_moneyness = abs(call_strike / params['reference_price'] - 1.0)

                # Check if strikes are close enough
                strike_diff = abs(put_strike - call_strike)
                if strike_diff > max_diff:
                    continue

                # Skip if we already have an exact match for this put strike
                if any(p['put_strike'] == put_strike
                       and p['is_exact_match'] for p in pairs):
                    continue

                # Calculate average moneyness
                avg_moneyness = (put_moneyness + call_moneyness) / 2

                # Create base pair info
                pair = {
                    'put_strike': put_strike,
                    'call_strike': call_strike,
                    'put_price': float(put_row['Last Price']),
                    'call_price': float(call_row['Last Price']),
                    'strike_diff': strike_diff,
                    'strike_diff_pct': strike_diff / params['reference_price'],
                    'moneyness': avg_moneyness,
                    'is_exact_match': False,
                    'years': float(put_row['Years To Expiry']),
                    'days': int(put_row['Days To Expiry']),
                }

                # Add scoring metrics
                pair = cls._add_enhanced_metrics(
                    pair=pair,
                    put_row=put_row,
                    call_row=call_row,
                    params=params
                )

                pairs.append(pair)

        return pairs


    @classmethod
    def _add_enhanced_metrics(
        cls,
        pair: Dict[str, Any],
        put_row: pd.Series,
        call_row: pd.Series,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Add enhanced selection metrics to the pair dictionary.
        """
        # Add volume score if applicable
        if params['consider_volume'] and params['has_volume'] and params['volume_stats']:
            pair['volume_score'] = cls._calculate_volume_score(
                put_row=put_row,
                call_row=call_row,
                volume_stats=params['volume_stats']
            )

        # Add recency score if applicable
        if params['prioritize_recent_trades'] and params['has_time']:
            pair['recency_score'] = cls._calculate_recency_score(
                put_trade_time=put_row['Last Trade Date'],
                call_trade_time=call_row['Last Trade Date'],
                reference_time=params['reference_time'],
                max_age_minutes=params['max_trade_age_minutes']
            )

        # Add roundness score if applicable
        if params['prioritize_round_strikes']:
            if pair['is_exact_match']:
                pair['roundness_score'] = cls._calculate_roundness_score(
                    strike=pair['put_strike'],
                    underlying_price=params['underlying_price']
                )
            else:
                # For non-exact pairs, calculate scores for both strikes
                put_roundness = cls._calculate_roundness_score(
                    strike=pair['put_strike'],
                    underlying_price=params['underlying_price']
                )
                call_roundness = cls._calculate_roundness_score(
                    strike=pair['call_strike'],
                    underlying_price=params['underlying_price']
                )
                pair['roundness_score'] = (put_roundness + call_roundness) / 2

        # Calculate final score
        pair['final_score'] = cls._calculate_final_score(pair, params)

        return pair


    @staticmethod
    def _calculate_volume_score(
        put_row: pd.Series,
        call_row: pd.Series,
        volume_stats: Dict[str, float]
    ) -> float:
        """
        Calculate a relative volume score based on median volume.

        Returns:
        --------
        float: Volume score between 0 and 1
        """
        median_vol = volume_stats['median'] or 1.0  # Prevent division by zero

        # Get volumes with default 0
        put_vol = float(put_row['Volume']) if not pd.isna(put_row['Volume']) else 0.0
        call_vol = float(call_row['Volume']) if not pd.isna(call_row['Volume']) else 0.0

        # Relative volume scores
        put_vol_rel = put_vol / median_vol
        call_vol_rel = call_vol / median_vol

        # Use average of normalized volumes (caps at 1.0)
        return min(1.0, (put_vol_rel + call_vol_rel) / 2)


    @staticmethod
    def _calculate_recency_score(
        put_trade_time: pd.Timestamp,
        call_trade_time: pd.Timestamp,
        reference_time: pd.Timestamp,
        max_age_minutes: int
    ) -> float:
        """
        Calculate a recency score based on the most recent of the two options.

        Returns:
        --------
        float: Recency score between 0 and 1
        """
        # Calculate age of each trade in minutes
        put_age_minutes = (reference_time - put_trade_time).total_seconds() / 60.0
        call_age_minutes = (reference_time - call_trade_time).total_seconds() / 60.0

        # Use the most recent trade for the score
        age_minutes = min(put_age_minutes, call_age_minutes)

        if age_minutes < 0:
            # Future trades (should not happen) get a zero score
            return 0.0
        if age_minutes <= max_age_minutes:
            # Linear score from 1.0 (age 0) to 0.0 (age max_age_minutes)
            return 1.0 - (age_minutes / max_age_minutes)

        # Older trades get zero score
        return 0.0


    @staticmethod
    def _calculate_roundness_score(strike: float, underlying_price: float) -> float:
        """
        Calculate roundness score based on mathematical properties.

        Returns a score between 0 and 1 where higher values indicate rounder strikes.
        """
        # Handle zero or negative strikes
        if strike <= 0:
            return 0.0

        # Identify nearest power of 10 relative to underlying price
        magnitude = 10 ** math.floor(math.log10(underlying_price))

        # Normalize strike relative to price magnitude
        relative_value = strike / magnitude

        # Common fractions in financial markets (1, 1/2, 1/4, 1/5, 1/10)
        key_fractions = [1.0, 2.0, 5.0, 10.0, 0.5, 0.25, 0.2, 0.1]

        # Calculate distance to nearest "round" level
        min_distance = float('inf')
        for fraction in key_fractions:
            # Find nearest multiple of this fraction × magnitude
            multiple = round(relative_value / fraction) * fraction
            distance = abs(relative_value - multiple)
            min_distance = min(min_distance, distance)

        # Convert distance to roundness score (inversely proportional)
        # Normalize to ensure scores are between 0 and 1
        # A perfect round number will have distance=0 and score=1
        roundness = 1.0 / (1.0 + 10.0 * min_distance)

        # Cap at 1.0 for perfect matches
        return min(roundness, 1.0)


    @staticmethod
    def _calculate_final_score(pair: Dict[str, Any], params: Dict[str, Any]) -> float:
        """
        Calculate final score from individual metrics.

        Returns:
        --------
        float: Final score between 0 and 1
        """
        # ATM score (1.0 for perfect ATM, decreases as moves away from ATM)
        # Cap moneyness for scoring to avoid extreme penalties for far OTM/ITM
        atm_score = 1.0 - min(pair['moneyness'], 0.2) / 0.2

        # Start with weighted ATM score
        atm_weight = params['atm_priority_factor']
        final_score = atm_weight * atm_score

        # If we have secondary factors, calculate their weighted contribution
        if params['secondary_factor_weight'] > 0:
            secondary_score = 0.0
            secondary_factors = 0

            # Include volume if available
            if 'volume_score' in pair:
                secondary_score += pair['volume_score']
                secondary_factors += 1

            # Include recency if available
            if 'recency_score' in pair:
                secondary_score += pair['recency_score']
                secondary_factors += 1

            # Include roundness if available
            if 'roundness_score' in pair:
                secondary_score += pair['roundness_score']
                secondary_factors += 1

            # Add weighted average of secondary factors
            if secondary_factors > 0:
                secondary_avg = secondary_score / secondary_factors
                final_score += params['secondary_factor_weight'] * secondary_avg

        # Penalize non-exact matches based on strike difference
        if not pair['is_exact_match']:
            # Penalty increases with larger strike differences
            # A 5% difference will reduce score by about 0.5
            diff_penalty = pair['strike_diff_pct'] * 10.0
            final_score = max(0.0, final_score - diff_penalty)

        return final_score


    @staticmethod
    def _sort_enhanced_pairs(
            pairs: List[Dict[str, Any]],
            params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Sort pairs based on the enhanced selection criteria.

        Returns:
        --------
        List[Dict[str, Any]]: Sorted pairs
        """
        # # Sort by exact match first, then by final score
        # pairs.sort(key=lambda p: (
        #     -1 if p['is_exact_match'] else 0,  # Exact matches first
        #     -p.get('final_score', 0.0)         # Then by final score (higher is better)
        # ))

        # return pairs
        # First pass: separate exact matches from non-exact
        exact_matches = [p for p in pairs if p['is_exact_match']]
        non_exact = [p for p in pairs if not p['is_exact_match']]

        # Sort exact matches by moneyness first, then by enhanced score
        exact_matches.sort(key=lambda p: (p['moneyness'], -p.get('final_score', 0.0)))

        # For non-exact matches, enforce maximum strike difference threshold
        max_allowed_diff_pct = min(0.1, params.get('max_strike_diff_pct', 0.5))  # Hard cap at 10%
        filtered_non_exact = [p for p in non_exact if p['strike_diff_pct'] <= max_allowed_diff_pct]

        # Sort non-exact by strike difference first, then moneyness, then by enhanced score
        filtered_non_exact.sort(
            key=lambda p: (p['strike_diff_pct'],
                           p['moneyness'], -p.get('final_score', 0.0)))

        # Return exact matches first, then filtered non-exact
        return exact_matches + filtered_non_exact

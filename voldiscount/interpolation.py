"""
Interpolation and Extrapolation methods
"""
from typing import Dict, Tuple, Any

import pandas as pd


class Interpolation():
    """
    Interpolation and Extrapolation methods
    """
    @staticmethod
    def interpolate_rate(
        df: pd.DataFrame,
        expiry_date: pd.Timestamp,
        days: int,
        years: float,
        params: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
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
            return df, params

        before = before_df.iloc[0]
        after = after_df.iloc[0]

        # Linear interpolation
        days_frac = (days - before['Days']) / (after['Days'] - before['Days'])
        rate = before['Discount Rate'] + days_frac * (
            after['Discount Rate'] - before['Discount Rate'])

        # Also interpolate reference_price if available
        reference_price = None
        forward_ratio = None
        if 'Forward Price' in before and 'Forward Price' in after:
            reference_price = before['Forward Price'] + days_frac * (
                after['Forward Price'] - before['Forward Price'])

        if 'Forward Ratio' in before and 'Forward Ratio' in after:
            forward_ratio = before['Forward Ratio'] + days_frac * (
                after['Forward Ratio'] - before['Forward Ratio'])

        print(f"Interpolated rate for {expiry_date} ({days} days): {rate:.6f}")
        print(f"  Between: {before['Expiry']} ({before['Days']} days): "
              f"{before['Discount Rate']:.6f}")
        print(f"  And: {after['Expiry']} ({after['Days']} days): "
              f"{after['Discount Rate']:.6f}")

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
            return pd.concat([df, pd.DataFrame([filtered_row])], ignore_index=True), params
        # If all values were None, just return the original DataFrame unchanged
        return df, params


    @staticmethod
    def extrapolate_early(
        df: pd.DataFrame,
        expiry_date: pd.Timestamp,
        days: int,
        years: float,
        params: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
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

        if len(df) < params['min_options_per_expiry']:
            print(f"Cannot extrapolate for {expiry_date}: insufficient data points")
            return df, params

        vals = {}
        # Use the first two points for early extrapolation
        vals['first'] = df.sort_values('Days').iloc[0]
        vals['second'] = df.sort_values('Days').iloc[1]

        # Simple linear extrapolation
        vals['days_diff'] = vals['second']['Days'] - vals['first']['Days']
        vals['rate_diff'] = vals['second']['Discount Rate'] - vals['first']['Discount Rate']
        vals['daily_rate_change'] = vals['rate_diff'] / vals['days_diff']

        vals['extrapolated_rate'] = (
            vals['first']['Discount Rate'] - (vals['first']['Days'] - days)
            * vals['daily_rate_change'])
        vals['extrapolated_rate'] = max(0.0, vals['extrapolated_rate'])  # Ensure non-negative

        # Also extrapolate reference_price if available
        reference_price = None
        forward_ratio = None
        if 'Forward Price' in vals['first'] and 'Forward Price' in vals['second']:
            price_diff = vals['second']['Forward Price'] - vals['first']['Forward Price']
            daily_price_change = price_diff / vals['days_diff']
            reference_price = (vals['first']['Forward Price']
                               - (vals['first']['Days'] - days) * daily_price_change)
            reference_price = max(0.0, reference_price)  # Ensure non-negative

        if 'Forward Ratio' in vals['first'] and 'Forward Ratio' in vals['second']:
            ratio_diff = vals['second']['Forward Ratio'] - vals['first']['Forward Ratio']
            daily_ratio_change = ratio_diff / vals['days_diff']
            forward_ratio = (vals['first']['Forward Ratio']
                             - (vals['first']['Days'] - days) * daily_ratio_change)
            forward_ratio = max(0.0, forward_ratio)  # Ensure non-negative

        print(f"Extrapolated early rate for {expiry_date} ({days} days): "
              f"{vals['extrapolated_rate']:.6f}")
        print(f"  Using: {vals['first']['Expiry']} ({vals['first']['Days']} days): "
              f"{vals['first']['Discount Rate']:.6f}")
        print(f"  And: {vals['second']['Expiry']} ({vals['second']['Days']} days): "
              f"{vals['second']['Discount Rate']:.6f}")

        # Create new row for the dataframe
        new_row = {
            'Expiry': expiry_date,
            'Days': days,
            'Years': years,
            'Discount Rate': vals['extrapolated_rate'],
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
            return pd.concat([df, pd.DataFrame([filtered_row])], ignore_index=True), params
        # If all values were None, just return the original DataFrame unchanged
        return df, params


    @staticmethod
    def extrapolate_late(
        df: pd.DataFrame,
        expiry_date: pd.Timestamp,
        days: int,
        years: float,
        params: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
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

        if len(df) < params['min_options_per_expiry']:
            print(f"Cannot extrapolate for {expiry_date}: insufficient data points")
            return df, params

        vals = {}
        # Use the last two points for late extrapolation
        vals['last'] = df.sort_values('Days', ascending=False).iloc[0]
        vals['second_last'] = df.sort_values('Days', ascending=False).iloc[1]

        # Simple linear extrapolation
        vals['days_diff'] = (vals['last']['Days']
                                      - vals['second_last']['Days'])
        vals['rate_diff'] = (
            vals['last']['Discount Rate']
            - vals['second_last']['Discount Rate'])
        vals['daily_rate_change'] = (
            vals['rate_diff'] / vals['days_diff'])

        vals['extrapolated_rate'] = (
            vals['last']['Discount Rate']
            + (days - vals['last']['Days'])
            * vals['daily_rate_change'])
        vals['extrapolated_rate'] = max(
            0.0, min(0.2, vals['extrapolated_rate']))  # Ensure reasonable bounds

        # Also extrapolate reference_price if available
        reference_price = None
        forward_ratio = None
        if ('Forward Price' in vals['last']
            and 'Forward Price' in vals['second_last']):
            price_diff = (vals['last']['Forward Price']
                          - vals['second_last']['Forward Price'])
            daily_price_change = price_diff / vals['days_diff']
            reference_price = (
                vals['last']['Forward Price']
                + (days - vals['last']['Days']) * daily_price_change)
            # Ensure reasonable bounds - allow significant growth for long-dated
            # forward prices
            reference_price = max(
                vals['last']['Forward Price'], reference_price)

        if ('Forward Ratio' in vals['last']
            and 'Forward Ratio' in vals['second_last']):
            ratio_diff = (vals['last']['Forward Ratio']
                          - vals['second_last']['Forward Ratio'])
            daily_ratio_change = ratio_diff / vals['days_diff']
            forward_ratio = (
                vals['last']['Forward Ratio']
                + (days - vals['last']['Days']) * daily_ratio_change)
            forward_ratio = max(vals['last']['Forward Ratio'], forward_ratio)

        print(f"Extrapolated late rate for {expiry_date} ({days} days): "
              f"{vals['extrapolated_rate']:.6f}")
        print(f"  Using: {vals['second_last']['Expiry']} "
              f"({vals['second_last']['Days']} days): "
              f"{vals['second_last']['Discount Rate']:.6f}")
        print(f"  And: {vals['last']['Expiry']} "
              f"({vals['last']['Days']} days): "
              f"{vals['last']['Discount Rate']:.6f}")

        # Create new row for the dataframe
        new_row = {
            'Expiry': expiry_date,
            'Days': days,
            'Years': years,
            'Discount Rate': vals['extrapolated_rate'],
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
            return pd.concat([df, pd.DataFrame([filtered_row])], ignore_index=True), params
        # If all values were None, just return the original DataFrame unchanged
        return df, params

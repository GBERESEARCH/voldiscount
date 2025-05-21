"""
Data functions for option processing
"""
import calendar
import io
import time
from datetime import datetime, date
from typing import Dict, Any, Tuple, Optional, Union, Set

import pandas as pd
import numpy as np
import yfinance as yf

from voldiscount.models import Models

implied_volatility = Models.implied_volatility
standardize_datetime = Models.standardize_datetime

# pylint: disable=invalid-name

class Data():
    """
    Data functions for option processing
    """
    @classmethod
    def load_data(
        cls,
        tables: Dict[str, Any],
        params: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Load data from specified source.

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
        start_time = time.time()
        params['timings'] = {'start': start_time}

        if params['filename'] is not None:
            # Load from file
            df, reference_date = cls._load_from_file(params)
            tables['source_data'] = df
            tables['raw_data'] = df
        else:
            # Load from ticker
            raw_df, df, fetched_price, reference_date = cls._load_from_ticker(params)
            tables['raw_data'] = raw_df
            tables['source_data'] = df

            # Set underlying price if not provided but fetched
            if params['underlying_price'] is None and fetched_price is not None:
                params['underlying_price'] = fetched_price
                print(f"Using fetched underlying price: {fetched_price}")

        # Store reference date in params
        params['reference_date'] = reference_date

        # Set underlying price if not set
        if params['underlying_price'] is None and tables['source_data'] is not None:
            near_term = tables['source_data'].sort_values('Days To Expiry').iloc[0]['Expiry']
            near_term_options = tables['source_data'][tables['source_data']['Expiry'] == near_term]
            params['underlying_price'] = near_term_options['Strike'].median()
            print(f"Using estimated underlying price: {params['underlying_price']}")

        params['timings']['data_loading'] = time.time() - start_time

        # Print data summary if data is available
        if tables['source_data'] is not None:
            cls._print_data_summary(tables['source_data'])

        return tables, params


    @classmethod
    def _load_from_file(
        cls,
        params: Dict[str, Any]
    ) -> Tuple[Optional[pd.DataFrame], Optional[date]]:
        """
        Load data from CSV file.

        Parameters:
        -----------
        params : dict
            Configuration parameters

        Returns:
        --------
        tuple : (DataFrame, date)
            Source data and reference date
        """
        filename = params['filename']
        reference_date = params.get('reference_date')

        try:
            df = pd.read_csv(filename)

            # Standardize datetime columns
            datetime_columns = ['Expiry', 'Last Trade Date']
            df = standardize_datetime(df, columns=datetime_columns)

            # Calculate reference date if not provided
            if reference_date is None:
                reference_date = df['Last Trade Date'].max().date()

            print(f"Loaded data from file: {filename}")
            print(f"Reference date: {reference_date}")

            # Add expiry metrics
            df['Days To Expiry'] = (df['Expiry'] - pd.Timestamp(reference_date)).dt.days
            df['Years To Expiry'] = df['Days To Expiry'] / 365.0

            return df, reference_date
        except (ValueError, ZeroDivisionError, OverflowError, RuntimeWarning) as e:
            print(f"Error loading data from file: {e}")
            return None, None


    @classmethod
    def _load_from_ticker(
        cls,
        params: Dict[str, Any]
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[float], Optional[date]]:
        """
        Load data from Yahoo Finance using the ticker.

        Parameters:
        -----------
        params : dict
            Configuration parameters:
                filename : str or None
                    Path to the CSV file containing options data
                ticker : str or None
                    Stock ticker to fetch option data for
                underlying_price : float or None
                    Underlying price, if None will be estimated

        Returns:
        --------
        tuple : (DataFrame, DataFrame, float, date)
            Raw data, source data, spot price, reference date
        """
        ticker = params['ticker']
        reference_date = params['reference_date']

        try:
            raw_df, df, fetched_price = cls._extract_option_data(
                ticker=ticker,
                params=params
            )

            if df is None or df.empty:
                print(f"ERROR: Failed to fetch data for ticker {ticker}")
                return None, None, None, None

            # Standardize datetime columns
            datetime_columns = ['Expiry', 'Last Trade Date']
            df = standardize_datetime(df, columns=datetime_columns)

            # Calculate reference date if not provided
            if reference_date is None:
                reference_date = df['Last Trade Date'].max().date()

            print(f"Reference date: {reference_date}")

            # Add expiry metrics
            df['Days To Expiry'] = (df['Expiry'] - pd.Timestamp(reference_date)).dt.days
            df['Years To Expiry'] = df['Days To Expiry'] / 365.0

            return raw_df, df, fetched_price, reference_date
        except (ValueError, ZeroDivisionError, OverflowError, RuntimeWarning) as e:
            print(f"Error fetching data for ticker {ticker}: {e}")
            return None, None, None, None


    @classmethod
    def _extract_option_data(
        cls,
        ticker: str,
        params: Dict[str, Any]
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[float]]:
        """
        Extract option chain data for a specified ticker

        Parameters:
        -----------
        ticker : str
            The ticker symbol to extract data for
        min_days : int, default=7
            Minimum days to expiry for included options
        min_volume : int, optional
            Minimum trading volume (if None, includes all)
        wait_time : float, default=0.5
            Wait time between API calls to avoid rate limiting

        Returns:
        --------
        tuple
            (option_data, spot_price) - Formatted option data and current spot price
        """

        try:
            # Get data from Yahoo Finance
            asset = yf.Ticker(ticker)

            # Extract spot price
            try:
                spot = asset.info['currentPrice']
            except KeyError:
                try:
                    spot = (asset.info['bid'] + asset.info['ask'])/2
                    if (abs(spot - asset.info['previousClose'])
                        / asset.info['previousClose']) > 0.2:
                        spot = asset.info['previousClose']
                except KeyError:
                    try:
                        spot = asset.info['navPrice']
                    except (ValueError, ZeroDivisionError, OverflowError, RuntimeWarning):
                        spot = asset.info['previousClose']

            # Get option expiry dates
            option_dates = asset.options

            # Initialize empty DataFrame
            all_options = pd.DataFrame()

            # Process each expiry date
            for expiry in option_dates:
                try:
                    # Get option chain for this expiry
                    chain = asset.option_chain(expiry)

                    # Process calls
                    calls = chain.calls
                    calls['Option Type'] = 'call'

                    # Process puts
                    puts = chain.puts
                    puts['Option Type'] = 'put'

                    # Combine and add expiry date
                    options = pd.concat([calls, puts])
                    options['Expiry'] = pd.to_datetime(expiry).date()

                    # Add to full data
                    all_options = pd.concat([all_options, options])

                except (ValueError, ZeroDivisionError, OverflowError, RuntimeWarning) as e:
                    print(f"Error processing {expiry}: {e}")

                # Wait to avoid rate limiting
                time.sleep(params['wait_time'])

            # If no data found, return early
            if all_options.empty:
                return None, None, spot

            # Rename columns to more readable format
            all_options = all_options.rename(columns={
                'lastPrice': 'Last Price',
                'bid': 'Bid',
                'ask': 'Ask',
                'lastTradeDate': 'Last Trade Date',
                'strike': 'Strike',
                'openInterest': 'Open Interest',
                'volume': 'Volume',
                'impliedVolatility': 'Implied Volatility'
            })

            # Clean and transform the data
            processed_data = cls._process_option_data(
                data=all_options,
                params=params
                )

            # Format for output
            # formatted_data = _format_output(data=processed_data)

            return all_options, processed_data, spot

        except (ValueError, ZeroDivisionError, OverflowError, RuntimeWarning) as e:
            print(f"Error extracting option data for {ticker}: {e}")
            return None, None, None


    @staticmethod
    def _process_option_data(
        data: pd.DataFrame,
        params: Dict[str, Any]) -> pd.DataFrame:
        """
        Clean and process option data

        Parameters:
        -----------
        data : pandas.DataFrame
            Raw option data
        min_days : int, default=7
            Minimum days to expiry
        min_volume : int, optional
            Minimum trading volume

        Returns:
        --------
        pandas.DataFrame
            Processed option data
        """

        # Convert dates to datetime
        data['Last Trade Date'] = pd.to_datetime(data['Last Trade Date'])
        data['Expiry_datetime'] = pd.to_datetime(data['Expiry'])

        # Calculate days to expiry
        today = pd.to_datetime('today').date()
        data['TTM'] = (data['Expiry_datetime']
                       - pd.to_datetime(today)) / pd.Timedelta(days=365)
        data['Days'] = np.round(data['TTM'] * 365, 0)

        # Clean numeric columns
        for col in ['Volume', 'Open Interest']:
            data[col] = data[col].fillna(0)
            data[col] = data[col].replace('-', 0).astype(int)

        for col in ['Bid', 'Ask']:
            data[col] = data[col].fillna(0)
            data[col] = data[col].replace('-', 0).astype(float)

        # Create Mid column
        data['Mid'] = (data['Ask'] + data['Bid']) / 2

        # Apply filters
        # Remove options already expired
        data = data[data['Days'] > 0]

        # Filter by minimum days to expiry
        if params['min_days'] is not None:
            data = data[data['Days'] >= params['min_days']]

        # Filter by minimum volume
        if params['min_volume'] is not None:
            data = data[data['Volume'] >= params['min_volume']]

        return data


    @staticmethod
    def _print_data_summary(
        df: pd.DataFrame
    ) -> None:
        """
        Print data summary.

        Parameters:
        -----------
        df : DataFrame
            Source data
        """
        unique_expiries = sorted(df['Expiry'].unique())
        print(f"\nFound {len(unique_expiries)} expiry dates in dataset:")
        for i, expiry in enumerate(unique_expiries):
            expiry_df = df[df['Expiry'] == expiry]
            puts = expiry_df[expiry_df['Option Type'].str.lower() == 'put'].shape[0]
            calls = expiry_df[expiry_df['Option Type'].str.lower() == 'call'].shape[0]
            print(f"{i+1}. {expiry.strftime('%Y-%m-%d')}: {puts} puts, {calls} calls")


    @classmethod
    def filter_monthlies(
        cls,
        tables: Dict[str, Any],
        params: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Filter DataFrame to just standard monthly expiry dates.

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
        df = tables['source_data']

        monthly_df = df[df['Expiry'].apply(cls._is_standard_monthly_expiry)].copy()

        tables['source_data'] = monthly_df

        return tables, params


    @staticmethod
    def _is_standard_monthly_expiry(dt):
        """
        Check if a date is a standard monthly option expiration (normally 3rd Friday,
        but Thursday when the Friday is a holiday).

        Parameters:
        -----------
        dt : datetime or pandas.Timestamp
            The date to check

        Returns:
        --------
        bool : True if the date is a standard monthly expiration
        """
        dt = pd.to_datetime(dt)

        # Calculate the date of the 3rd Friday
        c = calendar.monthcalendar(dt.year, dt.month)
        # Find all Fridays (weekday 4 = Friday)
        fridays = [week[4] for week in c if week[4] != 0]
        third_friday = pd.Timestamp(dt.year, dt.month, fridays[2])

        # Check if the date is the 3rd Friday
        if dt.date() == third_friday.date():
            return True

        # Check if the date is the Thursday before the 3rd Friday (potential holiday adjustment)
        if dt.weekday() == 3:  # Thursday
            next_day = dt + pd.Timedelta(days=1)
            if next_day.date() == third_friday.date():
                # This is the Thursday before the 3rd Friday
                # For longer-dated options, this is likely a holiday-adjusted expiry
                return True

        return False


    @classmethod
    def load_options_data(
        cls,
        filename: str,
        reference_date: Optional[Union[str, datetime, date]] = None
    ) -> Tuple[pd.DataFrame, date]:
        """
        Load and preprocess options data.

        Parameters:
        -----------
        filename : str
            Path to the CSV file containing options data

        Returns:
        --------
        pandas.DataFrame
            DataFrame with processed options data
        """
        # Read the data
        df = pd.read_csv(filename)

        # Standardize datetime columns
        datetime_columns = ['Expiry', 'Last Trade Date']
        df = standardize_datetime(df, columns=datetime_columns)

        # Calculate days to expiry based on the last trade date
        last_trade_dates = df['Last Trade Date']
        if reference_date is None:
            reference_date = last_trade_dates.max().date()

        print(f"Reference date: {reference_date}")

        # Add expiry metrics
        df['Days_To_Expiry'] = (df['Expiry'] - pd.Timestamp(reference_date)).dt.days #type: ignore
        df['Years_To_Expiry'] = df['Days_To_Expiry'] / 365.0

        return df, reference_date #type: ignore


    @staticmethod
    def _format_output(
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Format data for output with selected columns

        Parameters:
        -----------
        data : pandas.DataFrame
            Processed option data

        Returns:
        --------
        pandas.DataFrame
            Formatted output data
        """
        # Select relevant columns
        columns = ['Expiry', 'Strike', 'Last Trade Date',
                   'Last Price', 'Bid', 'Ask', 'Option Type']

        # Only include columns that exist
        valid_columns = [col for col in columns if col in data.columns]

        return data[valid_columns]


    @staticmethod
    def from_paste_data(
        text_data: str
    ) -> pd.DataFrame:
        """
        Parse option data from pasted text

        Parameters:
        -----------
        text_data : str
            Tab-separated option data text

        Returns:
        --------
        pandas.DataFrame
            Option data in DataFrame format
        """

        # Read the tab-separated data into a DataFrame
        df = pd.read_csv(io.StringIO(text_data), sep='\t')

        # Convert date columns if needed
        if 'Last Trade Date' in df.columns:
            df['Last Trade Date'] = pd.to_datetime(df['Last Trade Date'])

        if 'Expiry' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['Expiry']):
            df['Expiry'] = pd.to_datetime(df['Expiry'])

        return df


    @classmethod
    def calculate_implied_vols(
        cls,
        tables: Dict[str, Any],
        params: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Calculate implied volatilities using both discount rates.

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
        iv_start = time.time()

        # Create combined term structure
        combined_term_structure = cls._create_combined_term_structure(tables)

        # Generate set of expiries to exclude
        df = tables['source_data']
        expiries_to_exclude = set()

        for expiry in df['Expiry'].unique():
            puts = df[(df['Expiry'] == expiry) &
                      (df['Option Type'].str.lower() == 'put') &
                      (df['Last Price'] > params['min_option_price'])]
            calls = df[(df['Expiry'] == expiry) &
                       (df['Option Type'].str.lower() == 'call') &
                       (df['Last Price'] > params['min_option_price'])]

            if (len(puts) < params['min_options_per_type']
                or len(calls) < params['min_options_per_type']):
                expiries_to_exclude.add(expiry)

        # Create option data with both discount rates
        discount_df = cls._create_option_data_with_rates(
            df=df,
            params=params,
            term_structure=combined_term_structure,
            expiries_to_exclude=expiries_to_exclude,
            include_both_rates=True
        )

        # Add forward prices and moneyness calculations
        tables, params = cls._calc_forward_moneyness(
            discount_df=discount_df, tables=tables, params=params)

        params['timings']['implied_vols'] = time.time() - iv_start
        params['timings']['total'] = time.time() - params['timings']['start']

        # Print timing summary
        print(f"\nAnalysis completed in {params['timings']['total']:.2f} seconds.")
        print(f"- Data preparation: {params['timings']['data_loading']:.2f} seconds")
        print(f"- Calibration: {params['timings']['calibration']:.2f} seconds")
        print(f"- IV calculation: {params['timings']['implied_vols']:.2f} seconds")

        # Save outputs if requested
        if params['save_output']:
            cls._save_outputs(tables, params)

        return tables, params


    @staticmethod
    def _create_combined_term_structure(
        tables: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Create combined term structure with both discount rates.

        Parameters:
        -----------
        tables : dict
            Tables dictionary

        Returns:
        --------
        DataFrame : Combined term structure
        """
        direct_ts = tables.get('direct_term_structure')
        smooth_ts = tables.get('smooth_term_structure')

        # If either term structure is empty, return the non-empty one
        if ((direct_ts is None or direct_ts.empty)
            and smooth_ts is not None and not smooth_ts.empty):
            # Add placeholder Direct Discount Rate column
            smooth_ts['Direct Discount Rate'] = None
            # Rename the existing Discount Rate column to Smooth Discount Rate
            result = smooth_ts.rename(columns={'Discount Rate': 'Smooth Discount Rate'})
            return result

        if ((smooth_ts is None or smooth_ts.empty)
            and direct_ts is not None and not direct_ts.empty):
            # Add placeholder Smooth Discount Rate column
            direct_ts['Smooth Discount Rate'] = None
            # Rename the existing Discount Rate column to Direct Discount Rate
            result = direct_ts.rename(columns={'Discount Rate': 'Direct Discount Rate'})
            return result

        if (direct_ts is None or direct_ts.empty) and (smooth_ts is None or smooth_ts.empty):
            # Both are empty, return empty DataFrame with required columns
            return pd.DataFrame(columns=[
                'Expiry', 'Days', 'Years', 'Direct Discount Rate', 'Smooth Discount Rate'])

        # Create a combined DataFrame by merging on Expiry
        # Start with direct term structure and rename Discount Rate column
        direct_ts = direct_ts.rename( #type: ignore
            columns={'Discount Rate': 'Direct Discount Rate'}) #type: ignore
        # Rename columns in smooth term structure to avoid conflicts
        smooth_ts = smooth_ts.rename( #type: ignore
            columns={'Discount Rate': 'Smooth Discount Rate'}) #type: ignore

        # Columns to use from each term structure for merging
        direct_cols = ['Expiry', 'Days', 'Years', 'Direct Discount Rate']
        smooth_cols = ['Expiry', 'Smooth Discount Rate']

        # Merge the term structures on Expiry
        merged = pd.merge(
            direct_ts[direct_cols],
            smooth_ts[smooth_cols],
            on='Expiry',
            how='outer'
        )

        # Add any additional columns from direct_ts that might be useful
        for col in ['Put Strike', 'Call Strike', 'Put Price', 'Call Price',
                    'Forward Price', 'Forward Ratio']:
            if col in direct_ts.columns:
                merged[col] = merged['Expiry'].map(
                    direct_ts.set_index('Expiry')[col].to_dict()
                )

        return merged


    @staticmethod
    def _create_option_data_with_rates(
        df: pd.DataFrame,
        params: Dict,
        term_structure: pd.DataFrame,
        expiries_to_exclude: Optional[Set[pd.Timestamp]] = None,
        include_both_rates: bool = False
    ) -> pd.DataFrame:
        """
        Create a dataframe where each row is an option with the appropriate
        discount rate(s).

        Parameters:
        -----------
        df : pandas.DataFrame
            Original options data
        S : float
            Underlying price
        term_structure : pandas.DataFrame
            Term structure with discount rates (may include both direct and
                                                smooth rates)
        reference_date : datetime.date
            Reference date for the analysis
        expiries_to_exclude : set, optional
            Set of expiry dates to exclude
        include_both_rates : bool, default=False
            Whether to include both direct and smooth discount rates in the output

        Returns:
        --------
        pandas.DataFrame
            Options data with calculated discount rate(s)
        """
        # Create lookup dictionaries for the discount rates
        direct_rate_lookup = {}
        smooth_rate_lookup = {}

        # Check which rate columns are present
        column_check = {}
        column_check['has_direct'] = 'Direct Discount Rate' in term_structure.columns
        column_check['has_smooth'] = 'Smooth Discount Rate' in term_structure.columns
        column_check['has_legacy'] = (
            'Discount Rate' in term_structure.columns
            and not column_check['has_direct']
            and not column_check['has_smooth'])

        # Create appropriate lookups based on available columns
        if column_check['has_direct']:
            direct_rate_lookup = {row['Expiry']: row['Direct Discount Rate']
                                for _, row in term_structure.iterrows()}
        elif column_check['has_legacy']:
            # If only the legacy 'Discount Rate' column exists, use it for direct rates
            direct_rate_lookup = {row['Expiry']: row['Discount Rate']
                                for _, row in term_structure.iterrows()}

        if column_check['has_smooth']:
            smooth_rate_lookup = {row['Expiry']: row['Smooth Discount Rate']
                                for _, row in term_structure.iterrows()}

        # Create a list to store option data
        option_data = []

        for _, row in df.iterrows():
            expiry = row['Expiry']

            # Skip if expiry is in the exclusion list
            if expiries_to_exclude is not None and expiry in expiries_to_exclude:
                continue

            # Skip if trade date is before the reference date
            if pd.to_datetime(row['Last Trade Date']) < pd.to_datetime(params['reference_date']):
                continue

            # Find matching discount rates
            direct_rate = direct_rate_lookup.get(expiry)
            smooth_rate = smooth_rate_lookup.get(expiry)

            # Skip if no rates are available for this expiry
            if direct_rate is None and smooth_rate is None and not include_both_rates:
                continue

            # Create the option data dictionary
            option_dict = {
                'Contract Symbol': row.get('contractSymbol', None),
                'Reference Date': params['reference_date'],
                'Last Trade Date': row['Last Trade Date'],
                'Spot Price': params['underlying_price'],
                'Expiry': expiry,
                'Days': row['Days To Expiry'],
                'Years': row['Years To Expiry'],
                'Strike': row['Strike'],
                'Option Type': row['Option Type'],
                'Last Price': row['Last Price'],
                'Bid': row.get('Bid', None),
                'Ask': row.get('Ask', None),
                'Open Interest': row.get('Open Interest', None),
                'Volume': row.get('Volume', None),
                'Implied Volatility': row.get('Implied Volatility', None)
            }

            # Add the appropriate discount rate(s)
            if include_both_rates:
                option_dict['Direct Discount Rate'] = direct_rate
                option_dict['Smooth Discount Rate'] = smooth_rate
            elif direct_rate is not None:
                option_dict['Discount Rate'] = direct_rate
            elif smooth_rate is not None:
                option_dict['Discount Rate'] = smooth_rate

            option_data.append(option_dict)

        # Create dataframe of option data with rates
        option_df = pd.DataFrame(option_data)

        return option_df


    @staticmethod
    def _calc_forward_moneyness(discount_df, tables, params):

        # Direct method fields
        discount_df['Direct Forward Price'] = discount_df['Expiry'].map(
            lambda x: tables['direct_forwards'].get(x, params['underlying_price'])
            if tables['direct_forwards'] else params['underlying_price']
        )
        discount_df['Direct Forward Ratio'] = (
            discount_df['Direct Forward Price'] / params['underlying_price'])
        discount_df['Direct Moneyness Forward'] = (
            discount_df['Strike'] / discount_df['Direct Forward Price'] - 1.0)

        # Smooth method fields
        discount_df['Smooth Forward Price'] = discount_df['Expiry'].map(
            lambda x: tables['smooth_forwards'].get(x, params['underlying_price'])
            if tables['smooth_forwards'] else params['underlying_price']
        )
        discount_df['Smooth Forward Ratio'] = (
            discount_df['Smooth Forward Price'] / params['underlying_price'])
        discount_df['Smooth Moneyness Forward'] = (
            discount_df['Strike'] / discount_df['Smooth Forward Price'] - 1.0)

        # Calculate implied volatilities if requested
        if params.get('calculate_ivs', False):
            print("Calculating implied volatilities...")

            # Initialize columns for implied volatilities
            discount_df['Direct IV'] = np.nan
            discount_df['Smooth IV'] = np.nan

            # Calculate IVs for each option row
            for idx, row in discount_df.iterrows():
                # Extract parameters
                price = row['Last Price']
                S = params['underlying_price']
                K = row['Strike']
                T = row['Years']
                option_type = row['Option Type'].lower()

                # Calculate implied volatility using Direct Discount Rate
                if not pd.isna(row.get('Direct Discount Rate')):
                    try:
                        direct_iv = implied_volatility(
                            price=price,
                            S=S,
                            K=K,
                            T=T,
                            r=row['Direct Discount Rate'],
                            option_type=option_type,
                            q=0
                        )

                        if (not np.isnan(direct_iv)
                            and params['volatility_lower_bound'] <=
                            direct_iv <= params['volatility_upper_bound']):
                            discount_df.at[idx, 'Direct IV'] = direct_iv
                    except (ValueError, ZeroDivisionError, OverflowError, RuntimeWarning):
                        pass

                # Calculate implied volatility using Smooth Discount Rate
                if not pd.isna(row.get('Smooth Discount Rate')):
                    try:
                        smooth_iv = implied_volatility(
                            price=price,
                            S=S,
                            K=K,
                            T=T,
                            r=row['Smooth Discount Rate'],
                            option_type=option_type,
                            q=0
                        )

                        if (not np.isnan(smooth_iv)
                            and params['volatility_lower_bound'] <=
                            smooth_iv <= params['volatility_upper_bound']):
                            discount_df.at[idx, 'Smooth IV'] = smooth_iv
                    except (ValueError, ZeroDivisionError, OverflowError, RuntimeWarning):
                        pass

            # Count the number of options with valid IVs
            valid_direct_ivs = discount_df['Direct IV'].notna().sum()
            valid_smooth_ivs = discount_df['Smooth IV'].notna().sum()

            print(f"Calculated {valid_direct_ivs} valid Direct IVs and "
                     f"{valid_smooth_ivs} valid Smooth IVs")

        tables['discount_data'] = discount_df

        return tables, params


    @staticmethod
    def _save_outputs(
        tables: Dict[str, Any],
        params: Dict[str, Any]
    ) -> None:
        """
        Static method to save results to CSV files.

        Parameters:
        -----------
        tables : dict
            Tables dictionary
        params : dict
            Configuration parameters
        """
        # Save direct term structure
        direct_ts = tables.get('direct_term_structure')
        if direct_ts is not None and not direct_ts.empty:
            direct_file = params['output_file'].replace('.csv', '_direct.csv')
            direct_ts.to_csv(direct_file, index=False)
            print(f"Direct term structure saved to {direct_file}")

        # Save smooth term structure
        smooth_ts = tables.get('smooth_term_structure')
        if smooth_ts is not None and not smooth_ts.empty:
            smooth_file = params['output_file'].replace('.csv', '_smooth.csv')
            smooth_ts.to_csv(smooth_file, index=False)
            print(f"Smooth term structure saved to {smooth_file}")

        # Save discount data
        discount_df = tables.get('discount_data')
        if discount_df is not None:
            discount_df.to_csv(params['iv_output_file'], index=False)
            print(f"Implied volatilities saved to {params['iv_output_file']}")

        # Save raw data
        raw_df = tables.get('raw_data')
        if raw_df is not None:
            raw_df.to_csv(params['raw_output_file'], index=False)
            print(f"Raw options data saved to {params['raw_output_file']}")

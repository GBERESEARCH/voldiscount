"""
Class-based interface for Put-Call Parity calibration tool
"""
import numpy as np
import pandas as pd
import time
import calendar
from voldiscount.calibration.direct import direct_discount_rate_calibration
from voldiscount.calibration.smooth import smooth_curve_calibration
from voldiscount.config.config import DEFAULT_PARAMS
from voldiscount.core.option_extractor import extract_option_data, create_option_data_with_rates
from voldiscount.core.utils import standardize_datetime
from datetime import date
from typing import Dict, Any, Optional, Tuple

class VolDiscount:
    """
    Class for calibrating discount rates from option prices using put-call parity.
    These discount rates can then be used in volatility surface calibration.
    
    Attributes:
    -----------
    term_structure : pd.DataFrame
        Calibrated term structure of discount rates
    discount_df : pd.DataFrame
        Option data with implied volatilities
    raw_df : pd.DataFrame
        Raw option data
    forward_prices : dict
        Dictionary of forward prices keyed by expiry date
    underlying_price : float
        Price of the underlying asset
    reference_date : datetime.date
        Reference date for the analysis
    params : dict
        Configuration parameters
    ticker : str
        Ticker symbol of the underlying asset (if provided)
    """

    def __init__(
        self, 
        filename: Optional[str] = None, 
        ticker: Optional[str] = None, 
        underlying_price: Optional[float] = None, 
        **kwargs
    ) -> None:
        """
        Initialize calibration with data source and parameters.

        Parameters:
        -----------
        filename : str or None
            Path to the CSV file containing options data. If None, ticker must be provided.
        ticker : str or None
            Stock ticker to fetch option data for. If None, filename must be provided.
        underlying_price : float or None
            Underlying price, if None will be estimated
        **kwargs : dict
            Additional parameters:
                initial_rate : float
                    Initial guess for discount rates (annualized)
                min_days : int
                    Minimum days to expiry for options when fetching from ticker
                min_volume : int
                    Minimum trading volume for options when fetching from ticker
                debug : bool
                    Whether to print debug information
                best_pair_only : bool
                    Whether to use only the most ATM pair for each expiry
                save_output : bool
                    Whether to save results to CSV files
                output_file : str
                    Filename for term structure output
                iv_output_file : str
                    Filename for implied volatilities output
                raw_output_file : str
                    Filename for raw options data output
                skip_iv_calculation : bool
                    Whether to skip the IV calculation and just return option data with rates
                calibration_method : str, 'joint' or 'direct'
                    Whether to use joint calibration for the smoothest curve or direct to minimize IV differences per tenor
                use_forwards : bool
                    Whether to use forward prices instead of spot for moneyness calculation
                consider_volume : bool
                    Whether to consider volume/open interest in pair selection
                min_pair_volume : int
                    Minimum combined volume for a pair to be considered
        """
        # Initialize tables dictionary with None values
        tables = {
            'raw_data': None,
            'source_data': None,
            'direct_term_structure': None,
            'smooth_term_structure': None,
            'discount_data': None,
            'direct_forwards': {},
            'smooth_forwards': {}
        }

        # Initialize params with defaults and user overrides
        params = DEFAULT_PARAMS.copy()
        params.update(kwargs)
        params.update({
            'ticker': ticker,
            'filename': filename,
            'underlying_price': underlying_price
        })

        # Validate input sources
        if filename is None and ticker is None:
            raise ValueError("Either filename or ticker must be provided")

        # Execute calibration flow with explicit parameter passing
        tables, params = self._load_data(tables=tables, params=params)
        tables, params = self._filter_monthlies(tables=tables, params=params)
        if tables['source_data'] is not None:
            tables, params = self._calibrate_rates(tables=tables, params=params)
            tables, params = self._calculate_implied_vols(tables=tables, params=params)

        # Only at the end, store as instance attributes
        self.tables = tables
        self.params = params


    @staticmethod
    def _load_from_file(
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
        except Exception as e:
            print(f"Error loading data from file: {e}")
            return None, None

    @staticmethod
    def _load_from_ticker(
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
        min_days = params['min_days']
        min_volume = params['min_volume']
        reference_date = params.get('reference_date')

        try:
            raw_df, df, fetched_price = extract_option_data(
                ticker=ticker,
                min_days=min_days,
                min_volume=min_volume
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
        except Exception as e:
            print(f"Error fetching data for ticker {ticker}: {e}")
            return None, None, None, None

    @classmethod
    def _load_data(
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


    @staticmethod
    def _filter_monthlies(
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

        tables['source_data'] = monthly_df

        return tables, params


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
    def _calibrate_rates(
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

        # Extract required parameters
        df = tables['source_data']
        S = params['underlying_price']

        # Prepare calibration arguments (explicit parameter passing)
        calibration_args = {
            'min_option_price': params['min_option_price'],
            'min_options_per_expiry': params['min_options_per_expiry'],
            'reference_date': params['reference_date'],
            'monthlies': params['monthlies'],
            'max_strike_diff_pct': params['max_strike_diff_pct'],
            'consider_volume': params['consider_volume'],
            'min_pair_volume': params['min_pair_volume'],
            'debug': params['debug']
        }

        # Run direct calibration
        print("Running direct discount rate calibration...")
        direct_term_structure = direct_discount_rate_calibration(df, S, **calibration_args)

        # Run smooth curve calibration
        print("\nRunning smooth curve calibration...")
        smooth_term_structure = smooth_curve_calibration(df, S, **calibration_args)

        # Standardize datetime in term structures
        if direct_term_structure is not None and not direct_term_structure.empty:
            direct_term_structure = standardize_datetime(direct_term_structure, columns=['Expiry'])
            direct_forwards = {
                row['Expiry']: row['Forward Price']
                for _, row in direct_term_structure.iterrows()
                if 'Forward Price' in direct_term_structure.columns
            }
            tables['direct_term_structure'] = direct_term_structure
            tables['direct_forwards'] = direct_forwards

        if smooth_term_structure is not None and not smooth_term_structure.empty:
            smooth_term_structure = standardize_datetime(smooth_term_structure, columns=['Expiry'])
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
            cols_to_print = ['Expiry', 'Days', 'Years', 'Discount Rate', 'Forward Price', 'Forward Ratio']
            cols_available = [col for col in cols_to_print if col in direct_ts.columns]
            print(direct_ts[cols_available])

            # Also print calibration details
            print("\nOptions Used for Direct Calibration:")
            detail_cols = ['Expiry', 'Put Strike', 'Call Strike', 'Put Price', 'Call Price',
                          'Put Implied Volatility', 'Call Implied Volatility', 'Implied Volatility Diff']
            valid_detail_cols = [col for col in detail_cols if col in direct_ts.columns]
            print(direct_ts[valid_detail_cols])
        else:
            print("No valid term structure from direct calibration.")

        # Print smooth term structure
        print("\nSmooth Curve Term Structure:")
        smooth_ts = tables.get('smooth_term_structure')
        if smooth_ts is not None and not smooth_ts.empty:
            cols_to_print = ['Expiry', 'Days', 'Years', 'Discount Rate', 'Forward Price', 'Forward Ratio']
            cols_available = [col for col in cols_to_print if col in smooth_ts.columns]
            print(smooth_ts[cols_available])
        else:
            print("No valid term structure from smooth curve calibration.")


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
        if (direct_ts is None or direct_ts.empty) and smooth_ts is not None and not smooth_ts.empty:
            # Add placeholder Direct Discount Rate column
            smooth_ts['Direct Discount Rate'] = None
            # Rename the existing Discount Rate column to Smooth Discount Rate
            result = smooth_ts.rename(columns={'Discount Rate': 'Smooth Discount Rate'})
            return result

        if (smooth_ts is None or smooth_ts.empty) and direct_ts is not None and not direct_ts.empty:
            # Add placeholder Smooth Discount Rate column
            direct_ts['Smooth Discount Rate'] = None
            # Rename the existing Discount Rate column to Direct Discount Rate
            result = direct_ts.rename(columns={'Discount Rate': 'Direct Discount Rate'})
            return result

        if (direct_ts is None or direct_ts.empty) and (smooth_ts is None or smooth_ts.empty):
            # Both are empty, return empty DataFrame with required columns
            return pd.DataFrame(columns=['Expiry', 'Days', 'Years', 'Direct Discount Rate', 'Smooth Discount Rate'])

        # Create a combined DataFrame by merging on Expiry
        # Start with direct term structure and rename Discount Rate column
        direct_ts = direct_ts.rename(columns={'Discount Rate': 'Direct Discount Rate'}) #type: ignore
        # Rename columns in smooth term structure to avoid conflicts
        smooth_ts = smooth_ts.rename(columns={'Discount Rate': 'Smooth Discount Rate'}) #type: ignore

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

    @classmethod
    def _calculate_implied_vols(
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

            if len(puts) < params['min_options_per_type'] or len(calls) < params['min_options_per_type']:
                expiries_to_exclude.add(expiry)

        # Create option data with both discount rates
        discount_df = create_option_data_with_rates(
            df=df,
            S=params['underlying_price'],
            term_structure=combined_term_structure,
            reference_date=params['reference_date'],
            expiries_to_exclude=expiries_to_exclude,
            include_both_rates=True
        )

        # Add forward prices and moneyness calculations
        if discount_df is not None and not discount_df.empty:
            # Direct method fields
            discount_df['Direct Forward Price'] = discount_df['Expiry'].map(
                lambda x: tables['direct_forwards'].get(x, params['underlying_price'])
                if tables['direct_forwards'] else params['underlying_price']
            )
            discount_df['Direct Forward Ratio'] = discount_df['Direct Forward Price'] / params['underlying_price']
            discount_df['Direct Moneyness Forward'] = discount_df['Strike'] / discount_df['Direct Forward Price'] - 1.0

            # Smooth method fields
            discount_df['Smooth Forward Price'] = discount_df['Expiry'].map(
                lambda x: tables['smooth_forwards'].get(x, params['underlying_price'])
                if tables['smooth_forwards'] else params['underlying_price']
            )
            discount_df['Smooth Forward Ratio'] = discount_df['Smooth Forward Price'] / params['underlying_price']
            discount_df['Smooth Moneyness Forward'] = discount_df['Strike'] / discount_df['Smooth Forward Price'] - 1.0

            # Calculate implied volatilities if requested
            if params.get('calculate_ivs', False):
                print("Calculating implied volatilities...")
                from voldiscount.core.black_scholes import implied_volatility
                
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
                            
                            if not np.isnan(direct_iv) and params['volatility_lower_bound'] <= direct_iv <= params['volatility_upper_bound']:
                                discount_df.at[idx, 'Direct IV'] = direct_iv
                        except:
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
                            
                            if not np.isnan(smooth_iv) and params['volatility_lower_bound'] <= smooth_iv <= params['volatility_upper_bound']:
                                discount_df.at[idx, 'Smooth IV'] = smooth_iv
                        except:
                            pass
                            
                # Count the number of options with valid IVs
                valid_direct_ivs = discount_df['Direct IV'].notna().sum()
                valid_smooth_ivs = discount_df['Smooth IV'].notna().sum()
                
                print(f"Calculated {valid_direct_ivs} valid Direct IVs and {valid_smooth_ivs} valid Smooth IVs")

            tables['discount_data'] = discount_df
        else:
            print("WARNING: No valid option data created.")

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

    # Public accessor methods
    def get_direct_term_structure(self) -> pd.DataFrame:
        """Get the direct calibration term structure."""
        return self.tables.get('direct_term_structure', pd.DataFrame())

    def get_smooth_term_structure(self) -> pd.DataFrame:
        """Get the smooth calibration term structure."""
        return self.tables.get('smooth_term_structure', pd.DataFrame())

    def get_data_with_rates(self) -> pd.DataFrame:
        """Get options data with both discount rates."""
        return self.tables.get('discount_data', pd.DataFrame())

    def get_raw_data(self) -> pd.DataFrame:
        """Get raw option data."""
        return self.tables.get('raw_data', pd.DataFrame())

    def get_direct_forwards(self) -> Dict[pd.Timestamp, float]:
        """Get forward prices from direct calibration."""
        return self.tables.get('direct_forwards', {})

    def get_smooth_forwards(self) -> Dict[pd.Timestamp, float]:
        """Get forward prices from smooth calibration."""
        return self.tables.get('smooth_forwards', {})

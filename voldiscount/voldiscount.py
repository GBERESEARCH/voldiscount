"""
Class-based interface for PCP calibration tool
"""
import pandas as pd
import time
from voldiscount.core.utils import load_options_data, standardize_datetime
from voldiscount.calibration.direct import direct_discount_rate_calibration
from voldiscount.core.option_extractor import extract_option_data, create_option_data_with_rates
from voldiscount.config.config import DEFAULT_PARAMS
from typing import Dict, Any


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

    def __init__(self, filename=None, ticker=None, underlying_price=None, **kwargs):
        """
        Initialize the Voldiscount object.

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
        # Initialize results attributes
        self.term_structure = None
        self.discount_df = None
        self.raw_df = None
        self.forward_prices = {}
        self.underlying_price = underlying_price
        self.reference_date = None
        self.ticker = ticker
        
        # Store configuration parameters
        self.params: Dict[str, Any] = DEFAULT_PARAMS.copy()
        self.params.update(kwargs)
        
        # Validate inputs
        if filename is None and ticker is None:
            raise ValueError("Either filename or ticker must be provided")
        
        # Run calibration
        self._run_calibration(filename, ticker, underlying_price)

    def _run_calibration(self, filename=None, ticker=None, underlying_price=None):
        """
        Run the calibration process.

        Parameters:
        -----------
        filename : str or None
            Path to the CSV file containing options data
        ticker : str or None
            Stock ticker to fetch option data for
        underlying_price : float or None
            Underlying price, if None will be estimated
        """
        start_time = time.time()
        timings = {}
        
        # Load data either from file or from ticker
        if filename is not None:
            df, self.reference_date = load_options_data(filename)
            print(f"Loaded options data from file: {filename}")
        else:
            print(f"Fetching options data for ticker: {ticker}")
            self.raw_df, df, fetched_price = extract_option_data(
                ticker, 
                min_days=self.params['min_days'], 
                min_volume=self.params['min_volume']
            )
            
            if df is None or df.empty:
                print(f"ERROR: Failed to fetch data for ticker {ticker}")
                return
            
            # If underlying price wasn't provided but we fetched it, use the fetched price
            if underlying_price is None and fetched_price is not None:
                self.underlying_price = fetched_price
                print(f"Using fetched underlying price: {self.underlying_price}")
            
            # Ensure we have the expected columns and formats
            datetime_columns = ['Expiry', 'Last Trade Date']
            df = standardize_datetime(df, columns=datetime_columns)
            
            # Apply the same processing as in load_options_data
            # Use the most recent last trade date as reference date
            last_trade_dates = df['Last Trade Date']
            self.reference_date = last_trade_dates.max().date()
            print(f"Reference date: {self.reference_date}")
            
            # Add expiry metrics
            df['Days To Expiry'] = (df['Expiry'] - pd.Timestamp(self.reference_date)).dt.days
            df['Years To Expiry'] = df['Days To Expiry'] / 365.0
        
        # Set underlying price
        if self.underlying_price is not None:
            S = self.underlying_price
            print(f"Using provided underlying price: {S}")
        else:
            # Estimate underlying price from ATM options
            near_term = df.sort_values('Days To Expiry').iloc[0]['Expiry']
            near_term_options = df[df['Expiry'] == near_term]
            S = near_term_options['Strike'].median()
            self.underlying_price = S
            print(f"Using estimated underlying price: {S}")

        # Print summary of data
        self._print_data_summary(df)

        # Run calibration
        timings['pre_calibration'] = time.time() - start_time
        
        calibration_start = time.time()

        # Direct "dirty fit" calibration for best IV matches
        calibration_args = {
            'min_option_price': self.params['min_option_price'],
            'min_options_per_expiry': self.params['min_options_per_expiry'],
            'reference_date': self.reference_date,
            'monthlies': self.params['monthlies']
        }
        self.term_structure = direct_discount_rate_calibration(df, S, **calibration_args)

        timings['calibration'] = time.time() - calibration_start

        # Standardize datetime in term structure
        if self.term_structure is not None:
            self.term_structure = standardize_datetime(self.term_structure, columns=['Expiry'])

            self.forward_prices = {row['Expiry']: row['Forward Price'] 
                           for _, row in self.term_structure.iterrows() 
                           if 'Forward Price' in self.term_structure.columns}

            if self.term_structure.empty:
                print("ERROR: Failed to build term structure. Exiting.")
                return

            # Print term structure
            self._print_term_structure()

            # Calculate implied volatilities using the calibrated term structure
            iv_start = time.time()

            # Generate set of expiries to exclude
            expiries_to_exclude = set()
            for expiry in df['Expiry'].unique():
                puts = df[(df['Expiry'] == expiry) & (df['Option Type'].str.lower() == 'put') & 
                        (df['Last Price'] > self.params['min_option_price'])]
                calls = df[(df['Expiry'] == expiry) & (df['Option Type'].str.lower() == 'call') & 
                        (df['Last Price'] > self.params['min_option_price'])]
                
                if len(puts) < self.params['min_options_per_type'] or len(calls) < self.params['min_options_per_type']:
                    expiries_to_exclude.add(expiry)    

            # Just create option data with discount rates (much faster)
            self.discount_df = create_option_data_with_rates(df, S, self.term_structure, self.reference_date, expiries_to_exclude)
            timings['data_preparation'] = time.time() - iv_start
            
            if self.discount_df is not None and not self.discount_df.empty:
                self.discount_df['Forward Price'] = self.discount_df['Expiry'].map(lambda x: self.forward_prices.get(x, S))
                self.discount_df['Forward Ratio'] = self.discount_df['Forward Price'] / S
                self.discount_df['Moneyness Forward'] = self.discount_df['Strike'] / self.discount_df['Forward Price'] - 1.0
            else:
                print("WARNING: No valid option data created.")

            total_time = time.time() - start_time
            print(f"\nAnalysis completed in {total_time:.2f} seconds.")
            print(f"- Data preparation: {timings['pre_calibration']:.2f} seconds")
            print(f"- Calibration: {timings['calibration']:.2f} seconds")

            # Save to CSV if requested
            self._save_outputs()

    def _print_data_summary(self, df):
        """
        Print a summary of the option data.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing option data
        """
        unique_expiries = sorted(df['Expiry'].unique())
        print(f"\nFound {len(unique_expiries)} expiry dates in dataset:")
        for i, expiry in enumerate(unique_expiries):
            expiry_df = df[df['Expiry'] == expiry]
            puts = expiry_df[expiry_df['Option Type'].str.lower() == 'put'].shape[0]
            calls = expiry_df[expiry_df['Option Type'].str.lower() == 'call'].shape[0]
            print(f"{i+1}. {expiry.strftime('%Y-%m-%d')}: {puts} puts, {calls} calls")

    def _print_term_structure(self):
        """
        Print the calibrated term structure.
        """
        if self.term_structure is None or self.term_structure.empty:
            return
            
        print("\nTerm Structure of Discount Rates:")
        cols_to_print = ['Expiry', 'Days', 'Years', 'Discount Rate', 'Forward Price', 'Forward Ratio']
        cols_available = [col for col in cols_to_print if col in self.term_structure.columns]
        print(self.term_structure[cols_available])

        print("\nOptions Used for Calibration:")
        detail_cols = ['Expiry', 'Put Strike', 'Call Strike', 'Put Price', 'Call Price', 
                        'Put Implied Volatility', 'Call Implied Volatility', 'Implied Volatility Diff']
        # Only include columns that exist to avoid KeyError
        valid_detail_cols = [col for col in detail_cols if col in self.term_structure.columns]
        print(self.term_structure[valid_detail_cols])

    def _save_outputs(self):
        """
        Save the results to CSV files if requested.
        """
        if not self.params['save_output']:
            return
            
        if self.term_structure is not None:
            self.term_structure.to_csv(self.params['output_file'], index=False)
            print(f"Term structure saved to {self.params['output_file']}")

        if self.discount_df is not None:
            self.discount_df.to_csv(self.params['iv_output_file'], index=False)
            print(f"Implied volatilities saved to {self.params['iv_output_file']}")
            
        if hasattr(self, 'raw_df') and self.raw_df is not None:
            self.raw_df.to_csv(self.params['raw_output_file'], index=False)
            print(f"Raw options data saved to {self.params['raw_output_file']}")
            
    def get_term_structure(self) -> pd.DataFrame:
        """
        Get the term structure of discount rates.
        
        Returns:
        --------
        pd.DataFrame : Term structure
        """
        return self.term_structure #type: ignore
        
    def get_data_with_rates(self) -> pd.DataFrame:
        """
        Get the options data with discount rates.
        
        Returns:
        --------
        pd.DataFrame : Options data with calculated discount rates
        """
        return self.discount_df #type: ignore
        
    def get_raw_data(self) -> pd.DataFrame:
        """
        Get the raw option data.
        
        Returns:
        --------
        pd.DataFrame : Raw option data
        """
        return self.raw_df #type: ignore
        
    def get_forward_prices(self) -> dict:
        """
        Get the forward prices dictionary.
        
        Returns:
        --------
        dict : Forward prices keyed by expiry
        """
        return self.forward_prices

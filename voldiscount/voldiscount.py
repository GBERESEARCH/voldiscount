"""
Class-based interface for Put-Call Parity calibration tool
"""
from typing import Dict, Optional

import pandas as pd

from voldiscount.data import Data
from voldiscount.calibration import Calibration
from voldiscount.vol_params import DEFAULT_PARAMS


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
            Path to the CSV file containing options data. If None, ticker must
            be provided.
        ticker : str or None
            Stock ticker to fetch option data for. If None, filename must be
            provided.
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
                calculate_ivs: bool
                    Whether to skip the IV calculation and just return option
                    data with rates
                calibration_method : str, 'joint' or 'direct'
                    Whether to use joint calibration for the smoothest curve or
                    direct to minimize IV differences per tenor
                use_forwards : bool
                    Whether to use forward prices instead of spot for moneyness
                    calculation
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

        # Execute calibration flow
        tables, params = Data.load_data(tables=tables, params=params)
        tables, params = Data.filter_monthlies(tables=tables, params=params)
        if tables['source_data'] is not None:
            tables, params = Calibration.calibrate_rates(
                tables=tables, params=params)
            tables, params = Data.calculate_implied_vols(
                tables=tables, params=params)

        self.tables = tables
        self.params = params


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

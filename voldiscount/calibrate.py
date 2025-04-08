"""
Main CLI interface for the PCP calibration tool
"""
import pandas as pd
import argparse
import time
from voldiscount.calibration.direct import direct_discount_rate_calibration
from voldiscount.calibration.smooth import smooth_curve_calibration
from voldiscount.config.config import DEFAULT_PARAMS
from voldiscount.core.option_extractor import extract_option_data, create_option_data_with_rates
from voldiscount.core.utils import load_options_data, standardize_datetime
from typing import Dict, Any, Tuple, Optional


def calibrate(
    filename: Optional[str] = None, 
    ticker: Optional[str] = None, 
    underlying_price: Optional[float] = None, 
    **kwargs
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[Dict[pd.Timestamp, float]], Optional[Dict[pd.Timestamp, float]]]:
    """
    Main function to calibrate options data using both direct and smooth methods.

    Parameters:
    -----------
    filename : str or None
        Path to the CSV file containing options data. If None, ticker must be provided.
    ticker : str or None
        Stock ticker to fetch option data for. If None, filename must be provided.
    underlying_price : float or None
        Underlying price, if None will be estimated
    **kwargs : dict
        Additional parameters (see DEFAULT_PARAMS for details)

    Returns:
    --------
    tuple : (direct_term_structure DataFrame, smooth_term_structure DataFrame, 
             discount_df DataFrame, raw_df DataFrame, 
             direct_forwards dict, smooth_forwards dict)
    """
    
    # Update with provided kwargs
    params: Dict[str, Any] = DEFAULT_PARAMS.copy()
    params.update(kwargs)

    if filename is None and ticker is None:
        raise ValueError("Either filename or ticker must be provided")
    
    start_time = time.time()
    
    # Load data either from file or from ticker
    if filename is not None:
        df, reference_date = load_options_data(filename)
        print(f"Loaded options data from file: {filename}")
        raw_df = df  # For consistent return structure
    else:
        print(f"Fetching options data for ticker: {ticker}")
        raw_df, df, fetched_price = extract_option_data(
            ticker,  #type: ignore
            min_days=params['min_days'], 
            min_volume=params['min_volume']
        )
        
        if df is None or df.empty:
            print(f"ERROR: Failed to fetch data for ticker {ticker}")
            return None, None, None, None, None, None
        
        # If underlying price wasn't provided but we fetched it, use the fetched price
        if underlying_price is None and fetched_price is not None:
            underlying_price = fetched_price
            print(f"Using fetched underlying price: {underlying_price}")
        
        # Ensure we have the expected columns and formats
        datetime_columns = ['Expiry', 'Last Trade Date']
        df = standardize_datetime(df, columns=datetime_columns)
        
        # Apply the same processing as in load_options_data
        # Use the most recent last trade date as reference date
        last_trade_dates = df['Last Trade Date']
        reference_date = last_trade_dates.max().date()
        print(f"Reference date: {reference_date}")
        
        # Add expiry metrics
        df['Days To Expiry'] = (df['Expiry'] - pd.Timestamp(reference_date)).dt.days
        df['Years To Expiry'] = df['Days To Expiry'] / 365.0
    
    # Set underlying price
    if underlying_price is not None:
        S = underlying_price
        print(f"Using provided underlying price: {S}")
    else:
        # Estimate underlying price from ATM options
        near_term = df.sort_values('Days To Expiry').iloc[0]['Expiry']
        near_term_options = df[df['Expiry'] == near_term]
        S = near_term_options['Strike'].median()
        print(f"Using estimated underlying price: {S}")

    # Print summary of data
    unique_expiries = sorted(df['Expiry'].unique())
    print(f"\nFound {len(unique_expiries)} expiry dates in dataset:")
    for i, expiry in enumerate(unique_expiries):
        expiry_df = df[df['Expiry'] == expiry]
        puts = expiry_df[expiry_df['Option Type'].str.lower() == 'put'].shape[0]
        calls = expiry_df[expiry_df['Option Type'].str.lower() == 'call'].shape[0]
        print(f"{i+1}. {expiry.strftime('%Y-%m-%d')}: {puts} puts, {calls} calls")

    # Run calibration
    timings = {}
    timings['pre_calibration'] = time.time() - start_time
    
    # Calibration arguments
    calibration_args = {
        'min_option_price': params['min_option_price'],
        'min_options_per_expiry': params['min_options_per_expiry'],
        'reference_date': params.get('reference_date', reference_date),
        'monthlies': params['monthlies'],
        'max_strike_diff_pct': params['max_strike_diff_pct'],
        'consider_volume': params['consider_volume'],
        'min_pair_volume': params['min_pair_volume'],
        'debug': params['debug']
    }
    
    # Start calibration - run both methods
    calibration_start = time.time()
    
    # Run direct calibration first
    print("Running direct discount rate calibration...")
    direct_term_structure = direct_discount_rate_calibration(df, S, **calibration_args)
    
    # Run smooth curve calibration
    print("\nRunning smooth curve calibration...")
    smooth_term_structure = smooth_curve_calibration(df, S, **calibration_args)
    
    timings['calibration'] = time.time() - calibration_start

    # Standardize datetime in term structures
    direct_term_structure = standardize_datetime(direct_term_structure, columns=['Expiry'])
    smooth_term_structure = standardize_datetime(smooth_term_structure, columns=['Expiry'])

    # Extract forward prices from both term structures
    direct_forwards = {row['Expiry']: row['Forward Price'] 
                    for _, row in direct_term_structure.iterrows() 
                    if 'Forward Price' in direct_term_structure.columns}
                    
    smooth_forwards = {row['Expiry']: row['Forward Price'] 
                    for _, row in smooth_term_structure.iterrows() 
                    if 'Forward Price' in smooth_term_structure.columns}

    if direct_term_structure.empty and smooth_term_structure.empty:
        print("ERROR: Failed to build term structure with either method. Exiting.")
        return None, None, None, raw_df, None, None

    # Print direct term structure
    print("\nDirect Calibration Term Structure:")
    cols_to_print = ['Expiry', 'Days', 'Years', 'Discount Rate', 'Forward Price', 'Forward Ratio']
    direct_cols = [col for col in cols_to_print if col in direct_term_structure.columns]
    if not direct_term_structure.empty:
        print(direct_term_structure[direct_cols])
    else:
        print("No valid term structure from direct calibration.")

    # Print smooth term structure
    print("\nSmooth Curve Term Structure:")
    smooth_cols = [col for col in cols_to_print if col in smooth_term_structure.columns]
    if not smooth_term_structure.empty:
        print(smooth_term_structure[smooth_cols])
    else:
        print("No valid term structure from smooth curve calibration.")

    # Calculate implied volatilities using both calibrated term structures
    iv_start = time.time()
    
    # Create combined term structure with both discount rates
    combined_term_structure = create_combined_term_structure(direct_term_structure, smooth_term_structure)
    
    # Create option data with both discount rates
    discount_df = create_option_data_with_rates(
        df, S, combined_term_structure, reference_date, 
        include_both_rates=True  # Flag to indicate we want both rates in output
    )
    
    timings['data_preparation'] = time.time() - iv_start
    
    if discount_df.empty:
        print("WARNING: No valid option data created.")
        return direct_term_structure, smooth_term_structure, None, raw_df, direct_forwards, smooth_forwards   

    # Add forward prices and moneyness calculations for both methods
    if not discount_df.empty:
        # Direct method fields
        discount_df['Direct Forward Price'] = discount_df['Expiry'].map(
            lambda x: direct_forwards.get(x, S) if direct_forwards else S
        )
        discount_df['Direct Forward Ratio'] = discount_df['Direct Forward Price'] / S
        discount_df['Direct Moneyness Forward'] = discount_df['Strike'] / discount_df['Direct Forward Price'] - 1.0
        
        # Smooth method fields
        discount_df['Smooth Forward Price'] = discount_df['Expiry'].map(
            lambda x: smooth_forwards.get(x, S) if smooth_forwards else S
        )
        discount_df['Smooth Forward Ratio'] = discount_df['Smooth Forward Price'] / S
        discount_df['Smooth Moneyness Forward'] = discount_df['Strike'] / discount_df['Smooth Forward Price'] - 1.0

    total_time = time.time() - start_time
    print(f"\nAnalysis completed in {total_time:.2f} seconds.")
    print(f"- Data preparation: {timings['pre_calibration']:.2f} seconds")
    print(f"- Calibration: {timings['calibration']:.2f} seconds")
    print(f"- IV calculation: {timings['data_preparation']:.2f} seconds")

    # Save to CSV if requested
    if params['save_output']:
        if direct_term_structure is not None:
            direct_file = params['output_file'].replace('.csv', '_direct.csv')
            direct_term_structure.to_csv(direct_file, index=False)
            print(f"Direct term structure saved to {direct_file}")
            
        if smooth_term_structure is not None:
            smooth_file = params['output_file'].replace('.csv', '_smooth.csv')
            smooth_term_structure.to_csv(smooth_file, index=False)
            print(f"Smooth term structure saved to {smooth_file}")

        if discount_df is not None:
            discount_df.to_csv(params['iv_output_file'], index=False)
            print(f"Implied volatilities with both discount rates saved to {params['iv_output_file']}")
            
        if raw_df is not None:
            raw_df.to_csv(params['raw_output_file'], index=False)
            print(f"Raw options data saved to {params['raw_output_file']}")

    return direct_term_structure, smooth_term_structure, discount_df, raw_df, direct_forwards, smooth_forwards


def create_combined_term_structure(
    direct_ts: pd.DataFrame, 
    smooth_ts: pd.DataFrame
) -> pd.DataFrame:
    """
    Create a combined term structure DataFrame with both direct and smooth discount rates.
    
    Parameters:
    -----------
    direct_ts : pd.DataFrame
        Term structure from direct calibration
    smooth_ts : pd.DataFrame
        Term structure from smooth curve calibration
        
    Returns:
    --------
    pd.DataFrame : Combined term structure with both discount rates
    """
    # If either term structure is empty, return the non-empty one
    if direct_ts.empty and not smooth_ts.empty:
        # Add placeholder Direct Discount Rate column
        smooth_ts['Direct Discount Rate'] = None
        # Rename the existing Discount Rate column to Smooth Discount Rate
        smooth_ts = smooth_ts.rename(columns={'Discount Rate': 'Smooth Discount Rate'})
        return smooth_ts
        
    if smooth_ts.empty and not direct_ts.empty:
        # Add placeholder Smooth Discount Rate column
        direct_ts['Smooth Discount Rate'] = None
        # Rename the existing Discount Rate column to Direct Discount Rate
        direct_ts = direct_ts.rename(columns={'Discount Rate': 'Direct Discount Rate'})
        return direct_ts
    
    if direct_ts.empty and smooth_ts.empty:
        # Both are empty, return empty DataFrame with required columns
        return pd.DataFrame(columns=['Expiry', 'Days', 'Years', 'Direct Discount Rate', 'Smooth Discount Rate'])
    
    # Create a combined DataFrame by merging on Expiry
    # Start with direct term structure and rename Discount Rate column
    direct_ts = direct_ts.rename(columns={'Discount Rate': 'Direct Discount Rate'})
    # Rename columns in smooth term structure to avoid conflicts
    smooth_ts = smooth_ts.rename(columns={'Discount Rate': 'Smooth Discount Rate'})
    
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


# Add command-line interface if run directly
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PCP Calibration Tool')
    parser.add_argument('--filename', help='Path to CSV file with options data')
    parser.add_argument('--ticker', help='Stock ticker to fetch option data for')
    parser.add_argument('--price', type=float, help='Underlying price', default=None)
    parser.add_argument('--rate', type=float, help='Initial discount rate guess', default=0.05)
    parser.add_argument('--min-days', type=int, help='Minimum days to expiry when fetching from ticker', default=7)
    parser.add_argument('--min-volume', type=int, help='Minimum volume when fetching from ticker', default=10)
    parser.add_argument('--output', help='Output CSV file for term structure', default='term_structure.csv')
    parser.add_argument('--iv-output', help='Output CSV file for IVs', default='implied_volatilities.csv')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--save', action='store_true', help='Save results to CSV files')
    parser.add_argument('--reference-date', help='Reference date for options (YYYY-MM-DD)', default=None)
    parser.add_argument('--monthlies', action='store_true', 
                   help='Use only standard monthly options (3rd Friday)', default=True)
    parser.add_argument('--all-expiries', dest='monthlies', action='store_false',
                   help='Use all available expiry dates')

    args = parser.parse_args()
    
    # Check that at least one data source is provided
    if args.filename is None and args.ticker is None:
        parser.error("Either --filename or --ticker must be provided")

    direct_ts, smooth_ts, discount_df, raw_df, direct_forwards, smooth_forwards = calibrate(
        filename=args.filename, 
        ticker=args.ticker,
        underlying_price=args.price, 
        initial_rate=args.rate,
        min_days=args.min_days,
        min_volume=args.min_volume,
        debug=args.debug,
        save_output=args.save,
        output_file=args.output,
        iv_output_file=args.iv_output
    )
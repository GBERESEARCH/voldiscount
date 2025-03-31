"""
Main CLI interface for the PCP calibration tool
"""
import pandas as pd
import argparse
import time
from core.utils import load_options_data, standardize_datetime
from calibration.direct import direct_discount_rate_calibration
from core.option_extractor import extract_option_data, create_option_data_with_rates
from config.config import DEFAULT_PARAMS
from typing import Dict, Any


def calibrate(filename=None, ticker=None, underlying_price=None, **kwargs):
    """
    Main function to calibrate options data.

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

    Returns:
    --------
    tuple : (term_structure DataFrame, discount_df DataFrame, raw_df DataFrame, forward_prices dict)
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
    else:
        print(f"Fetching options data for ticker: {ticker}")
        raw_df, df, fetched_price = extract_option_data(
            ticker, 
            min_days=params['min_days'], 
            min_volume=params['min_volume']
        )
        
        if df is None or df.empty:
            print(f"ERROR: Failed to fetch data for ticker {ticker}")
            return None, None, None, None
        
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
        
        # Filter out options with very low prices
        # df = df[df['Last Price'] > 0.05].copy()
    
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
    
    calibration_start = time.time()

    # Direct "dirty fit" calibration for best IV matches
    calibration_args = {
        'min_option_price': params['min_option_price'],
        'min_options_per_expiry': params['min_options_per_expiry'],
        'reference_date': params['reference_date'],
        'monthlies': params['monthlies']
    }
    term_structure = direct_discount_rate_calibration(df, S, **calibration_args)

    timings['calibration'] = time.time() - calibration_start

    # Standardize datetime in term structure
    term_structure = standardize_datetime(term_structure, columns=['Expiry'])

    calibrated_forwards = {row['Expiry']: row['Forward Price'] 
                       for _, row in term_structure.iterrows() 
                       if 'Forward Price' in term_structure.columns}

    if term_structure.empty:
        print("ERROR: Failed to build term structure. Exiting.")
        return None, None, raw_df, calibrated_forwards

    # Print term structure
    print("\nTerm Structure of Discount Rates:")
    cols_to_print = ['Expiry', 'Days', 'Years', 'Discount Rate', 'Forward Price', 'Forward Ratio']
    print(term_structure[cols_to_print])

    print("\nOptions Used for Calibration:")
    detail_cols = ['Expiry', 'Put Strike', 'Call Strike', 'Put Price', 'Call Price', 'Put Implied Volatility', 'Call Implied Volatility', 'Implied Volatility Diff']
    # Only include columns that exist to avoid KeyError
    valid_detail_cols = [col for col in detail_cols if col in term_structure.columns]
    print(term_structure[valid_detail_cols])

    # Calculate implied volatilities using the calibrated term structure
    iv_start = time.time()
    # Just create option data with discount rates (much faster)
    discount_df = create_option_data_with_rates(df, S, term_structure, reference_date)
    timings['data_preparation'] = time.time() - iv_start
    if discount_df.empty:
        print("WARNING: No valid option data created.")
        return term_structure, None, raw_df, calibrated_forwards   

    discount_df['Forward Price'] = discount_df['Expiry'].map(lambda x: calibrated_forwards.get(x, S))
    discount_df['Forward Ratio'] = discount_df['Forward Price'] / S
    discount_df['Moneyness Forward'] = discount_df['Strike'] / discount_df['Forward Price'] - 1.0

    total_time = time.time() - start_time
    print(f"\nAnalysis completed in {total_time:.2f} seconds.")
    print(f"- Data preparation: {timings['pre_calibration']:.2f} seconds")
    print(f"- Calibration: {timings['calibration']:.2f} seconds")

    # Save to CSV if requested
    if params['save_output']:
        if term_structure is not None:
            term_structure.to_csv(params['output_file'], index=False)
            print(f"Term structure saved to {params['output_file']}")

        if discount_df is not None:
            discount_df.to_csv(params['iv_output_file'], index=False)
            print(f"Implied volatilities saved to {params['iv_output_file']}")
            
        if raw_df is not None:
            raw_df.to_csv(params['raw_output_file'], index=False)
            print(f"Raw options data saved to {params['raw_output_file']}")

    return term_structure, discount_df, raw_df, calibrated_forwards


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

    term_structure, discount_df, raw_df, forward_prices = calibrate(
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
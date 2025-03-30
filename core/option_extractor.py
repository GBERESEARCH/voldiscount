"""
Standalone option data extraction utility
Extracts option chain data from Yahoo Finance in a lean, functional way
"""
import pandas as pd
import numpy as np
import time
import yfinance as yf
from config.config import DEFAULT_PARAMS
from typing import Dict, Any

def extract_option_data(ticker, **kwargs):
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

    # Create function-specific parameters with defaults
    params: Dict[str, Any] = DEFAULT_PARAMS.copy()
    params.update(kwargs)

    try:
        # Get data from Yahoo Finance
        asset = yf.Ticker(ticker)
        
        # Extract spot price
        try:
            spot = asset.info['currentPrice']
        except KeyError:
            try:
                spot = (asset.info['bid'] + asset.info['ask'])/2
                if (abs(spot - asset.info['previousClose']) / asset.info['previousClose']) > 0.2:
                    spot = asset.info['previousClose']
            except KeyError:
                try:
                    spot = asset.info['navPrice']
                except:
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
                
            except Exception as e:
                print(f"Error processing {expiry}: {e}")
            
            # Wait to avoid rate limiting
            time.sleep(params['wait_time']) 
        
        # If no data found, return early
        if all_options.empty:
            return None, None, None, spot
        
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
        processed_data = _process_option_data(
            data=all_options, 
            min_days=params['min_days'], 
            min_volume=params['min_volume']
            )
        
        # Format for output
        formatted_data = _format_output(data=processed_data)
        
        return all_options, processed_data, formatted_data, spot
        
    except Exception as e:
        print(f"Error extracting option data for {ticker}: {e}")
        return None, None, None, None

def _process_option_data(data, **kwargs):
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
    params: Dict[str, Any] = DEFAULT_PARAMS.copy()
    params.update(kwargs)

    # Convert dates to datetime
    data['Last Trade Date'] = pd.to_datetime(data['Last Trade Date'])
    data['Expiry_datetime'] = pd.to_datetime(data['Expiry'])
    
    # Calculate days to expiry
    today = pd.to_datetime('today').date()
    data['TTM'] = (data['Expiry_datetime'] - pd.to_datetime(today)) / pd.Timedelta(days=365)
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

def _format_output(data):
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
    columns = ['Expiry', 'Strike', 'Last Trade Date', 'Last Price', 'Bid', 'Ask', 'Option Type']
    
    # Only include columns that exist
    valid_columns = [col for col in columns if col in data.columns]
    
    return data[valid_columns]

def from_paste_data(text_data):
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
    import io
    
    # Read the tab-separated data into a DataFrame
    df = pd.read_csv(io.StringIO(text_data), sep='\t')
    
    # Convert date columns if needed
    if 'Last Trade Date' in df.columns:
        df['Last Trade Date'] = pd.to_datetime(df['Last Trade Date'])
    
    if 'Expiry' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['Expiry']):
        df['Expiry'] = pd.to_datetime(df['Expiry'])
        
    return df

def create_option_data_with_rates(df, S, term_structure, reference_date):
    """
    Create a dataframe where each row is an option with the appropriate discount rate.
    """
    # Create a lookup dictionary for the discount rates
    rate_lookup = {row['Expiry']: row['Discount Rate'] 
                  for _, row in term_structure.iterrows()}
    
    # Create a list to store option data
    option_data = []
    
    for _, row in df.iterrows():
        expiry = row['Expiry']
        
        # Find matching discount rate
        if expiry in rate_lookup:
            discount_rate = rate_lookup[expiry]
            
            option_data.append({
                'Reference Date': reference_date,
                'Spot Price': S,
                'Expiry': expiry,
                'Days': row['Days To Expiry'],
                'Years': row['Years To Expiry'],
                'Strike': row['Strike'],  # Now using capitalized "Strike"
                'Option Type': row['Option Type'],
                'Price': row['Last Price'],
                'Discount Rate': discount_rate
            })
    
    # Create dataframe of option data with rates
    option_df = pd.DataFrame(option_data)
    
    return option_df

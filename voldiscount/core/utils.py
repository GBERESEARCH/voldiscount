"""
General utility functions
"""
import pandas as pd

def standardize_datetime(df, columns=None):
    """
    Standardize datetime columns in a DataFrame to be timezone-naive.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    columns : list, optional
        List of column names to standardize. 
        If None, attempts to standardize all datetime columns.
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with standardized datetime columns
    """
    # Create a copy to avoid modifying the original DataFrame
    df = df.copy()
    
    # If no columns specified, find datetime columns
    if columns is None:
        columns = df.select_dtypes(include=['datetime64']).columns
    
    for col in columns:
        if col in df.columns:
            try:
                # Convert to timezone-naive, preserving local time
                df[col] = pd.to_datetime(df[col], utc=False).dt.tz_localize(None)
            except TypeError:
                # Handle columns that might already be timezone-naive
                df[col] = pd.to_datetime(df[col], utc=False)
    
    return df

def load_options_data(filename):
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
    reference_date = last_trade_dates.max().date()
    
    print(f"Reference date: {reference_date}")
    
    # Add expiry metrics
    df['Days_To_Expiry'] = (df['Expiry'] - pd.Timestamp(reference_date)).dt.days
    df['Years_To_Expiry'] = df['Days_To_Expiry'] / 365.0
    
    return df, reference_date

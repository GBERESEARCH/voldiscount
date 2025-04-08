"""
General utility functions
"""
import pandas as pd
from datetime import datetime, date
from typing import List, Tuple, Optional, Union

def standardize_datetime(
    df: pd.DataFrame, 
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
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
        columns = df.select_dtypes(include=['datetime64']).columns #type: ignore
    
    for col in columns: #type: ignore
        if col in df.columns:
            try:
                # Convert to timezone-naive, preserving local time
                df[col] = pd.to_datetime(df[col], utc=False).dt.tz_localize(None)
            except TypeError:
                # Handle columns that might already be timezone-naive
                df[col] = pd.to_datetime(df[col], utc=False)
    
    return df


@staticmethod
def load_options_data(
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

"""
API endpoints for PCP calibration
"""
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import io
import json
from datetime import datetime, date

from voldiscount.calibrate import create_option_data_with_rates, create_combined_term_structure
from voldiscount.calibration.direct import direct_discount_rate_calibration
from voldiscount.calibration.smooth import smooth_curve_calibration
from voldiscount.config.config import DEFAULT_PARAMS
from voldiscount.core.option_extractor import extract_option_data
from voldiscount.core.utils import standardize_datetime

app = FastAPI(title="Options Volatility API")

# Add CORS middleware to allow requests from your React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your React app's address
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response validation with defaults from configuration
class CalibrationInput(BaseModel):
    underlying_price: Optional[float] = None  # Made optional for ticker-based endpoint
    initial_rate: float = Field(default=DEFAULT_PARAMS["initial_rate"])
    max_strike_diff_pct: float = Field(default=DEFAULT_PARAMS["max_strike_diff_pct"])
    min_option_price: float = Field(default=DEFAULT_PARAMS["min_option_price"])
    min_options_per_expiry: int = Field(default=DEFAULT_PARAMS["min_options_per_expiry"])
    consider_volume: bool = Field(default=DEFAULT_PARAMS["consider_volume"])
    min_pair_volume: int = Field(default=DEFAULT_PARAMS["min_pair_volume"])
    best_pair_only: bool = Field(default=DEFAULT_PARAMS["best_pair_only"])
    # Additional parameters for ticker-based calibration
    min_days: int = Field(default=DEFAULT_PARAMS["min_days"])
    min_volume: int = Field(default=DEFAULT_PARAMS["min_volume"])
    # Whether to include both calibration methods in output
    include_both_methods: bool = Field(default=True)

class CalibrationResult(BaseModel):
    direct_term_structure: List[Dict[str, Any]]
    smooth_term_structure: List[Dict[str, Any]]
    implied_volatilities: List[Dict[str, Any]]

class NumpyDateEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        try:
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, float):
                return round(obj, 2)
            elif isinstance(obj, np.floating):
                float_obj = float(obj)
                return round(float_obj, 5)
            elif isinstance(obj, (np.ndarray, pd.Series)):
                return obj.tolist()
            elif isinstance(obj, (datetime, date)):
                return obj.isoformat()
            elif isinstance(obj, (pd.DatetimeIndex)):
                return obj.date.tolist()
            elif isinstance(obj, (pd.DataFrame)):
                return obj.to_json()
        except TypeError:
            print("Error", obj)
        return json.JSONEncoder.default(self, obj)

@app.post("/csvcalibrate", response_model=CalibrationResult)
async def calibrate_options(
    file: UploadFile = File(...),
    params: str = Form(...)
) -> Dict[str, Any]:
    try:
        # Parse parameters
        params_dict = json.loads(params)
        calibration_input = CalibrationInput(**params_dict)
        
        # Read the uploaded file
        contents = await file.read()
        data_stream = io.BytesIO(contents)
        
        # Use pandas to read CSV
        df = pd.read_csv(data_stream)
        
        # Standardize datetime columns
        df = standardize_datetime(df, columns=['Expiry', 'Last Trade Date'])
        
        # Calculate days to expiry based on the last trade date
        if 'Last Trade Date' in df.columns:
            last_trade_dates = df['Last Trade Date']
            reference_date = last_trade_dates.max()
            
            df['Days To Expiry'] = (df['Expiry'] - reference_date).dt.days
            df['Years To Expiry'] = df['Days To Expiry'] / 365.0
        else:
            # Handle case where Last Trade Date is missing
            reference_date = datetime.now()
            df['Days To Expiry'] = (df['Expiry'] - pd.Timestamp(reference_date)).dt.days
            df['Years To Expiry'] = df['Days To Expiry'] / 365.0
        
        # Extract parameters
        S = calibration_input.underlying_price
        
        # Create calibration parameters dictionary from the model
        calibration_params = calibration_input.dict()
        # Remove fields that shouldn't be passed to calibration functions
        if 'underlying_price' in calibration_params:
            del calibration_params['underlying_price']
        if 'include_both_methods' in calibration_params:
            del calibration_params['include_both_methods']
       
        # Run direct calibration
        print("Running direct discount rate calibration...")
        direct_term_structure = direct_discount_rate_calibration(df, S, **calibration_params) #type: ignore
        
        # Run smooth curve calibration if requested
        print("Running smooth curve calibration...")
        # Add explicit type check before calling the function
        if S is None:
            raise ValueError("Underlying price (S) cannot be None for calibration")
        smooth_term_structure = smooth_curve_calibration(df, S, **calibration_params)
        
        if direct_term_structure.empty and smooth_term_structure.empty:
            raise HTTPException(status_code=400, detail="Failed to build term structure with either method")
        
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
        
        # Create combined term structure with both discount rates
        combined_term_structure = create_combined_term_structure(direct_term_structure, smooth_term_structure)
        
        # Calculate implied volatilities using the calibrated term structures
        iv_df = create_option_data_with_rates(
            df, S, combined_term_structure, reference_date, 
            include_both_rates=calibration_input.include_both_methods
        )
        
        # Add forward prices and moneyness data for both methods
        if not iv_df.empty:
            # Direct method fields
            iv_df['Direct Forward Price'] = iv_df['Expiry'].map(
                lambda x: direct_forwards.get(x, S) if direct_forwards else S
            )
            iv_df['Direct Forward Ratio'] = iv_df['Direct Forward Price'] / S
            iv_df['Direct Moneyness Forward'] = iv_df['Strike'] / iv_df['Direct Forward Price'] - 1.0
            
            # Smooth method fields
            iv_df['Smooth Forward Price'] = iv_df['Expiry'].map(
                lambda x: smooth_forwards.get(x, S) if smooth_forwards else S
            )
            iv_df['Smooth Forward Ratio'] = iv_df['Smooth Forward Price'] / S
            iv_df['Smooth Moneyness Forward'] = iv_df['Strike'] / iv_df['Smooth Forward Price'] - 1.0
        
        # Convert to dict then use the custom encoder
        direct_ts_data = json.loads(json.dumps(direct_term_structure.to_dict('records'), cls=NumpyDateEncoder))
        direct_ts_list = json.loads(json.dumps(direct_term_structure.to_dict(orient='list'), cls=NumpyDateEncoder))
        
        smooth_ts_data = json.loads(json.dumps(smooth_term_structure.to_dict('records'), cls=NumpyDateEncoder))
        smooth_ts_list = json.loads(json.dumps(smooth_term_structure.to_dict(orient='list'), cls=NumpyDateEncoder))
        
        iv_data = json.loads(json.dumps(iv_df.to_dict('records'), cls=NumpyDateEncoder))
        iv_data_list = json.loads(json.dumps(iv_df.to_dict(orient='list'), cls=NumpyDateEncoder))
        
        return {
            "direct_term_structure": direct_ts_data,
            "direct_term_structure_array": direct_ts_list,
            "smooth_term_structure": smooth_ts_data,
            "smooth_term_structure_array": smooth_ts_list,
            "implied_volatilities": iv_data,
            "implied_volatilities_array": iv_data_list
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing data: {str(e)}")


@app.get("/voldiscount")
async def get_discount_rates(
    request: Request
) -> Dict[str, Any]:
    """
    Calibrates options data fetched from Yahoo Finance with both direct and smooth methods.
    
    Parameters can be provided as query parameters (?ticker=AAPL&include_both_methods=true)
    
    Returns
    -------
    json
        Term structures and implied volatilities with both discount rates
    """
    # Extract all query parameters as dictionary
    query_params = dict(request.query_params)
    print(f"Received parameters: {query_params}")
    
    try:
        # Required parameter - must be present
        ticker = query_params.get('ticker')
        if not ticker:
            raise HTTPException(status_code=400, detail="Ticker symbol is required")
        
        # Extract and type-convert all other parameters with defaults from config
        # Numeric parameters
        underlying_price = float(query_params.get('underlying_price')) if 'underlying_price' in query_params else None #type: ignore
        try:
            initial_rate = request.query_params['initial_rate']
            initial_rate = float(initial_rate)
        except KeyError:
            initial_rate = DEFAULT_PARAMS['initial_rate']
        # initial_rate = float(query_params.get('initial_rate', DEFAULT_PARAMS['initial_rate']))
        max_strike_diff_pct = float(query_params.get('max_strike_diff_pct', DEFAULT_PARAMS['max_strike_diff_pct'])) #type: ignore
        min_option_price = float(query_params.get('min_option_price', DEFAULT_PARAMS['min_option_price'])) #type: ignore
        
        # Integer parameters
        min_options_per_expiry = int(query_params.get('min_options_per_expiry', DEFAULT_PARAMS['min_options_per_expiry'])) #type: ignore
        min_pair_volume = int(query_params.get('min_pair_volume', DEFAULT_PARAMS['min_pair_volume'])) #type: ignore
        min_days = int(query_params.get('min_days', DEFAULT_PARAMS['min_days'])) #type: ignore
        min_volume = int(query_params.get('min_volume', DEFAULT_PARAMS['min_volume'])) #type: ignore
        
        # Boolean parameters (handle various string representations)
        def parse_bool(val, default):
            if val is None:
                return default
            if isinstance(val, str):
                return val.lower() in ('true', 't', 'yes', 'y', '1')
            return bool(val)
        
        consider_volume = parse_bool(query_params.get('consider_volume'), DEFAULT_PARAMS['consider_volume'])
        best_pair_only = parse_bool(query_params.get('best_pair_only'), DEFAULT_PARAMS['best_pair_only'])
        monthlies = parse_bool(query_params.get('monthlies'), DEFAULT_PARAMS['monthlies'])
        include_both_methods = parse_bool(query_params.get('include_both_methods'), True)
        
        # Construct calibration parameters
        calibration_params = {
            'initial_rate': initial_rate,
            'max_strike_diff_pct': max_strike_diff_pct,
            'min_option_price': min_option_price,
            'min_options_per_expiry': min_options_per_expiry,
            'consider_volume': consider_volume,
            'min_pair_volume': min_pair_volume,
            'best_pair_only': best_pair_only,
            'min_days': min_days,
            'min_volume': min_volume,
            'monthlies': monthlies,
            'debug': True
        }
        
        # Log parameters for debugging
        print(f"Calibration parameters: {calibration_params}")
        
        # Fetch option data from yFinance
        _, df, fetched_price = extract_option_data(
            ticker, 
            min_days=min_days, 
            min_volume=min_volume
        )
        
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail=f"Failed to fetch valid options data for ticker {ticker}")
        
        # Determine underlying price (user-provided or fetched)
        S = underlying_price if underlying_price is not None else fetched_price
        if S is None:
            raise HTTPException(status_code=400, detail="Underlying price could not be determined")
            
        print(f"Using underlying price: {S}")
        
        # Standardize datetime columns
        df = standardize_datetime(df, columns=['Expiry', 'Last Trade Date'])
        
        # Calculate days to expiry based on the last trade date
        if 'Last Trade Date' in df.columns:
            last_trade_dates = df['Last Trade Date']
            reference_date = last_trade_dates.max()
            
            df['Days To Expiry'] = (df['Expiry'] - reference_date).dt.days
            df['Years To Expiry'] = df['Days To Expiry'] / 365.0
        else:
            # Handle case where Last Trade Date is missing
            reference_date = datetime.now()
            df['Days To Expiry'] = (df['Expiry'] - pd.Timestamp(reference_date)).dt.days
            df['Years To Expiry'] = df['Days To Expiry'] / 365.0
        
        # Run both calibration methods
        print("Running direct discount rate calibration...")
        direct_term_structure = direct_discount_rate_calibration(df, S, **calibration_params)
        
        print("Running smooth curve calibration...")
        smooth_term_structure = smooth_curve_calibration(df, S, **calibration_params)
        
        if direct_term_structure.empty and smooth_term_structure.empty:
            raise HTTPException(status_code=400, detail="Failed to build term structure with either method")
        
        # Standardize datetime in term structures
        direct_term_structure = standardize_datetime(direct_term_structure, columns=['Expiry'])
        smooth_term_structure = standardize_datetime(smooth_term_structure, columns=['Expiry'])

        # Extract forward prices from term structures
        direct_forwards = {row['Expiry']: row['Forward Price'] 
                       for _, row in direct_term_structure.iterrows() 
                       if 'Forward Price' in direct_term_structure.columns}
                       
        smooth_forwards = {row['Expiry']: row['Forward Price'] 
                       for _, row in smooth_term_structure.iterrows() 
                       if 'Forward Price' in smooth_term_structure.columns}

        # Create combined term structure with both discount rates
        combined_term_structure = create_combined_term_structure(direct_term_structure, smooth_term_structure)

        # Calculate implied volatilities with both discount rates
        iv_df = create_option_data_with_rates(
            df, S, combined_term_structure, reference_date, 
            include_both_rates=include_both_methods
        )
        
        # Add forward prices and moneyness data for both methods
        if not iv_df.empty:
            # Direct method fields
            iv_df['Direct Forward Price'] = iv_df['Expiry'].map(
                lambda x: direct_forwards.get(x, S) if direct_forwards else S
            )
            iv_df['Direct Forward Ratio'] = iv_df['Direct Forward Price'] / S
            iv_df['Direct Moneyness Forward'] = iv_df['Strike'] / iv_df['Direct Forward Price'] - 1.0
            
            # Smooth method fields
            iv_df['Smooth Forward Price'] = iv_df['Expiry'].map(
                lambda x: smooth_forwards.get(x, S) if smooth_forwards else S
            )
            iv_df['Smooth Forward Ratio'] = iv_df['Smooth Forward Price'] / S
            iv_df['Smooth Moneyness Forward'] = iv_df['Strike'] / iv_df['Smooth Forward Price'] - 1.0

        # Convert to dict then use JSON encoder
        direct_ts_data = json.loads(json.dumps(direct_term_structure.to_dict('records'), cls=NumpyDateEncoder))
        direct_ts_list = json.loads(json.dumps(direct_term_structure.to_dict(orient='list'), cls=NumpyDateEncoder))
        
        smooth_ts_data = json.loads(json.dumps(smooth_term_structure.to_dict('records'), cls=NumpyDateEncoder))
        smooth_ts_list = json.loads(json.dumps(smooth_term_structure.to_dict(orient='list'), cls=NumpyDateEncoder))
        
        iv_data = json.loads(json.dumps(iv_df.to_dict('records'), cls=NumpyDateEncoder))
        iv_data_list = json.loads(json.dumps(iv_df.to_dict(orient='list'), cls=NumpyDateEncoder))
        
        return {
            "direct_term_structure": direct_ts_data,
            "direct_term_structure_array": direct_ts_list,
            "smooth_term_structure": smooth_ts_data,
            "smooth_term_structure_array": smooth_ts_list,
            "implied_volatilities": iv_data,
            "implied_volatilities_array": iv_data_list
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing data: {str(e)}")


@app.get("/")
def health_check() -> Dict[str, str]:
    return {"status": "healthy"}
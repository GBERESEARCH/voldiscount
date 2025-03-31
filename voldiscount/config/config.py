# config.py
"""
Centralized parameters for option calibration
"""

DEFAULT_PARAMS = {
    # Pricing parameters
    "initial_rate": 0.05,           # Initial discount rate guess
    "min_rate": 0.0,                # Minimum allowable rate
    "max_rate": 0.2,                # Maximum allowable rate
    "fallback_growth": 0.03,        # Annual growth for fallback calculations
    'consider_volume': False,
    'reference_date': None,  # Default to None (use max trade date)
    'monthlies': True,  # Default to using only standard monthly expiries
    
    # Option selection parameters
    'option_type':'call', 
    'q': 0.0,
    'max_iterations': 50,
    "max_strike_diff_pct": 0.05,    # Maximum strike difference percentage
    "min_option_price": 0.0,        # Minimum option price
    'min_options_per_type': 3,  # Minimum options of each type (puts/calls) per expiry
    "min_options_per_expiry": 2,    # Minimum options required per type (puts/calls)
    "volatility_lower_bound": 0.001,
    "volatility_upper_bound": 10,
    "vol_lb_scalar": 0.5,
    "vol_ub_scalar": 1.5,
    
    # Extraction parameters
    "min_days": 7,                  # Minimum days to expiry
    "min_volume": 0, 
    "wait_time": 0.5,

    # Option Pair Selection
    'forward_prices': None, 
    'max_strike_diff_pct': 0.5, 
    'min_option_price': 0.0, 
    'min_pair_volume': 0, 
    'best_pair_only': False,
    'close_strike_min_pairs': 3,

    # Forward Pricing
    'debug_threshold': 0.0,
    'min_forward_ratio': 0.5, 
    'max_forward_ratio': 2.0,
    'min_price': 0.0,
    
    # Debug/output control
    "debug": True,                 # Enable debug output
    "save_output": False,           # Save results to files
    "skip_iv_calculation": True,    # Skip IV calculation for faster processing
    
    # Calibration method
    "use_forwards": True,           # Use forward prices for reference
    "calibration_method": "direct",  # 'joint' or 'direct'
    "min_int_rate": -0.20,
    "max_int_rate": 1.00,
}

# File paths
DEFAULT_PATHS = {
    "output_file": "term_structure.csv",
    "iv_output_file": "implied_volatilities.csv",
    "raw_output_file": "raw_options_data.csv",
}


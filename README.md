# Put-Call Parity Discount Rate Calibration Tool

## Overview

This application extracts, calibrates, and analyzes discount rates from option prices using put-call parity. It addresses a critical problem in options trading and volatility surface calibration: theoretical risk-free rates rarely align with market-implied discount rates across option expiries. The calibrated rates enable accurate implied volatility calculations and proper volatility surface construction.

## Technical Background

### Put-Call Parity and Discount Rates

Put-call parity is a fundamental arbitrage relationship in option pricing that must hold in efficient markets:

```
C - P = S * e^(-q*T) - K * e^(-r*T)
```

Where:
- C = Call price
- P = Put price
- S = Underlying price
- K = Strike price
- r = Risk-free interest rate
- q = Dividend/repo rate
- T = Time to expiry in years

By observing market prices of put-call pairs at equivalent strikes, we can solve for the discount rate (r) that satisfies this relationship. This market-implied rate typically varies across different option expiries, creating a term structure that more accurately reflects trading conditions than theoretical rates.

## Architecture

The application follows a modular design with specialized components:

1. **Interfaces**:
   - **CLI Interface** (`calibrate.py`): Command-line entry point
   - **Class Interface** (`voldiscount.py`): Object-oriented wrapper for programmatic use
   - **API Interface** (`calibration_api.py`): FastAPI-based REST endpoints

2. **Core Components**:
   - **Option Data Extraction** (`option_extractor.py`): Fetches option chains from Yahoo Finance
   - **Black-Scholes Pricing** (`black_scholes.py`): Option pricing and implied volatility calculations
   - **Put-Call Parity** (`pcp.py`): Implementation of parity relationships
   - **Utility Functions** (`utils.py`): Data standardization and processing

3. **Calibration Engine**:
   - **Direct Calibration** (`direct.py`): Per-tenor discount rate optimization
   - **Forward Pricing** (`forward_pricing.py`): Forward price calculation
   - **Interpolation** (`interpolation.py`): Rate interpolation for missing expiries
   - **Pair Selection** (`pair_selection.py`): Optimal put-call pair identification

4. **Configuration**:
   - **Centralized Parameters** (`config.py`): Default settings and constraints

## Implementation Details

### Discount Rate Calibration Algorithm

The direct calibration process follows these steps:

1. **Option Pair Selection**: For each expiry, identify put-call pairs with identical or similar strikes.
   - For exact strike matches: Use identical strikes for perfect parity.
   - For approximate matches: Use close strike pairs below a configurable threshold.
   - Weight pairs by liquidity (volume/open interest) and ATM proximity.

2. **Rate Optimization**: For each expiry, determine the discount rate that minimizes implied volatility differences between put-call pairs:
   ```python
   def objective_function(rate):
       put_iv = implied_volatility(put_price, S, K, T, rate, 'put')
       call_iv = implied_volatility(call_price, S, K, T, rate, 'call')
       return abs(put_iv - call_iv)
   ```

3. **Forward Price Calculation**: Derive forward prices from calibrated discount rates:
   ```
   Forward = Strike + (Call - Put) / discount_factor
   ```

4. **Term Structure Interpolation**: For expiries without sufficient option data:
   - Linear interpolation for intermediate expiries
   - Extrapolation for near and far expiries

### Implied Volatility Calculation

After discount rate calibration, the tool computes implied volatilities for all options using the calibrated term structure:

1. Map each option to its expiry-specific discount rate
2. Calculate implied volatility using the Black-Scholes model with the proper discount rate
3. Add forward-price-based moneyness metrics for accurate volatility surface construction

## Usage Guide

### Command Line Interface

```bash
python calibrate.py --ticker AAPL --price 175.0 --save
```

Key parameters:
- `--filename`: Path to CSV with option data (alternative to ticker)
- `--ticker`: Stock symbol for fetching live option data
- `--price`: Underlying price (optional, auto-fetched if not provided)
- `--rate`: Initial discount rate guess (default 0.05)
- `--min-days`: Minimum days to expiry (default 7)
- `--min-volume`: Minimum option volume (default 10)
- `--save`: Save results to CSV files
- `--monthlies`: Use only standard monthly expirations (default True)
- `--all-expiries`: Use all expiry dates (overrides --monthlies)

### Class Interface

```python
from voldiscount import VolDiscount

# Initialize with ticker
vd = VolDiscount(ticker="AAPL", underlying_price=175.0, save_output=True)

# Access results
term_structure = vd.get_term_structure()
option_data = vd.get_data_with_rates()
forward_prices = vd.get_forward_prices()
```

### API Interface

The application provides a FastAPI-based REST API with two main endpoints:

1. **CSV Upload Calibration**:
   ```
   POST /csvcalibrate
   ```
   Upload option data as CSV with calibration parameters.

2. **Ticker-Based Calibration**:
   ```
   GET /voldiscount?ticker=AAPL&underlying_price=175.0
   ```
   Fetches live data and returns calibrated discount rates and implied volatilities.

Response format:
```json
{
  "term_structure": [
    {
      "Expiry": "2023-04-21",
      "Days": 30,
      "Years": 0.0822,
      "Discount Rate": 0.0485,
      "Forward Price": 176.23,
      "Forward Ratio": 1.0070
    },
    ...
  ],
  "implied_volatilities": [
    {
      "Contract Symbol": "AAPL230421C00170000",
      "Expiry": "2023-04-21",
      "Strike": 170.0,
      "Option Type": "call",
      "Last Price": 7.85,
      "Implied Volatility": 0.2368,
      "Discount Rate": 0.0485,
      "Moneyness Forward": -0.0355
    },
    ...
  ]
}
```

## Configuration

The application uses a centralized configuration (`config.py`) with sensible defaults:

```python
DEFAULT_PARAMS = {
    # Pricing parameters
    "initial_rate": 0.05,
    "min_rate": 0.0,
    "max_rate": 0.2,
    
    # Option selection parameters
    "max_strike_diff_pct": 0.05,
    "min_option_price": 0.0,
    "min_options_per_expiry": 2,
    "volatility_lower_bound": 0.001,
    "volatility_upper_bound": 10,
    
    # Extraction parameters
    "min_days": 7,
    "min_volume": 0,
    
    # Forward Pricing
    "min_forward_ratio": 0.5,
    "max_forward_ratio": 2.0,
    
    # Boolean flags
    "debug": True,
    "save_output": False,
    "monthlies": True,
}
```

These parameters can be overridden through command-line arguments, class initialization parameters, or API request parameters.

## Practical Applications

The calibrated discount rates and implied volatilities enable:

1. **Accurate Volatility Surface Construction**: By using proper discount rates, the volatility surface better reflects true market conditions.

2. **Arbitrage Detection**: Significant deviations between put and call implied volatilities may indicate trading opportunities.

3. **Forward Price Analysis**: The term structure of forward prices provides insights into market expectations for dividends and financing costs.

4. **Risk-Free Rate Implied by Options**: The calibrated rates can be compared with Treasury yields to understand market financing conditions.

## Example Output

Term Structure:
```
Expiry      Days    Years   Discount Rate    Forward Price   Forward Ratio
2023-04-21  30      0.0822  0.0485           176.23          1.0070
2023-05-19  58      0.1589  0.0492           176.85          1.0106
2023-06-16  86      0.2356  0.0498           177.56          1.0146
2023-07-21  121     0.3315  0.0504           178.45          1.0197
2023-09-15  177     0.4849  0.0512           179.85          1.0277
2023-12-15  268     0.7342  0.0525           182.24          1.0414
2024-06-21  457     1.2521  0.0540           187.65          1.0723
```

This output shows the term structure of discount rates increasing with time to expiry, and forward prices reflecting the market's expectations for future dividends and financing costs.
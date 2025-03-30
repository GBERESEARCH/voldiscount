# Options Volatility Surface Calibration Tool

## Overview

This application calibrates discount rates from option prices using put-call parity relationships. These rates are essential inputs for volatility surface calibration, ensuring that at-the-money implied volatilities of puts and calls match at each tenor. The tool solves for a term structure of rates that minimize implied volatility differences between corresponding put and call options.

## Core Functionality

The application performs the following key functions:

1. **Options Data Processing**: Ingests options data from CSV files or directly from Yahoo Finance
2. **Discount Rate Calibration**: Identifies put-call option pairs and optimizes discount rates to satisfy put-call parity
3. **Implied Volatility Calculation**: Computes implied volatilities using the calibrated discount rates
4. **Term Structure Construction**: Builds a complete discount rate term structure across all option tenors

## Usage

### Command Line Interface

The primary entry point is `main.py`, which supports both file-based and ticker-based calibration:

```bash
# Calibrate from a CSV file
python main.py --filename options_data.csv --price 100.0

# Calibrate from Yahoo Finance
python main.py --ticker AAPL --min-days 7 --min-volume 10 --save

# Get help on available parameters
python main.py --help
```

#### Key Command Line Parameters

| Parameter | Description |
|-----------|-------------|
| `--filename` | Path to CSV file with options data |
| `--ticker` | Stock ticker symbol for Yahoo Finance data |
| `--price` | Underlying asset price (optional, will be estimated if not provided) |
| `--rate` | Initial discount rate guess (default: 0.05) |
| `--min-days` | Minimum days to expiry when fetching from ticker (default: 7) |
| `--min-volume` | Minimum volume when fetching from ticker (default: 10) |
| `--save` | Save results to CSV files |
| `--debug` | Enable detailed debug output |
| `--monthlies` | Use only standard monthly options (3rd Friday) (default: true) |
| `--all-expiries` | Use all available expiry dates (overrides `--monthlies`) |

### API Interface

The application also provides a FastAPI-based web API for integration with other systems:

```
# Start the API server
uvicorn calibration_api:app --reload
```

#### API Endpoints

1. **POST /api/calibrate**
   - Calibrates options data from an uploaded CSV file
   - Parameters:
     - `file`: CSV file with options data
     - `params`: JSON string with calibration parameters

2. **POST /api/calibrate-ticker**
   - Calibrates options data fetched from Yahoo Finance
   - Parameters:
     - `ticker`: Stock ticker symbol
     - `underlying_price`: (Optional) Override for underlying price
     - `params`: JSON string with calibration parameters

3. **GET /health**
   - Health check endpoint for monitoring

#### Calibration Parameters

The `params` JSON object accepts the following fields:

```json
{
  "underlying_price": 100.0,
  "initial_rate": 0.05,
  "max_strike_diff_pct": 0.05,
  "min_option_price": 0.0,
  "min_options_per_expiry": 2,
  "consider_volume": false,
  "min_pair_volume": 0,
  "best_pair_only": false
}
```

## Input Data Format

The CSV input file should contain the following columns:

| Column | Description |
|--------|-------------|
| Expiry | Option expiration date |
| Last Trade Date | Date of last trade (for reference date) |
| Strike | Option strike price |
| Option Type | `call` or `put` |
| Last Price | Option price |
| Volume | (Optional) Trading volume |
| Open Interest | (Optional) Open interest |

## Output

The calibration process produces:

1. **Term Structure**: Discount rate for each tenor, with method and diagnostic information
2. **Implied Volatilities**: Implied volatilities calculated for all options using the calibrated term structure

## Technical Implementation

### Architecture

The application is organized into modular components:

- `calibration/`: Contains rate calibration algorithms
- `core/`: Core mathematical functions and utilities
- `config/`: Configuration parameters
- API endpoints for web integration

### Calibration Methods

The primary calibration method is direct calibration, which:

1. Identifies ATM representative option pairs for each expiry date
2. Optimizes discount rates to minimize implied volatility differences for equal strikes
3. Uses put-call parity relationships for different strikes
4. Produces a point estimate for each tenor in the term structure

### Dependencies

Required Python packages:

```
numpy>=2.2.3
pandas>=2.2.3
scipy>=1.15.2
yfinance>=0.2.54
fastapi>=0.115.11
uvicorn>=0.34.0
python-multipart>=0.0.20
pydantic>=2.10.6
requests>=2.32.3
```

## Limitations

1. Quality of calibration depends on the quality and liquidity of input option data
2. Rate calibration is most reliable for tenors with liquid ATM options
3. Yahoo Finance data may have limitations for certain symbols or market conditions

## Example Usage

### Python API

```python
from main import main

# Run calibration with custom parameters
term_structure, iv_df, raw_df, _, _, _ = main(
    ticker="SPY",
    underlying_price=430.0,
    initial_rate=0.04,
    min_days=7,
    min_volume=10,
    debug=True
)

# Access calibrated rates
print(term_structure[['expiry_date', 'days', 'years', 'discount_rate']])
```
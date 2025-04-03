# VolDiscount: Option-Based Discount Rate Calibration Tool

## Overview

VolDiscount is a robust, high-performance tool for calibrating risk-neutral discount rates from options markets using put-call parity (PCP) relationships. The library extracts the term structure of discount rates implied by option prices, which can be used for:

- More accurate pricing of derivatives
- Calculation of consistent implied volatilities across strikes and expiries
- Extraction of forward prices
- Risk management and valuation adjustments

The library implements two complementary calibration methodologies, both executed simultaneously:

1. **Direct calibration**: Optimizes discount rates independently for each expiry, minimizing implied volatility differences between corresponding put and call pairs.

2. **Smooth curve calibration**: Implements a Nelson-Siegel parametric model that fits a consistent yield curve across all expiries simultaneously, ideal for noisy or inconsistent option data.

Both methods' results are provided in parallel, allowing direct comparison and selection of the most appropriate discount rate for each specific use case.

## Installation

### Prerequisites

- Python 3.13
- pandas
- numpy
- scipy
- yfinance (for ticker data fetching)
- fastapi, uvicorn (for API functionality)

### Install from source

```bash
git clone https://github.com/GBERESEARCH/voldiscount.git 
cd voldiscount
pip install -e .
```

## Core Features

- **Multiple data sources**: Process option data from CSV files or fetch directly from Yahoo Finance
- **Dual calibration execution**: Both direct per-expiry calibration and smooth Nelson-Siegel curve fitting run simultaneously
- **Comparative output**: Access both direct and smooth discount rates for each option in a single dataset
- **Discount rate extraction**: Calculate risk-neutral discount rates from option prices using put-call parity
- **Forward price determination**: Extract implied forward prices for each expiry from both methods
- **Implied volatility calculation**: Compute consistent implied volatilities using either calibrated discount rate
- **Multiple interfaces**: Access functionality via command-line, Python API, or HTTP API
- **Detailed diagnostics**: Comprehensive output including term structures, calibration metrics, and execution statistics

## Usage

### Command Line Interface

#### Calibrate from a ticker symbol:

```bash
python -m voldiscount.calibrate --ticker AAPL --min-days 7 --min-volume 10 --save --output aapl_term_structure.csv
```

#### Calibrate from a CSV file:

```bash
python -m voldiscount.calibrate --filename options_data.csv --price 150.75 --save
```

#### Command Line Arguments:

| Argument | Description | Default |
|----------|-------------|---------|
| `--filename` | Path to CSV file with options data | None |
| `--ticker` | Stock ticker to fetch option data for | None |
| `--price` | Underlying price | Auto-detected |
| `--rate` | Initial discount rate guess | 0.05 |
| `--min-days` | Minimum days to expiry when fetching from ticker | 7 |
| `--min-volume` | Minimum volume when fetching from ticker | 10 |
| `--output` | Output CSV file for term structure | term_structure.csv |
| `--iv-output` | Output CSV file for IVs | implied_volatilities.csv |
| `--debug` | Enable debug output | False |
| `--save` | Save results to CSV files | False |
| `--reference-date` | Reference date for options (YYYY-MM-DD) | Latest date |
| `--monthlies` | Use only standard monthly options (3rd Friday) | True |
| `--all-expiries` | Use all available expiry dates | False |

### Python API

```python
from voldiscount import VolDiscount

# Initialize from ticker
vd = VolDiscount(
    ticker='AAPL',
    min_days=7,
    min_volume=10,
    save_output=True,
    output_file='aapl_term_structure.csv'
)

# Get results
direct_term_structure = vd.get_direct_term_structure()
smooth_term_structure = vd.get_smooth_term_structure()
options_with_rates = vd.get_data_with_rates()
direct_forwards = vd.get_direct_forwards()
smooth_forwards = vd.get_smooth_forwards()

# Access both discount rates for a specific option
option = options_with_rates.iloc[0]
direct_rate = option['Direct Discount Rate']
smooth_rate = option['Smooth Discount Rate']

# Compute implied volatility using either rate
from voldiscount.core.black_scholes import implied_volatility

iv_direct = implied_volatility(
    price=option['Last Price'], 
    S=option['Spot Price'], 
    K=option['Strike'], 
    T=option['Years'], 
    r=direct_rate,
    option_type=option['Option Type'].lower(),
    q=0
)

iv_smooth = implied_volatility(
    price=option['Last Price'], 
    S=option['Spot Price'], 
    K=option['Strike'], 
    T=option['Years'], 
    r=smooth_rate,
    option_type=option['Option Type'].lower(),
    q=0
)
```

### HTTP API

Start the API server:

```bash
uvicorn voldiscount.calibration_api:app --reload
```

#### Endpoints:

1. `GET /voldiscount?ticker=AAPL`
   - Calibrates options data for the specified ticker using both methods
   - Returns both term structures and implied volatility data with both rate types

2. `POST /csvcalibrate`
   - Upload a CSV file with options data
   - Parameters passed as form data
   - Returns both term structures and implied volatility data with both rate types

#### API Response Structure:

```json
{
  "direct_term_structure": [
    {
      "Expiry": "2023-04-21",
      "Days": 45,
      "Years": 0.1233,
      "Discount Rate": 0.0521,
      "Put Strike": 150.0,
      "Call Strike": 150.0,
      "Put Price": 5.4,
      "Call Price": 6.75,
      "Put Implied Volatility": 0.2873,
      "Call Implied Volatility": 0.2854,
      "Implied Volatility Diff": 0.0019,
      "Forward Price": 151.45,
      "Forward Ratio": 1.0097
    },
    ...
  ],
  "smooth_term_structure": [
    {
      "Expiry": "2023-04-21",
      "Days": 45,
      "Years": 0.1233,
      "Discount Rate": 0.0501,
      "Forward Price": 151.26,
      "Forward Ratio": 1.0084
    },
    ...
  ],
  "implied_volatilities": [
    {
      "Contract Symbol": "AAPL230421C00150000",
      "Reference Date": "2023-03-07",
      "Last Trade Date": "2023-03-06",
      "Spot Price": 150.0,
      "Expiry": "2023-04-21",
      "Days": 45,
      "Years": 0.1233,
      "Strike": 150.0,
      "Option Type": "call",
      "Last Price": 6.75,
      "Bid": 6.6,
      "Ask": 6.9,
      "Open Interest": 15423,
      "Volume": 423,
      "Direct Discount Rate": 0.0521,
      "Smooth Discount Rate": 0.0501,
      "Direct Forward Price": 151.45,
      "Direct Forward Ratio": 1.0097,
      "Direct Moneyness Forward": -0.0096,
      "Smooth Forward Price": 151.26,
      "Smooth Forward Ratio": 1.0084,
      "Smooth Moneyness Forward": -0.0084
    },
    ...
  ]
}
```

## Input Data Format

### CSV File Format

The CSV input file should contain the following columns:

| Column | Description | Required |
|--------|-------------|----------|
| Expiry | Option expiry date (YYYY-MM-DD) | Yes |
| Strike | Strike price | Yes |
| Option Type | 'call' or 'put' | Yes |
| Last Price | Option price | Yes |
| Last Trade Date | Last trade date (YYYY-MM-DD) | Yes |
| Bid | Bid price | No |
| Ask | Ask price | No |
| Open Interest | Open interest | No |
| Volume | Trading volume | No |

## Output Format

### Term Structure CSV Files

Two term structure CSV files are generated:

#### Direct Term Structure (`term_structure_direct.csv`):
- `Expiry`: Expiry date
- `Days`: Days to expiry
- `Years`: Years to expiry
- `Discount Rate`: Calibrated discount rate
- `Put Strike`: Strike price of the put option used for calibration
- `Call Strike`: Strike price of the call option used for calibration
- `Put Price`: Price of the put option
- `Call Price`: Price of the call option
- `Put Implied Volatility`: Implied volatility of the put option
- `Call Implied Volatility`: Implied volatility of the call option
- `Implied Volatility Diff`: Absolute difference between put and call IVs
- `Forward Price`: Implied forward price
- `Forward Ratio`: Forward price / Spot price

#### Smooth Term Structure (`term_structure_smooth.csv`):
- `Expiry`: Expiry date
- `Days`: Days to expiry
- `Years`: Years to expiry
- `Discount Rate`: Calibrated discount rate using smooth curve method
- `Forward Price`: Implied forward price
- `Forward Ratio`: Forward price / Spot price
- `Method`: Calibration method ('smooth_curve')

### Implied Volatilities CSV

The implied volatilities CSV file contains one row per option with the following columns:

- `Contract Symbol`: Option contract symbol
- `Reference Date`: Reference date for pricing
- `Last Trade Date`: Last trade date of the option
- `Spot Price`: Underlying price
- `Expiry`: Expiry date
- `Days`: Days to expiry
- `Years`: Years to expiry
- `Strike`: Strike price
- `Option Type`: 'call' or 'put'
- `Last Price`: Option price
- `Bid`: Bid price
- `Ask`: Ask price
- `Open Interest`: Open interest
- `Volume`: Trading volume
- `Direct Discount Rate`: Discount rate from direct calibration
- `Smooth Discount Rate`: Discount rate from smooth curve calibration
- `Direct Forward Price`: Forward price from direct calibration
- `Direct Forward Ratio`: Forward price / Spot price from direct calibration
- `Direct Moneyness Forward`: (Strike / Direct Forward price) - 1.0
- `Smooth Forward Price`: Forward price from smooth calibration
- `Smooth Forward Ratio`: Forward price / Spot price from smooth calibration
- `Smooth Moneyness Forward`: (Strike / Smooth Forward price) - 1.0

## Configuration Parameters

Key configuration parameters with defaults can be found in `voldiscount/config/config.py`:

```python
DEFAULT_PARAMS = {
    # Pricing parameters
    "initial_rate": 0.05,           # Initial discount rate guess
    "min_rate": 0.0,                # Minimum allowable rate
    "max_rate": 0.2,                # Maximum allowable rate
    "fallback_growth": 0.03,        # Annual growth for fallback calculations
    'consider_volume': False,       # Whether to consider volume in pair selection
    'reference_date': None,         # Default to None (use max trade date)
    'monthlies': True,              # Default to using only standard monthly expiries
    
    # Option selection parameters
    'option_type':'call',           # Default option type
    'q': 0.0,                       # Dividend/repo rate
    'max_iterations': 50,           # Maximum iterations for IV calculation
    "max_strike_diff_pct": 0.05,    # Maximum strike difference percentage
    "min_option_price": 0.0,        # Minimum option price
    'min_options_per_type': 3,      # Minimum options of each type per expiry
    "min_options_per_expiry": 2,    # Minimum options required per type (puts/calls)
}
```

## Calibration Methods

Both calibration methods are executed simultaneously, providing two different discount rates for each option:

### Direct Calibration

The direct calibration method optimizes discount rates independently for each expiry:

1. For each expiry date, finds all valid put-call pairs based on strike matching criteria
2. Identifies the most ATM (at-the-money) put-call pair for each expiry
3. Optimizes the discount rate to minimize the implied volatility difference between the put and call
4. For expiries without valid option pairs, interpolates/extrapolates rates from surrounding expiries

This method is responsive to local market conditions and provides accurate results when option data is clean and consistent for each individual expiry.

### Smooth Curve Calibration

The smooth curve method fits a Nelson-Siegel parametric model across all expiries simultaneously:

1. Collects all valid put-call pairs across all expiries
2. Optimizes the parameters of the Nelson-Siegel yield curve model:
   - β₀: Long-term rate level
   - β₁: Short-term component
   - β₂: Medium-term component
   - τ: Decay factor
3. Weighs option pairs by moneyness (favoring ATM) and strike matching (favoring exact matches)
4. Produces a smooth, consistent term structure that can better handle noisy or inconsistent data

The Nelson-Siegel model equation:

```
r(t) = β₀ + β₁ * ((1 - e^(-t/τ))/(t/τ)) + β₂ * (((1 - e^(-t/τ))/(t/τ)) - e^(-t/τ))
```

### Choosing Between Methods

The simultaneous execution of both methods provides two sets of discount rates, allowing for:

1. **Comparison and validation**: Significant differences between methods may indicate data quality issues
2. **Method selection**: Choose the method that best suits specific requirements:
   - Direct method: When precise calibration to specific expiries is prioritized
   - Smooth method: When consistency across the term structure is more important
3. **Robustness checks**: Use both methods as a form of model validation
4. **Specific use cases**:
   - Use direct rates for pricing liquid options with high-quality data
   - Use smooth rates for illiquid options or when interpolating between expiries

## Troubleshooting

### Common Issues

1. **"No valid put-call pairs found"**
   - Increase `max_strike_diff_pct` to allow wider strike differences
   - Decrease `min_option_price` to include more options
   - Set `monthlies=False` to include all expiry dates

2. **"Failed to build term structure"**
   - Check if there are sufficient option pairs per expiry
   - Increase `min_days` to filter out very near-term options

3. **"Significant difference between direct and smooth rates"**
   - This may indicate market inefficiencies or data issues
   - Examine the implied volatility differences to determine which method is more appropriate
   - Consider using smooth rates when direct calibration shows high volatility differentials

## License

[MIT License](LICENSE)

## Contact

For issues and feature requests, please use the [GitHub issue tracker](https://github.com/GBERESEARCH/voldiscount/issues).
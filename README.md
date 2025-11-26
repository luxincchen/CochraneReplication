# Time-series predictability of stock returns 
A Python replication of Cochrane’s return predictability analysis, applied to modern data (1965–2023)

## Replication of Cochrane (2007, 2011)

This repository contains a replication of the return-predictability results in:

- Cochrane (2007) - Financial Markets and the Real Economy, Handbook of the Equity Risk Premium, Chapter 7

    --> Table 1: OLS regressions of excess returns and real dividend growth

- Cochrane (2011) - Presidential address: Discount rates, Journal of Finance

    --> Figure 1: Dividend yield (multiplied by 4) and following annualized 7-year return

Data used:
- Shiller S&P 500 dataset (1965–2023)
- Fama–French market returns (1965–2023)

The analysis examines whether the dividend-price ratio (D/P) predicts future stock returns and dividend growth, using **S&P 500 data (1965–2023)**. 
The project was completed as part of Advanced Investments at the University of Amsterdam, and was implemented in Python within a reproducible Jupyter Notebook.



## Key Findings

- The predictive power of D/P is **substantially weaker** in the modern sample (1965-2023) compared to Cochrane’s original results.
- The slope coefficients remain **positive** across horizons (1–5 years), but **t-statistics are mostly insignificant**, with R² values close to zero.
- Dividend growth predictability remains **near zero**, consistent with the classical finding that D/P variation does not reflect expected cashflow growth.
- The long-horizon co-movement between 4×D/P and future 7-year returns is **visible** before mid-1990s, although weaker than Cochrane's findings. This pattern cannot be observed after 2000. 
- The weakening of D/P predictability is partly due to the rise of **share buybacks**, which increasingly replace dividend payouts. As a result, dividend yield no longer captures a firm's total payout policy, reducing its relevance as a valuation ratio. 



## Methods Summary
1. Construct annual dividend-price ratio using real dividends and real prices (December observations).
2. Create long-horizon variables
- k-year log return: ∑ log(1 + Rₜ)
- k-year dividend growth: Δ log(Dₜ)
3. Estimate OLS regressions with Newey–West standard errors (lag = k−1).
4. Figure 1 replication
- Plot 4× dividend yield
- Plot the subsequent 7-year annualized total return



## Repository Structure
```
├── cochrane_replication.ipynb        # Main notebook: analysis, regressions, figures
├── cochrane_replication.py           # Clean modular Python implementation
├── shiller_data.xls                  # Shiller S&P 500 dataset (real prices & dividends)
├── F-F_Research_Data_Factors.csv     # Fama–French market excess returns
├── README.md                         
└── .DS_Store                         # (macOS system file – safe to delete, optional)
```



## How to Run
1. Install requirements:
   pip install pandas numpy statsmodels matplotlib
2. Run the script
   python cochrane_replication.py



## Skills Demonstrated
- Financial econometrics
- Time-series regression modeling with Newey-West standard errors
- Data cleaning, merging, and transformation of financial time series  




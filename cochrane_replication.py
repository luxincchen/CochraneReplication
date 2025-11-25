'''
Cochrane (2007, 2011) Replication
- Table 1 (Cochrane, 2007): Predictive regressions of returns and dividend growth on log(D/P)
- Figure 1(Cochrane, 2011): 4x D/P vs annualized following 7-year total return

Sample: 1965 - 2003
'''

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

'''

TABLE 1 COCHRANCE(2017)

'''

# ---------------------------------------------------------------------
# 1. DATA LOADING
# ---------------------------------------------------------------------

def load_shiller(shiller_path: str, start_year: int = 1965, end_year: int = 2023):
    # Sheet 1 contains data needed, first column = date (YYYY.MM)
    shiller = pd.read_excel(shiller_path, sheet_name=1)
    
    # Keep only rows with numeric dates 
    shiller.rename(columns={shiller.columns[0]: "date"}, inplace=True)
    shiller = shiller[pd.to_numeric(shiller['date'], errors='coerce').notnull()] 

    
    # Parse year/month from numeric date 
    shiller['year']  = shiller['date'].astype(str).str.slice(0,4).astype(int)
    shiller['month'] = shiller['date'].astype(str).str.slice(5,7).astype(int)

    # Rename columns
    shiller.columns.values[1] = "price"
    shiller.columns.values[2] = "dividend"
    shiller.columns.values[3] = "earnings"
    shiller.columns.values[4] = "CPI"
    shiller.columns.values[5] = "date_fractions"
    shiller.columns.values[6] = "GS10"
    shiller.columns.values[7] = "real_price"
    shiller.columns.values[8] = "real_dividend"
    shiller.columns.values[9] = "real_total_return_price"
    shiller.columns.values[10] = "real_earnings"
    shiller.columns.values[11] = "real_scaled_earnings"
    shiller.columns.values[12] = "CAPE"
    shiller.columns.values[13] = "TR_CAPE"
    
    P = shiller['real_price']
    D = shiller['real_dividend']

    shiller['dp_month'] = D / P
    
    # Keep December observations only
    december = shiller[shiller['month'] == 12].copy() 
    
    # Ensure numberic 
    december['real_dividend'] = pd.to_numeric(december['real_dividend'], errors='coerce')
    december['real_price']    = pd.to_numeric(december['real_price'], errors='coerce')
    
    december['dp_level'] = december['real_dividend'] / december['real_price']
    december['dp_log'] = np.log(december['real_dividend']) - np.log(december['real_price'])
    december = december[['year', 'dp_log', 'dp_level', 'real_dividend']]
    december.rename(columns={'dp_log': 'dp'}, inplace=True) 
    
    annual = december[(december['year'] >= 1965) & (december['year'] <= 2023)].copy()
    annual.rename(columns={'real_dividend': 'div'}, inplace=True)
    annual.set_index('year', inplace=True)
    
    return annual 

def load_fama_french(ff_path = str, start_year: int = 1965, end_year: int = 2023):

    ff = pd.read_csv(ff_path, skiprows=3)
    ff.rename(columns={ff.columns[0]: 'date'}, inplace=True)
    ff = ff[pd.to_numeric(ff['date'], errors='coerce').notna()].copy()

    # Convert the factor columns from strings to numbers 
    for col in ['Mkt-RF', 'SMB', 'HML', 'RF']:
        ff[col] = pd.to_numeric(ff[col], errors='coerce')
    
    ff = ff[ff['Mkt-RF'].notna()].copy()
        
    ff['year']  = ff['date'].astype(str).str.slice(0,4).astype(int)
    ff['month'] = ff['date'].astype(str).str.slice(4,6).astype(int)
    
    # Monthly returns 
    ff['mkt_excess'] = ff['Mkt-RF'] / 100.0
    
    # Annual excess
    ff_annual = ff.groupby('year')['mkt_excess'].sum().to_frame('Rx')
    ff_annual = ff_annual.loc[1965:2023]
    
    return ff_annual

# ---------------------------------------------------------------------
# 2. REGRESSIONS
# ---------------------------------------------------------------------

def make_horizon_vars(data: pd.DataFrame, k:int):
    out = data.copy()
    
    out['div'] = pd.to_numeric(out['div'], errors='coerce')
    
    out['logret'] = np.log(1 + out['Rx'])
    out['logDiv'] = np.log(out['div'])
    
    
    # Cumulative k-year log return, shifted up to align with year t 
    out[f'Rk_{k}'] = out['logret'].rolling(k).sum().shift(-k)
    
    # Cumulative k-year log return, shifted up to align with year t 
    out[f'dg_{k}'] = (out['logDiv'].shift(-k) - out['logDiv']) / k
        #we're moving date up (towards the past) to align future values with today 
    return out 

def forecast_regression(df: pd.DataFrame, y_col: str, x_col: str, k: int):
    '''
    Run y = a + b x with Newey-West (HAC) standard errors (lags = k-1)
    '''
    sub = df[[y_col, x_col]].dropna().astype(float)
    y = sub[y_col]
    x = sm.add_constant(sub[x_col])

    model = sm.OLS(y, x)
    res = model.fit(cov_type = 'HAC', cov_kwds={'maxlags': k-1})
    
    b = res.params[x_col]
    tb = res.tvalues[x_col]
    R2 = res.rsquared
    
    return b, tb, R2

# ---------------------------------------------------------------------
# 3. TABLE 1
# ---------------------------------------------------------------------

def build_table1(data: pd.DataFrame, horizons=(1,2,3,5)):
    '''
    Left panel: regressions for returns
    Right panel: dividend growth
    '''
    df = data.copy()
    for k in horizons:
        df = make_horizon_vars(df, k)
    
    results_ret = []
    results_dg = []
    
    for k in horizons:
        b, tb, R2 = forecast_regression(df, f'Rk_{k}', 'dp', k)
        results_ret.append({'k': k, 'b':b, 't(b)': tb, 'R2': R2})
        
        b_D, tb_D, R2_D = forecast_regression(df, f'dg_{k}', 'dp', k)
        results_dg.append({'k':k, 'b': b_D, 't(b)': tb_D, 'R2': R2_D})
        
    table_ret = pd.DataFrame(results_ret)
    table_dg = pd.DataFrame(results_dg)
    table1 = table_ret.merge(table_dg, on="k")
    
    return table1 

# ---------------------------------------------------------------------
# 4. FIGURE 1
# ---------------------------------------------------------------------

def build_figure1(data: pd.DataFrame, k: int = 7):
    
    '''
    Plot 4 x D/P (level) and the following k-year annualized total return
    '''
    df = data.copy()
    
    data['R7_cum'] = (1 + data['Rx']).shift(-1).rolling(7).apply(np.prod, raw=True) - 1
    data['R7_ann'] = (1 + data['R7_cum']) ** (1/7) - 1

    plot_df = data[['dp_level', 'R7_ann']].dropna()


    plt.figure(figsize=(8,5))
    
    # 4 × dividend yield (in %)
    plt.plot(plot_df.index, 4 * plot_df['dp_level'] * 100, label='4 × D/P', linewidth=1.5)
    
    # 7-year forward annualized return (in %)
    plt.plot(plot_df.index, plot_df['R7_ann'] * 100, label='Annualized 7-year return', linewidth=1.5)
    
    plt.xlabel('Year')
    plt.ylabel('Percent')
    plt.title('4 × D/P and Annualized Following 7-Year Return')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------------
# 5. MAIN SCRIPT
# ---------------------------------------------------------------------

def main(shiller_path: str, ff_path: str, start_year: int = 1965, end_year = 2023):
    # Load data 
    shiller_ann = load_shiller(shiller_path, start_year, end_year)
    ff_ann = load_fama_french(ff_path, start_year, end_year)
    
    # Merge data 
    data = shiller_ann.join(ff_ann, how="inner")
    
    # Table 1
    table1 = build_table1(data)
    table1.to_csv("table1.csv", index=False, float_format="%.6f")
    
    # Figure 1
    build_figure1(data, k=7)

if __name__ == "__main__":
    # Replace with your local paths or pass via command line / config
    SHILLER_PATH = "shiller_data.xls"
    FF_PATH = "F-F_Research_Data_Factors.csv"
    main(SHILLER_PATH, FF_PATH)

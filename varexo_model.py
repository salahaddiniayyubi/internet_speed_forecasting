import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.api import VARMAX, VAR
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import bootstrap
import warnings
warnings.filterwarnings('ignore')

# Configuration
FILE_PATH = 'fixed_data.csv'
TARGET_COL = 'Download Speed Mbps'
FORECAST_HORIZON = 7
CONF_LEVEL = 0.95
N_BOOT = 1000
PLOT_START_DATE = '2024-11-01'

def load_and_preprocess():
    """Load and preprocess data with ISP-specific speeds"""
    df = pd.read_csv(FILE_PATH)
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter for median values
    df = df[df['metric_type'] == 'median']
    
    # Get top 10 ISPs by user count (adjust number as needed)
    isp_list = df[df['Provider Name'] != 'All Providers Combined']\
        .groupby('Provider Name')['User Count'].sum().nlargest(10).index.tolist()
    
    # Create wide format dataframe
    wide_df = df.pivot_table(
        index='date', 
        columns='Provider Name', 
        values=[TARGET_COL, 'Upload Speed Mbps', 'User Count', 'Sample Count']
    )
    wide_df.columns = [f"{col[1]}_{col[0].replace(' ', '_')}" for col in wide_df.columns]
    wide_df = wide_df.ffill().bfill()
    
    return wide_df, isp_list

def make_stationary(df, target_cols):
    """Apply first differences and test stationarity"""
    stationary_df = df.diff().dropna()
    
    # ADF test for stationarity
    for col in target_cols:
        series = stationary_df[col].dropna()
        if series.nunique() <= 1:
            print(f"[SKIP] Column '{col}' is constant or all NaN after differencing. Skipping ADF test.")
            continue
        try:
            result = adfuller(series)
            print(f"ADF Statistic ({col}): {result[0]:.4f}")
            print(f"p-value: {result[1]:.4f}")
            if result[1] > 0.05:
                print(f"Warning: {col} may still be non-stationary")
        except Exception as e:
            print(f"[ERROR] ADF test failed for column '{col}': {e}")
    
    return stationary_df

def create_lagged_exog(df, exog_cols, max_lag):
    """
    Create lagged versions of exogenous variables.
    df: DataFrame containing all data
    exog_cols: list of exogenous variable column names
    max_lag: maximum lag to create
    Returns: DataFrame with lagged exogenous variables
    """
    lagged = pd.DataFrame(index=df.index)
    for col in exog_cols:
        for lag in range(1, max_lag + 1):
            lagged[f'{col}_lag{lag}'] = df[col].shift(lag)
    return lagged

def varexo_modelling(stationary_df, isp_list, exog_cols, max_lag=2):
    """Build and analyze VARMAX model with exogenous variables"""
    # Select target variables
    target_cols = ["All Providers Combined_Download_Speed_Mbps"] + \
                [f"{isp}_{TARGET_COL.replace(' ', '_')}" for isp in isp_list]
    
    # Create lagged exogenous variables
    exog_data = create_lagged_exog(stationary_df, exog_cols, max_lag)
    df_with_exog = pd.concat([stationary_df, exog_data], axis=1).dropna()
    
    # Split data
    train_size = int(len(df_with_exog) * 0.8)
    train = df_with_exog.iloc[:train_size]
    test = df_with_exog.iloc[train_size:]
    
    # Prepare endogenous and exogenous variables
    endog_train = train[target_cols]
    exog_train = train.drop(columns=target_cols)
    endog_test = test[target_cols]
    exog_test = test.drop(columns=target_cols)
    
    # Fit VARMAX model
    print(f"\nFitting VARMAX model with {len(exog_train.columns)} exogenous variables, max_lag={max_lag}")
    model = VARMAX(endog_train, exog=exog_train, order=(max_lag, 0))
    model_fit = model.fit(disp=False)
    
    print("\nVARMAX Model Summary:")
    print(model_fit.summary())
    
    # Forecast
    forecast = model_fit.forecast(steps=len(exog_test), exog=exog_test)
    
    # Calculate metrics
    metrics = {}
    for i, col in enumerate(target_cols):
        y_true = endog_test[col]
        y_pred = forecast.iloc[:, i]
        metrics[col] = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
    
    return model_fit, metrics, forecast, endog_test

def var_modelling(stationary_df, isp_list):
    """Build and analyze standard VAR model (without exogenous variables)"""
    # Select target variables
    target_cols = ["All Providers Combined_Download_Speed_Mbps"] + \
               [f"{isp}_{TARGET_COL.replace(' ', '_')}" for isp in isp_list]
    
    # Split data
    train_size = int(len(stationary_df) * 0.8)
    train = stationary_df[target_cols].iloc[:train_size]
    test = stationary_df[target_cols].iloc[train_size:]
    
    # Determine optimal lag order
    model = VAR(train)
    n_obs = len(train)
    n_vars = len(target_cols)
    maxlags = min(15, (n_obs - 1) // n_vars, 7)
    if maxlags < 1:
        maxlags = 1
    print(f"Using maxlags={maxlags} for VAR order selection (n_obs={n_obs}, n_vars={n_vars})")
    lag_results = model.select_order(maxlags)
    optimal_lags = lag_results.aic
    
    print(f"\nOptimal lag order (AIC): {optimal_lags}")
    
    # Fit VAR model
    var_model = model.fit(optimal_lags)
    print("\nVAR Model Summary:")
    print(var_model.summary())
    
    # Forecast on test set
    input_data = train.values[-var_model.k_ar:]
    forecast = var_model.forecast(input_data, steps=len(test))
    forecast_df = pd.DataFrame(forecast, index=test.index, columns=target_cols)
    
    # Calculate metrics
    metrics = {}
    for col in target_cols:
        y_true = test[col]
        y_pred = forecast_df[col]
        metrics[col] = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
    
    return var_model, metrics, forecast_df, test

def compare_models(var_metrics, varexo_metrics, target_cols):
    """Compare performance of VAR and VARMAX models"""
    comparison = {}
    for col in target_cols:
        comparison[col] = {
            'var_rmse': var_metrics[col]['rmse'],
            'varexo_rmse': varexo_metrics[col]['rmse'],
            'rmse_improvement': ((var_metrics[col]['rmse'] - varexo_metrics[col]['rmse']) / var_metrics[col]['rmse']) * 100,
            
            'var_mae': var_metrics[col]['mae'],
            'varexo_mae': varexo_metrics[col]['mae'],
            'mae_improvement': ((var_metrics[col]['mae'] - varexo_metrics[col]['mae']) / var_metrics[col]['mae']) * 100,
            
            'var_r2': var_metrics[col]['r2'],
            'varexo_r2': varexo_metrics[col]['r2'],
            'r2_improvement': varexo_metrics[col]['r2'] - var_metrics[col]['r2']
        }
    return comparison

def plot_model_comparison(var_forecast, varexo_forecast, actual, target_col):
    """Plot comparison of VAR and VARMAX forecasts"""
    plt.figure(figsize=(16, 9))
    
    # Plot actual values
    plt.plot(actual.index, actual[target_col], label='Actual Values', color='navy', linewidth=2.5)
    
    # Plot VAR forecast
    plt.plot(var_forecast.index, var_forecast[target_col], 
             label='VAR Forecast', color='#ff7f0e', linewidth=2, linestyle='--')
    
    # Plot VARMAX forecast
    plt.plot(varexo_forecast.index, varexo_forecast[target_col], 
             label='VARMAX with Exogenous Variables', color='#2ca02c', linewidth=2, linestyle='-.')
    
    # Formatting
    plt.title(f'Model Comparison: {target_col} Forecasts', fontsize=16, pad=20)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Download Speed (Mbps)', fontsize=14)
    
    # Format x-axis dates
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(interval=1))
    plt.xticks(rotation=45)
    
    plt.legend(loc='best', frameon=True)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('var_varexo_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main execution function"""
    # Step 1: Load and preprocess data
    wide_df, isp_list = load_and_preprocess()
    print(f"Loaded data with shape {wide_df.shape}")
    print(f"Top ISPs: {isp_list}")
    
    # Step 2: Transform to stationary data
    target_cols = ["All Providers Combined_Download_Speed_Mbps"] + \
                [f"{isp}_{TARGET_COL.replace(' ', '_')}" for isp in isp_list]
    stationary_df = make_stationary(wide_df, target_cols)
    
    # Step 3: Identify exogenous variables
    exog_cols = [
        "All Providers Combined_User_Count",
        "All Providers Combined_Sample_Count"
    ]
    print(f"Using exogenous variables: {exog_cols}")
    
    # Step 4: Train standard VAR model
    print("\n===== Standard VAR Model (No Exogenous Variables) =====")
    var_model, var_metrics, var_forecast, test_data = var_modelling(stationary_df, isp_list)
    
    print("\nVAR Model Metrics:")
    for col, metrics in var_metrics.items():
        print(f"{col}:")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  R²: {metrics['r2']:.4f}")
    
    # Step 5: Train VARMAX model with exogenous variables
    print("\n===== VARMAX Model with Exogenous Variables =====")
    varexo_model, varexo_metrics, varexo_forecast, test_data_exog = varexo_modelling(
        stationary_df, isp_list, exog_cols, max_lag=2)
    
    print("\nVARMAX Model Metrics:")
    for col, metrics in varexo_metrics.items():
        print(f"{col}:")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  R²: {metrics['r2']:.4f}")
    
    # Step 6: Compare models
    comparison = compare_models(var_metrics, varexo_metrics, target_cols)
    
    print("\n===== Model Comparison =====")
    for col, metrics in comparison.items():
        print(f"\n{col}:")
        print(f"  RMSE: VAR={metrics['var_rmse']:.4f}, VARMAX={metrics['varexo_rmse']:.4f}, Improvement={metrics['rmse_improvement']:.2f}%")
        print(f"  MAE: VAR={metrics['var_mae']:.4f}, VARMAX={metrics['varexo_mae']:.4f}, Improvement={metrics['mae_improvement']:.2f}%")
        print(f"  R²: VAR={metrics['var_r2']:.4f}, VARMAX={metrics['varexo_r2']:.4f}, Difference={metrics['r2_improvement']:.4f}")
    
    # Step 7: Plot comparison for the main target
    main_target = "All Providers Combined_Download_Speed_Mbps"
    plot_model_comparison(var_forecast, varexo_forecast, test_data, main_target)
    
    # Save results
    comparison_results = pd.DataFrame({
        'variable': [k for k in comparison.keys()],
        'var_rmse': [v['var_rmse'] for v in comparison.values()],
        'varexo_rmse': [v['varexo_rmse'] for v in comparison.values()],
        'rmse_improvement_pct': [v['rmse_improvement'] for v in comparison.values()],
        'var_mae': [v['var_mae'] for v in comparison.values()],
        'varexo_mae': [v['varexo_mae'] for v in comparison.values()],
        'mae_improvement_pct': [v['mae_improvement'] for v in comparison.values()],
        'var_r2': [v['var_r2'] for v in comparison.values()],
        'varexo_r2': [v['varexo_r2'] for v in comparison.values()],
        'r2_improvement': [v['r2_improvement'] for v in comparison.values()]
    })
    
    comparison_results.to_csv('var_varexo_comparison.csv', index=False)
    print("\nComparison results saved to 'var_varexo_comparison.csv'")
    
    return comparison_results

if __name__ == "__main__":
    main()
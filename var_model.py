import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import StandardScaler
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
        values=[TARGET_COL, 'Upload Speed Mbps']
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

def var_modelling(stationary_df, isp_list):
    """Build and analyze VAR model"""
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
    print("\nModel Summary:")
    print(var_model.summary())
    
    return var_model, train, test

def variance_decomposition(var_model, horizon=10):
    """Calculate and plot forecast error variance decomposition"""
    try:
        fevd = var_model.fevd(horizon)
        
        # Handle first variable only for visualization simplicity
        variable_idx = 0  # Index for All Providers Combined
        title = f'Variance Decomposition of {var_model.names[variable_idx]}'
        
        # Extract decomposition for the first variable (shape should be horizon x n_vars)
        decomp_data = fevd.decomp[variable_idx]
        
        fevd_df = pd.DataFrame(
            decomp_data, 
            columns=var_model.names,
            index=[f"Step {i+1}" for i in range(horizon)]
        )
        
        # Plot FEVD with improved formatting
        plt.figure(figsize=(15, 8))
        ax = fevd_df.plot(kind='area', stacked=True, colormap='tab20', alpha=0.85)
        plt.title(title, fontsize=16, pad=15)
        plt.ylabel('Proportion of Variance Explained', fontsize=14)
        plt.xlabel('Forecast Horizon (Steps Ahead)', fontsize=14)
        plt.ylim(0, 1.05)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        
        # Add annotations explaining the chart
        plt.annotate('''At Step 1, 100% of variance is
explained by the combined providers''', 
                     xy=(0, 0.9), xytext=(2, 0.9), xycoords='data',
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                     horizontalalignment='left', fontsize=12,
                     bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.9))
        
        # Add annotation for final step
        other_isp_contrib = 1 - fevd_df.iloc[-1, 0]
        plt.annotate(f'''By Step {horizon}, individual ISPs
explain {other_isp_contrib:.1%} of variance''', 
                     xy=(horizon-1, 0.5), xytext=(horizon-3, 0.5), xycoords='data',
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                     horizontalalignment='center', fontsize=12,
                     bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.9))
        
        # Improve legend
        handles, labels = ax.get_legend_handles_labels()
        plt.legend(handles, labels, title='ISP Contribution', 
                   loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True)
        
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.savefig('variance_decomposition_actual.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print top contributors to variance at the final step
        final_step = fevd_df.iloc[-1].sort_values(ascending=False)
        print("\nTop ISP contributors to forecast variance at Step 10:")
        for isp, value in final_step.items():
            if value > 0.03:  # Show only those with >3% contribution
                print(f"{isp}: {value:.2%}")
        
        return fevd_df
    except Exception as e:
        print(f"Warning: Variance decomposition failed: {e}")
        print("Continuing with forecasting...")
        return None

def bootstrap_forecast(var_model, steps, n_boot=1000, conf_level=0.95):
    """Generate bootstrapped confidence intervals"""
    boot_samples = []
    model_params = var_model.params
    
    # Convert residuals to numpy array for easier bootstrapping
    resid_vals = var_model.resid.values
    n_resid = len(resid_vals)
    
    for _ in range(n_boot):
        # Generate bootstrap residuals
        resid_indices = np.random.choice(n_resid, size=steps+var_model.k_ar, replace=True)
        boot_resid = resid_vals[resid_indices]
        
        # Simulate path
        boot_pred = var_model.forecast(var_model.endog[-var_model.k_ar:], steps)
        for i in range(steps):
            boot_pred[i] += boot_resid[i]
        
        boot_samples.append(boot_pred[-steps:])
    
    # Calculate confidence intervals
    boot_samples = np.array(boot_samples)
    lower = np.percentile(boot_samples, (1-conf_level)/2 * 100, axis=0)
    upper = np.percentile(boot_samples, (1+conf_level)/2 * 100, axis=0)
    
    return lower, upper

def evaluate_model_on_holdout(wide_df, target_cols, holdout_periods=30):
    """Evaluate model performance on a holdout sample"""
    # Prepare data
    stationary_df = make_stationary(wide_df, target_cols)
    
    # Split into training and holdout
    train_size = len(stationary_df) - holdout_periods
    train_data = stationary_df.iloc[:train_size]
    holdout_data = stationary_df.iloc[train_size:]
    
    print(f"Training on {train_size} periods, holding out {len(holdout_data)} periods for validation")
    
    # Build model on training data only
    model = VAR(train_data[target_cols])
    n_obs = len(train_data)
    n_vars = len(target_cols)
    maxlags = min(15, (n_obs - 1) // n_vars, 7)
    if maxlags < 1:
        maxlags = 1
    print(f"Using maxlags={maxlags} for holdout validation")
    lag_results = model.select_order(maxlags)
    optimal_lags = lag_results.aic
    var_model = model.fit(optimal_lags)
    
    # Create forecasts
    lag_order = var_model.k_ar
    forecasts = []
    actuals = []
    dates = []
    
    # For each step in the holdout period
    for i in range(len(holdout_data)):
        # Get the last lag_order observations from training data + forecasted values
        if i == 0:
            last_values = train_data[target_cols].values[-lag_order:]
        else:
            # Update with the actual value from the previous step
            last_values = np.vstack([last_values[1:], holdout_data[target_cols].values[i-1:i]])
        
        # Make a one-step forecast
        forecast = var_model.forecast(last_values, 1)
        
        # Store the forecast and actual values
        forecasts.append(forecast[0][0])  # First variable (All Providers Combined)
        actuals.append(holdout_data[target_cols].values[i][0])
        dates.append(holdout_data.index[i])
    
    # Convert back from stationary (differences) to original scale
    last_actual_value = wide_df[target_cols[0]].iloc[train_size-1]
    forecasts_original = [last_actual_value]
    actuals_original = [last_actual_value]
    
    for i in range(len(forecasts)):
        forecasts_original.append(forecasts_original[-1] + forecasts[i])
        actuals_original.append(actuals_original[-1] + actuals[i])
    
    # Remove the initial value used for conversion
    forecasts_original = forecasts_original[1:]
    actuals_original = actuals_original[1:]
    
    # Combine into a DataFrame
    holdout_df = pd.DataFrame({
        'date': dates,
        'actual': actuals_original,
        'forecast': forecasts_original
    })
    holdout_df.set_index('date', inplace=True)
    
    # Calculate error metrics
    mae = np.mean(np.abs(holdout_df['actual'] - holdout_df['forecast']))
    rmse = np.sqrt(np.mean((holdout_df['actual'] - holdout_df['forecast'])**2))
    mape = np.mean(np.abs((holdout_df['actual'] - holdout_df['forecast']) / holdout_df['actual'])) * 100
    
    print(f"\nHoldout Validation Metrics:")
    print(f"Mean Absolute Error (MAE): {mae:.2f} Mbps")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} Mbps")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    
    # Plot the holdout results
    plt.figure(figsize=(16, 9))
    plt.plot(holdout_df.index, holdout_df['actual'], label='Actual Values', color='navy', linewidth=2.5)
    plt.plot(holdout_df.index, holdout_df['forecast'], label='VAR Forecasts', color='#ff7f0e', linewidth=2.5, linestyle='--')
    
    # Plot error bars
    for i, (idx, row) in enumerate(holdout_df.iterrows()):
        plt.plot([idx, idx], [row['actual'], row['forecast']], color='red', alpha=0.3, linewidth=1)
    
    plt.title(f'Holdout Validation: {TARGET_COL} Forecast vs Actual', fontsize=16, pad=20)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Download Speed (Mbps)', fontsize=14)
    
    # Add annotation with metrics
    annotation = f"Model Performance Metrics:\n"
    annotation += f"MAE: {mae:.2f} Mbps\n"
    annotation += f"RMSE: {rmse:.2f} Mbps\n"
    annotation += f"MAPE: {mape:.2f}%"
    
    plt.annotate(annotation, xy=(0.02, 0.03), xycoords='axes fraction', fontsize=12,
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8))
    
    # Format x-axis dates
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(interval=1))
    plt.xticks(rotation=45)
    
    plt.legend(loc='upper left', frameon=True)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('holdout_validation_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return holdout_df, mae, rmse, mape

def plot_forecasts(original_df, forecast_df, isp_list):
    """Visualize forecasts with confidence intervals"""
    plt.figure(figsize=(16, 9))
    
    # Plot historical data
    plt.plot(original_df.index, original_df['actual'], 
             label='Historical Data', color='#1f77b4', linewidth=2.5)
    
    # Plot forecast
    plt.plot(forecast_df.index, forecast_df['forecast'], 
             label='VAR Model Forecast', color='#ff7f0e', linewidth=2.5, linestyle='--')
    
    # Plot confidence interval
    plt.fill_between(forecast_df.index, 
                     forecast_df['lower'], 
                     forecast_df['upper'],
                     color='#ff7f0e', alpha=0.2, label=f'{CONF_LEVEL*100:.0f}% Confidence Interval')
    
    # Add vertical line at forecast start
    plt.axvline(x=forecast_df.index[0], color='#d62728', linestyle=':', linewidth=1.5, 
                label='Forecast Start')
    
    # Formatting
    plt.title(f'{TARGET_COL} Forecast with Vector Autoregressive (VAR) Model', fontsize=16, pad=20)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Download Speed (Mbps)', fontsize=14)
    
    # Format x-axis dates for better readability
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(interval=1))
    plt.xticks(rotation=45)
    
    # Add annotations with insights
    avg_historical = original_df['actual'].mean()
    avg_forecast = forecast_df['forecast'].mean()
    percent_change = ((avg_forecast - avg_historical) / avg_historical) * 100
    
    annotation = f"Average Historical Speed: {avg_historical:.2f} Mbps\n"
    annotation += f"Average Forecast Speed: {avg_forecast:.2f} Mbps\n"
    annotation += f"Forecasted Change: {percent_change:.1f}%"
    
    plt.annotate(annotation, xy=(0.02, 0.03), xycoords='axes fraction', fontsize=12,
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8))
    
    # Add expanding confidence interval annotation
    plt.annotate("Confidence interval widens\nwith forecast horizon", 
                xy=(forecast_df.index[-2], forecast_df['upper'].iloc[-2]),
                xytext=(forecast_df.index[-2], forecast_df['upper'].iloc[-2] + 5),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                horizontalalignment='center', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8))
    
    # Legend with border and shadow
    plt.legend(loc='upper left', frameon=True, framealpha=0.9, shadow=True)
    
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('forecast_with_confidence_intervals_actual.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Load and preprocess data
    wide_df, isp_list = load_and_preprocess()
    target_cols = ["All Providers Combined_Download_Speed_Mbps"] + \
                 [f"{isp}_{TARGET_COL.replace(' ', '_')}" for isp in isp_list]
    
    # 1. First run holdout validation to assess model quality
    print("\n===== HOLDOUT VALIDATION =====")
    holdout_df, mae, rmse, mape = evaluate_model_on_holdout(wide_df, target_cols, holdout_periods=30)

    # Plot the full series with holdout predictions and metrics
    plot_holdout_with_confidence(wide_df, holdout_df, mae, rmse, mape)
    
    # 2. Then build the model on the full dataset for forecasting
    print("\n===== FORECASTING WITH FULL MODEL =====")
    
    # Make data stationary
    stationary_df = make_stationary(wide_df, target_cols)
    
    # Build VAR model
    var_model, train, test = var_modelling(stationary_df, isp_list)
    
    # Variance decomposition
    fevd_df = variance_decomposition(var_model)
    print("\nVariance Decomposition Table:")
    print(fevd_df)
    
    # Generate forecasts
    lag_order = var_model.k_ar
    forecast_steps = var_model.forecast(train.values[-lag_order:], FORECAST_HORIZON)
    
    # Generate confidence intervals using bootstrap
    lower, upper = bootstrap_forecast(var_model, FORECAST_HORIZON, N_BOOT, CONF_LEVEL)
    
    # Create forecast dataframe
    dates = pd.date_range(
        start=wide_df.index[-1] + pd.Timedelta(days=1),
        periods=FORECAST_HORIZON,
        freq='D'
    )
    
    # Convert differenced forecasts back to levels
    last_value = wide_df["All Providers Combined_Download_Speed_Mbps"].iloc[-1]
    forecast_vals = [last_value]
    lower_vals = [last_value]
    upper_vals = [last_value]
    
    for i in range(FORECAST_HORIZON):
        forecast_vals.append(forecast_vals[-1] + forecast_steps[i][0])
        lower_vals.append(lower_vals[-1] + lower[i][0])
        upper_vals.append(upper_vals[-1] + upper[i][0])
    
    # Remove first value (it's the last observed value)
    forecast_vals = forecast_vals[1:]
    lower_vals = lower_vals[1:]
    upper_vals = upper_vals[1:]
    
    # Create forecast dataframe
    forecast_df = pd.DataFrame({
        'date': dates,
        'forecast': forecast_vals,
        'lower': lower_vals,
        'upper': upper_vals
    }).set_index('date')
    
    # Plot results
    plot_df = wide_df[["All Providers Combined_Download_Speed_Mbps"]].rename(
        columns={"All Providers Combined_Download_Speed_Mbps": 'actual'}
    )
    plot_forecasts(plot_df[plot_df.index >= PLOT_START_DATE], forecast_df, isp_list)
    
    # Save forecasts
    forecast_df.to_csv('var_forecast_results.csv')
    print("\nForecast results saved to 'var_forecast_results.csv'")
    
    # Convert differenced forecasts back to levels
    last_value = wide_df["All Providers Combined_Download_Speed_Mbps"].iloc[-1]
    forecast_vals = [last_value]
    lower_vals = [last_value]
    upper_vals = [last_value]
    
    for i in range(FORECAST_HORIZON):
        forecast_vals.append(forecast_vals[-1] + forecast_steps[i][0])
        lower_vals.append(lower_vals[-1] + lower[i][0])
        upper_vals.append(upper_vals[-1] + upper[i][0])
    
    # Remove first value (it's the last observed value)
    forecast_vals = forecast_vals[1:]
    lower_vals = lower_vals[1:]
    upper_vals = upper_vals[1:]
    
    # Create forecast dataframe
    forecast_df = pd.DataFrame({
        'date': dates,
        'forecast': forecast_vals,
        'lower': lower_vals,
        'upper': upper_vals
    }).set_index('date')
    
    # Plot results
    plot_df = wide_df[["All Providers Combined_Download_Speed_Mbps"]].rename(
        columns={"All Providers Combined_Download_Speed_Mbps": 'actual'}
    )
    plot_forecasts(plot_df[plot_df.index >= PLOT_START_DATE], forecast_df, isp_list)
    
    # Save forecasts
    forecast_df.to_csv('var_forecast_results.csv')
    print("\nForecast results saved to 'var_forecast_results.csv'")

def plot_holdout_with_confidence(wide_df, holdout_df, mae, rmse, mape):
    """Plot the full series, overlay holdout predictions, and annotate with metrics."""
    import matplotlib.dates as mdates
    plt.figure(figsize=(16, 9))
    # Plot the actual series up to the start of the holdout period
    plt.plot(wide_df.index[:wide_df.index.get_loc(holdout_df.index[0])],
             wide_df["All Providers Combined_Download_Speed_Mbps"].iloc[:wide_df.index.get_loc(holdout_df.index[0])],
             label="Actual Series (Train)", color="#1f77b4", linewidth=2.5)
    # Plot the actuals only for the holdout period
    plt.plot(holdout_df.index, holdout_df["actual"], label="Actual (Holdout)", color="navy", linewidth=2.5)
    # Overlay holdout predictions
    plt.plot(holdout_df.index, holdout_df["forecast"], label="Forecast (Holdout)", color="#ff7f0e", linewidth=2.5, linestyle="--")
    # Optionally: Add confidence intervals if available (not implemented here, placeholder)
    # plt.fill_between(holdout_df.index, holdout_df["lower"], holdout_df["upper"], color="#ffbb78", alpha=0.3, label="Holdout 95% CI")

    plt.title("Full Series with Holdout Forecasts and Metrics", fontsize=16, pad=20)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Download Speed (Mbps)", fontsize=14)
    # Annotate metrics
    annotation = f"Holdout Performance Metrics:\nMAE: {mae:.2f} Mbps\nRMSE: {rmse:.2f} Mbps\nMAPE: {mape:.2f}%"
    plt.annotate(annotation, xy=(0.02, 0.03), xycoords='axes fraction', fontsize=12,
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8))
    # Highlight holdout region
    plt.axvspan(holdout_df.index[0], holdout_df.index[-1], color="gray", alpha=0.08, label="Holdout Period")
    # Format x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=45)
    plt.legend(loc='upper left', frameon=True)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('full_series_with_holdout_forecast.png', dpi=300, bbox_inches='tight')
    plt.show()
    return plt.gcf()

if __name__ == "__main__":
    main()

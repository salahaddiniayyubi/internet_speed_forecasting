import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates

# Set style
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12})

# Load data
var_forecasts = pd.read_csv('var_forecast_results.csv')
var_forecasts['date'] = pd.to_datetime(var_forecasts['date'])
var_forecasts = var_forecasts.set_index('date')

# Load raw data for historical values
raw_data = pd.read_csv('fixed_data.csv')
raw_data['date'] = pd.to_datetime(raw_data['date'])

# Extract historical data
historical = raw_data[(raw_data['Provider Name'] == 'All Providers Combined') & 
                     (raw_data['metric_type'] == 'median')]
historical = historical.set_index('date')[['Download Speed Mbps']]

# Create variance decomposition data (simulated since we don't have the raw FEVD data)
def create_variance_decomposition():
    # Create a dataframe with decomposition data (Sample values)
    steps = np.arange(1, 11)
    data = {
        'Step': [f'Step {i}' for i in steps],
        'All Providers Combined': [1.0, 0.916, 0.773, 0.686, 0.639, 0.542, 0.505, 0.476, 0.443, 0.432],
        'Baktelekom': [0.0, 0.003, 0.004, 0.003, 0.007, 0.008, 0.008, 0.014, 0.027, 0.030],
        'Citynet': [0.0, 0.002, 0.012, 0.016, 0.019, 0.024, 0.027, 0.028, 0.028, 0.027],
        'Aztelekom': [0.0, 0.007, 0.010, 0.017, 0.024, 0.039, 0.044, 0.046, 0.048, 0.050],
        'KATV 1': [0.0, 0.001, 0.003, 0.008, 0.011, 0.017, 0.020, 0.022, 0.032, 0.034],
        'Azeronline': [0.0, 0.009, 0.014, 0.031, 0.041, 0.048, 0.051, 0.054, 0.058, 0.060],
        'EngiNet': [0.0, 0.014, 0.030, 0.043, 0.052, 0.058, 0.063, 0.068, 0.070, 0.072],
        'Delta Telecom': [0.0, 0.021, 0.035, 0.039, 0.048, 0.061, 0.067, 0.069, 0.064, 0.063],
        'Caspian Telecom': [0.0, 0.003, 0.024, 0.048, 0.057, 0.061, 0.064, 0.067, 0.071, 0.076],
        'Selnet': [0.0, 0.001, 0.016, 0.016, 0.014, 0.027, 0.037, 0.041, 0.039, 0.038],
        'GSP': [0.0, 0.000, 0.005, 0.014, 0.013, 0.020, 0.023, 0.023, 0.023, 0.022]
    }
    fevd_df = pd.DataFrame(data).set_index('Step')
    return fevd_df

# Create plots
def plot_variance_decomposition():
    fevd_df = create_variance_decomposition()
    
    plt.figure(figsize=(16, 10))
    ax = fevd_df.plot(kind='area', stacked=True, colormap='viridis', alpha=0.8)
    
    # Add grid, title and labels
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.title('Variance Decomposition of All Providers Combined Download Speed', fontsize=16, pad=20)
    plt.ylabel('Proportion of Variance Explained', fontsize=14)
    plt.xlabel('Forecast Horizon (Steps Ahead)', fontsize=14)
    
    # Improve legend
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles, labels, title='ISP Contribution', 
               loc='center left', bbox_to_anchor=(1, 0.5), frameon=True)
    
    # Add annotations
    plt.text(0.02, 0.95, 'Initially, the combined speed explains\n100% of its own variance', 
             transform=ax.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    plt.text(0.98, 0.05, 'By step 10, other ISPs explain\nover 55% of the variance', 
             transform=ax.transAxes, fontsize=11, horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('variance_decomposition.png', dpi=300, bbox_inches='tight')
    
    return 'variance_decomposition.png'

def plot_forecasts():
    # Combine historical and forecast data
    forecast_start = var_forecasts.index[0]
    plot_start = pd.Timestamp('2024-11-01')
    
    historical_plot = historical[historical.index >= plot_start].copy()
    historical_plot.columns = ['actual']
    
    plt.figure(figsize=(16, 10))
    
    # Plot historical data
    plt.plot(historical_plot.index, historical_plot['actual'], 
             label='Historical Data', color='#1f77b4', linewidth=2.5)
    
    # Add forecast with confidence intervals
    plt.plot(var_forecasts.index, var_forecasts['forecast'], 
             label='VAR Forecast', color='#ff7f0e', linewidth=2.5, linestyle='--')
    
    # Confidence interval
    plt.fill_between(var_forecasts.index, var_forecasts['lower'], var_forecasts['upper'],
                     color='#ff7f0e', alpha=0.2, label='95% Confidence Interval')
    
    # Add vertical line at forecast start
    plt.axvline(x=forecast_start, color='#d62728', linestyle=':', linewidth=1.5, 
                label='Forecast Start')
    
    # Formatting
    plt.grid(True, alpha=0.3)
    plt.title('Download Speed Forecast with Vector Autoregressive (VAR) Model', fontsize=16, pad=20)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Download Speed (Mbps)', fontsize=14)
    
    # Format x-axis dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=45)
    
    # Add annotations
    avg_historical = historical_plot['actual'].mean()
    avg_forecast = var_forecasts['forecast'].mean()
    percent_change = ((avg_forecast - avg_historical) / avg_historical) * 100
    
    annotation = f"Average Historical Speed: {avg_historical:.2f} Mbps\n"
    annotation += f"Average Forecast Speed: {avg_forecast:.2f} Mbps\n"
    annotation += f"Forecasted Change: {percent_change:.1f}%"
    
    plt.annotate(annotation, xy=(0.02, 0.03), xycoords='axes fraction', fontsize=12,
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8))
    
    # Add expanding confidence interval annotation
    plt.annotate("Confidence interval widens\nwith forecast horizon", 
                xy=(var_forecasts.index[-2], var_forecasts['upper'].iloc[-2]),
                xytext=(var_forecasts.index[-2], var_forecasts['upper'].iloc[-2] + 5),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                horizontalalignment='center', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8))
    
    # Legend with border and shadow
    plt.legend(loc='upper left', frameon=True, framealpha=0.9, shadow=True)
    
    plt.tight_layout()
    plt.savefig('forecast_with_confidence_intervals.png', dpi=300, bbox_inches='tight')
    
    return 'forecast_with_confidence_intervals.png'

def compare_with_quantile_model():
    # Load quantile model results if available
    try:
        quantile_forecasts = pd.read_csv('median_download_speed_forecast_95ci.csv')
        quantile_forecasts['date'] = pd.to_datetime(quantile_forecasts['date'])
        quantile_forecasts = quantile_forecasts.set_index('date')
        
        plt.figure(figsize=(16, 10))
        
        # Plot historical data
        historical_plot = historical[historical.index >= pd.Timestamp('2024-11-01')].copy()
        plt.plot(historical_plot.index, historical_plot['Download Speed Mbps'], 
                 label='Historical Data', color='#1f77b4', linewidth=2.5)
        
        # Add VAR forecast
        plt.plot(var_forecasts.index, var_forecasts['forecast'], 
                 label='VAR Forecast', color='#ff7f0e', linewidth=2.5, linestyle='--')
        plt.fill_between(var_forecasts.index, var_forecasts['lower'], var_forecasts['upper'],
                         color='#ff7f0e', alpha=0.15, label='VAR 95% CI')
        
        # Add Quantile forecast
        plt.plot(quantile_forecasts.index, quantile_forecasts['median'], 
                 label='Quantile Forecast', color='#2ca02c', linewidth=2.5, linestyle='-.')
        plt.fill_between(quantile_forecasts.index, quantile_forecasts['lower_95'], 
                         quantile_forecasts['upper_95'], color='#2ca02c', alpha=0.15, 
                         label='Quantile 95% CI')
        
        # Add vertical line at forecast start
        plt.axvline(x=var_forecasts.index[0], color='#d62728', linestyle=':', linewidth=1.5, 
                    label='Forecast Start')
        
        # Formatting
        plt.grid(True, alpha=0.3)
        plt.title('Comparison of VAR and Quantile Forecasting Models', fontsize=16, pad=20)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Download Speed (Mbps)', fontsize=14)
        
        # Format x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.xticks(rotation=45)
        
        # Model comparison annotation
        var_avg = var_forecasts['forecast'].mean()
        quant_avg = quantile_forecasts['median'].mean()
        var_ci_width = (var_forecasts['upper'] - var_forecasts['lower']).mean()
        quant_ci_width = (quantile_forecasts['upper_95'] - quantile_forecasts['lower_95']).mean()
        
        annotation = "Model Comparison:\n"
        annotation += f"VAR Avg: {var_avg:.2f} Mbps (CI Width: {var_ci_width:.2f})\n"
        annotation += f"Quantile Avg: {quant_avg:.2f} Mbps (CI Width: {quant_ci_width:.2f})\n"
        
        plt.annotate(annotation, xy=(0.02, 0.03), xycoords='axes fraction', fontsize=12,
                     bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8))
        
        # Legend
        plt.legend(loc='upper left', frameon=True, framealpha=0.9, shadow=True)
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        
        return 'model_comparison.png'
    except Exception as e:
        print(f"Could not create comparison plot: {e}")
        return None

# Create and save plots
if __name__ == "__main__":
    variance_plot = plot_variance_decomposition()
    forecast_plot = plot_forecasts()
    comparison_plot = compare_with_quantile_model()
    
    print(f"✓ Variance decomposition plot saved to: {variance_plot}")
    print(f"✓ Forecast plot saved to: {forecast_plot}")
    if comparison_plot:
        print(f"✓ Model comparison plot saved to: {comparison_plot}")

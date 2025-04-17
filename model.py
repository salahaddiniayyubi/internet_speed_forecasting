import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def load_and_preprocess_data(file_path):
    """Load and preprocess internet speed data, using MEDIAN values"""
    # Load data
    df = pd.read_csv(file_path)
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter for "All Providers Combined" data with MEDIAN metric type
    all_providers_df = df[(df['Provider Name'] == 'All Providers Combined') & 
                         (df['metric_type'] == 'median')]
    
    # Sort by date and reset index
    all_providers_df = all_providers_df.sort_values('date').reset_index(drop=True)
    
    print(f"Data range: {all_providers_df['date'].min()} to {all_providers_df['date'].max()}")
    print(f"Total data points: {len(all_providers_df)}")
    print(f"Using MEDIAN values for forecasting")
    
    return all_providers_df

def create_features(df, target_col='Download Speed Mbps'):
    """Create enhanced time-based features for forecasting with deterministic trend"""
    df_features = df.copy()
    
    # Create date features
    df_features['dayofweek'] = df_features['date'].dt.dayofweek
    df_features['is_weekend'] = df_features['dayofweek'].isin([5, 6]).astype(int)
    df_features['month'] = df_features['date'].dt.month
    df_features['year'] = df_features['date'].dt.year
    df_features['day'] = df_features['date'].dt.day
    df_features['quarter'] = df_features['date'].dt.quarter
    
    # Deterministic trend components
    df_features['days_since_start'] = (df_features['date'] - df_features['date'].min()).dt.days
    # Add polynomial trend terms
    df_features['trend_squared'] = df_features['days_since_start'] ** 2
    df_features['trend_cubed'] = df_features['days_since_start'] ** 3
    # Add cyclic features for weekly and monthly patterns
    df_features['weekofyear'] = df_features['date'].dt.isocalendar().week.astype(int)
    df_features['sin_day'] = np.sin(2 * np.pi * df_features['dayofweek']/7)
    df_features['cos_day'] = np.cos(2 * np.pi * df_features['dayofweek']/7)
    df_features['sin_month'] = np.sin(2 * np.pi * df_features['month']/12)
    df_features['cos_month'] = np.cos(2 * np.pi * df_features['month']/12)
    
    # Create lag features
    for lag in [1, 2, 3, 7, 14]:  # Add 14-day lag
        df_features[f'{target_col}_lag_{lag}'] = df_features[target_col].shift(lag)
    
    # Create rolling window features
    for window in [3, 7, 14]:
        df_features[f'{target_col}_rolling_mean_{window}'] = df_features[target_col].rolling(window=window).mean()
        df_features[f'{target_col}_rolling_median_{window}'] = df_features[target_col].rolling(window=window).median()
        df_features[f'{target_col}_rolling_std_{window}'] = df_features[target_col].rolling(window=window).std()
        df_features[f'{target_col}_rolling_min_{window}'] = df_features[target_col].rolling(window=window).min()
        df_features[f'{target_col}_rolling_max_{window}'] = df_features[target_col].rolling(window=window).max()
    
    # Create difference features
    df_features[f'{target_col}_diff_1'] = df_features[target_col].diff(1)
    df_features[f'{target_col}_diff_7'] = df_features[target_col].diff(7)
    df_features[f'{target_col}_pct_change_1'] = df_features[target_col].pct_change(1)
    df_features[f'{target_col}_pct_change_7'] = df_features[target_col].pct_change(7)
    
    # Outlier handling - cap at 1st/99th percentile to avoid extreme values
    q_low = df_features[target_col].quantile(0.01)
    q_high = df_features[target_col].quantile(0.99)
    df_features[target_col] = df_features[target_col].clip(q_low, q_high)
    
    # Drop rows with NaN values (due to lag and rolling features)
    df_features.dropna(inplace=True)
    
    return df_features

def split_data(df, test_size=0.15, val_size=0.15):
    """Split data into training, validation, and test sets chronologically"""
    train_end = int(len(df) * (1 - test_size - val_size))
    val_end = int(len(df) * (1 - test_size))
    
    train_data = df.iloc[:train_end]
    val_data = df.iloc[train_end:val_end]
    test_data = df.iloc[val_end:]
    
    print(f"Train set: {len(train_data)} samples")
    print(f"Validation set: {len(val_data)} samples")
    print(f"Test set: {len(test_data)} samples")
    
    return train_data, val_data, test_data

from sklearn.model_selection import ParameterSampler
from sklearn.metrics import mean_absolute_error

def train_quantile_models(X_train, y_train, X_val, y_val):
    """Enhanced quantile training with asymmetric parameter tuning"""
    quantiles = [0.025, 0.5, 0.975]
    models = {}
    
    # Define specific parameter grids for different quantiles
    # These are tailored to the characteristics of each quantile
    lower_param_grid = {
        'max_depth': [3, 4, 5],
        'learning_rate': [0.05, 0.1, 0.15],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'n_estimators': [200, 300, 400],
        'min_child_weight': [3, 5]
    }
    
    median_param_grid = {
        'max_depth': [4, 5, 6],
        'learning_rate': [0.03, 0.05, 0.1],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9],
        'n_estimators': [300, 400, 500],
        'min_child_weight': [3, 5, 7]
    }
    
    upper_param_grid = {
        'max_depth': [3, 4, 5],
        'learning_rate': [0.1, 0.15, 0.2],
        'subsample': [0.7, 0.8],
        'colsample_bytree': [0.7, 0.8],
        'n_estimators': [400, 500, 600],  # More estimators for better tail capture
        'min_child_weight': [3, 5]
    }
    
    # Map quantiles to their specific parameter grids
    param_grids = {
        0.025: lower_param_grid,
        0.5: median_param_grid,
        0.975: upper_param_grid
    }
    
    # Train separate models for each quantile with quantile-specific hyperparameter tuning
    for q in quantiles:
        print(f"\nPerforming hyperparameter tuning for {q*100:.1f}% quantile model...")
        best_loss = float('inf')
        best_params = None
        
        # Sample parameter combinations for this quantile
        n_iter = 10 if q == 0.5 else 8  # More iterations for median model
        for params in list(ParameterSampler(param_grids[q], n_iter=n_iter, random_state=int(q*100))):
            model = xgb.XGBRegressor(
                objective='reg:quantileerror',
                quantile_alpha=q,
                eval_metric='mae',
                early_stopping_rounds=20,  # Increased from 15
                random_state=42,
                **params
            )
            
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            # For upper and lower quantiles, evaluate on quantile-specific loss
            preds = model.predict(X_val)
            
            # For median, use standard MAE; for quantiles, use quantile-specific loss
            if q == 0.5:
                loss = mean_absolute_error(y_val, preds)
            else:
                # Quantile loss: q * (y - pred) if y > pred else (1-q) * (pred - y)
                errors = y_val - preds
                loss = np.mean(np.where(errors >= 0, q * errors, (q-1) * errors))
            
            if loss < best_loss:
                best_loss = loss
                best_params = params
                print(f"  New best loss for {q*100:.1f}%: {best_loss:.4f} with params: {best_params}")
        
        print(f"Best parameters for {q*100:.1f}% quantile: {best_params}")
        
        # Train final model with best parameters for this quantile
        model = xgb.XGBRegressor(
            objective='reg:quantileerror',
            quantile_alpha=q,
            eval_metric='mae',
            early_stopping_rounds=20,
            random_state=42,
            **best_params
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        models[f'q_{q}'] = model
        
        # Evaluate on validation set
        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        print(f"  Final {q*100:.1f}% quantile MAE: {mae:.4f}")
        
        # Print the range of predictions to check for issues
        print(f"  Prediction range: {np.min(preds):.2f} to {np.max(preds):.2f} (mean: {np.mean(preds):.2f})")
    
    return models

def update_features_for_quantile(current_data, pred, target_col):
    # Update lag features
    for lag in [1, 2, 3]:
        if lag == 1:
            current_data[f'{target_col}_lag_{lag}'] = pred
        else:
            current_data[f'{target_col}_lag_{lag}'] = current_data.get(f'{target_col}_lag_{lag-1}', pred)
    # 7-day lag
    if f'{target_col}_lag_7' in current_data:
        current_data[f'{target_col}_lag_7'] = current_data[f'{target_col}_lag_6'] if f'{target_col}_lag_6' in current_data else pred
    # Rolling features (simulate with pred for new values)
    for window in [3, 7]:
        current_data[f'{target_col}_rolling_mean_{window}'] = pred
        current_data[f'{target_col}_rolling_std_{window}'] = 0
    # Difference features
    current_data[f'{target_col}_diff_1'] = 0
    current_data[f'{target_col}_diff_7'] = 0
    return current_data

def quantile_forecast(models, last_data, feature_cols, target_col, days=7):
    """Improved forecasting with quantile-specific feature propagation and exogenous lags."""
    # Exogenous variables to propagate
    exo_vars = ['User Count', 'Sample Count', 'Test Count']
    exo_lags = [1, 2, 3, 7]
    exo_rolls = [3, 7]

    # Initialize forecast dataframe
    future_dates = pd.date_range(
        start=last_data['date'].values[0] + pd.Timedelta(days=1),
        periods=days,
        freq='D'
    )
    
    quantile_forecasts = {}
    for q_name, model in models.items():
        print(f"Generating {q_name} forecast with feature propagation...")
        current_data = last_data.copy()
        future_preds = []
        # For exogenous variables, keep a history buffer for lags/rolls
        exo_histories = {var: [current_data[var].values[0]] for var in exo_vars}
        for i, future_date in enumerate(future_dates):
            # Update date features
            current_data['date'] = future_date
            current_data['dayofweek'] = future_date.dayofweek
            current_data['is_weekend'] = int(future_date.dayofweek in [5, 6])
            current_data['month'] = future_date.month
            current_data['year'] = future_date.year
            current_data['day'] = future_date.day
            current_data['quarter'] = future_date.quarter
            # Trend & cyclic
            days_since_start = (future_date - last_data['date'].min()).days
            current_data['days_since_start'] = days_since_start
            current_data['trend_squared'] = days_since_start ** 2
            current_data['trend_cubed'] = days_since_start ** 3
            current_data['weekofyear'] = future_date.isocalendar().week
            current_data['sin_day'] = np.sin(2 * np.pi * current_data['dayofweek']/7)
            current_data['cos_day'] = np.cos(2 * np.pi * current_data['dayofweek']/7)
            current_data['sin_month'] = np.sin(2 * np.pi * current_data['month']/12)
            current_data['cos_month'] = np.cos(2 * np.pi * current_data['month']/12)
            # Predict
            pred = model.predict(current_data[feature_cols])[0]
            future_preds.append(pred)
            # Update target lags/rolls
            for lag in [1, 2, 3, 7, 14]:
                if lag <= i + 1:
                    current_data[f'{target_col}_lag_{lag}'] = future_preds[-lag]
            for window in [3, 7, 14]:
                if len(future_preds) >= window:
                    current_data[f'{target_col}_rolling_mean_{window}'] = np.mean(future_preds[-window:])
                    current_data[f'{target_col}_rolling_median_{window}'] = np.median(future_preds[-window:])
                    current_data[f'{target_col}_rolling_std_{window}'] = np.std(future_preds[-window:])
                    current_data[f'{target_col}_rolling_min_{window}'] = np.min(future_preds[-window:])
                    current_data[f'{target_col}_rolling_max_{window}'] = np.max(future_preds[-window:])
            if len(future_preds) >= 2:
                current_data[f'{target_col}_diff_1'] = future_preds[-1] - future_preds[-2]
                current_data[f'{target_col}_pct_change_1'] = (future_preds[-1] / future_preds[-2]) - 1 if future_preds[-2] != 0 else 0
            if len(future_preds) >= 8:
                current_data[f'{target_col}_diff_7'] = future_preds[-1] - future_preds[-8]
                current_data[f'{target_col}_pct_change_7'] = (future_preds[-1] / future_preds[-8]) - 1 if future_preds[-8] != 0 else 0
            # --- Exogenous lags/rolls ---
            for var in exo_vars:
                exo_histories[var].append(exo_histories[var][-1])  # propagate last value
                for lag in exo_lags:
                    if lag <= len(exo_histories[var]):
                        current_data[f'{var}_lag_{lag}'] = exo_histories[var][-lag]
                for window in exo_rolls:
                    if len(exo_histories[var]) >= window:
                        current_data[f'{var}_rolling_mean_{window}'] = np.mean(exo_histories[var][-window:])
                        current_data[f'{var}_rolling_std_{window}'] = np.std(exo_histories[var][-window:])
            # --- End exogenous ---
        quantile_forecasts[q_name] = future_preds
    forecast = pd.DataFrame({'date': future_dates})
    for q_name, preds in quantile_forecasts.items():
        forecast[q_name] = preds
    print("Forecast ranges (min-max):")
    for q_name in ['q_0.025', 'q_0.5', 'q_0.975']:
        print(f"  {q_name}: {min(forecast[q_name]):.2f} - {max(forecast[q_name]):.2f}")
    return forecast

def train_xgboost(X_train, y_train, X_val, y_val):
    """Train XGBoost model for time series forecasting (point estimate)"""
    model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='rmse',
        early_stopping_rounds=15
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    return model

def evaluate_model(y_true, y_pred, model_name=''):
    """Evaluate model performance using multiple metrics"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"{model_name} Performance:")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R²: {r2:.4f}")
    
    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}

def time_series_cv(data, feature_cols, target_col, n_splits=5):
    """Perform time series cross-validation"""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores = []
    
    for i, (train_idx, val_idx) in enumerate(tscv.split(data)):
        train_data = data.iloc[train_idx]
        val_data = data.iloc[val_idx]
        
        X_train = train_data[feature_cols]
        y_train = train_data[target_col]
        X_val = val_data[feature_cols]
        y_val = val_data[target_col]
        
        model = train_xgboost(X_train, y_train, X_val, y_val)
        val_preds = model.predict(X_val)
        
        cv_score = evaluate_model(y_val, val_preds, f"CV Split {i+1}")
        cv_scores.append(cv_score)
    
    # Calculate average CV metrics
    avg_rmse = np.mean([score['rmse'] for score in cv_scores])
    avg_mae = np.mean([score['mae'] for score in cv_scores])
    avg_r2 = np.mean([score['r2'] for score in cv_scores])
    
    print("\nAverage Cross-Validation Metrics:")
    print(f"Average RMSE: {avg_rmse:.2f}")
    print(f"Average MAE: {avg_mae:.2f}")
    print(f"Average R²: {avg_r2:.4f}")
    
    return cv_scores

def forecast_future(model, last_data, feature_cols, target_col, days=7):
    """Generate forecasts for future days"""
    current_data = last_data.copy()
    future_dates = pd.date_range(
        start=current_data['date'].values[0] + pd.Timedelta(days=1),
        periods=days,
        freq='D'
    )
    
    future_preds = []
    
    for future_date in future_dates:
        # Update date features
        current_data['date'] = future_date
        current_data['dayofweek'] = future_date.dayofweek
        current_data['is_weekend'] = int(future_date.dayofweek in [5, 6])
        current_data['month'] = future_date.month
        current_data['year'] = future_date.year
        current_data['day'] = future_date.day
        current_data['quarter'] = future_date.quarter
        
        # Predict
        pred = model.predict(current_data[feature_cols])[0]
        future_preds.append(pred)
        
        # Update lag features for next prediction
        for lag in range(1, 4):
            if lag < len(future_preds):
                current_data[f'{target_col}_lag_{lag}'] = future_preds[-lag]
            else:
                if lag == 1:
                    current_data[f'{target_col}_lag_{lag}'] = pred
                else:
                    current_data[f'{target_col}_lag_{lag}'] = current_data[f'{target_col}_lag_{lag-1}']
        
        # Update weekly lag if we have enough predictions
        if len(future_preds) >= 7:
            current_data[f'{target_col}_lag_7'] = future_preds[-7]
        
        # Update rolling features
        current_data[f'{target_col}_rolling_mean_3'] = np.mean(future_preds[-3:]) if len(future_preds) >= 3 else pred
        current_data[f'{target_col}_rolling_std_3'] = np.std(future_preds[-3:]) if len(future_preds) >= 3 else 0
        current_data[f'{target_col}_rolling_mean_7'] = np.mean(future_preds[-7:]) if len(future_preds) >= 7 else pred
        current_data[f'{target_col}_rolling_std_7'] = np.std(future_preds[-7:]) if len(future_preds) >= 7 else 0
        
        # Update difference features
        current_data[f'{target_col}_diff_1'] = pred - future_preds[-1] if len(future_preds) > 0 else 0
        current_data[f'{target_col}_diff_7'] = pred - future_preds[-7] if len(future_preds) >= 7 else 0
    
    forecast_df = pd.DataFrame({
        'date': future_dates,
        target_col: future_preds
    })
    
    return forecast_df

def main():
    """Main function: median download speed forecast with 95% CI, trend component and improved features."""
    target_col = 'Download Speed Mbps'
    # Load and preprocess data
    data = load_and_preprocess_data('fixed_data.csv')
    
    # Analyze historical trend for deterministic component
    print("\nAnalyzing historical trend...")
    # Calculate average monthly values to see trend
    data['month_year'] = data['date'].dt.to_period('M')
    monthly_avg = data.groupby('month_year')[target_col].mean().reset_index()
    monthly_avg['month_num'] = range(len(monthly_avg))
    
    # Fit a linear trend to the monthly data
    from sklearn.linear_model import LinearRegression
    X_trend = monthly_avg[['month_num']]
    y_trend = monthly_avg[target_col]
    trend_model = LinearRegression().fit(X_trend, y_trend)
    monthly_trend = trend_model.predict(X_trend)
    
    # Calculate monthly growth rate
    trend_slope = trend_model.coef_[0]
    monthly_growth = trend_slope  # Mbps per month
    daily_growth = monthly_growth / 30.0  # Approximate daily growth
    
    print(f"Historical trend analysis: {monthly_growth:.4f} Mbps/month increase")
    print(f"Daily growth rate: {daily_growth:.4f} Mbps/day")
    
    # Feature engineering
    data_features = create_features(data, target_col)
    # Split data
    train_data, val_data, test_data = split_data(data_features)
    # Define feature columns
    exclude_cols = ['date', 'Provider Name', 'metric_type', 'Download Speed', 'Upload Speed', 'Download Speed Mbps', 'Upload Speed Mbps', 'Minimum Latency', 'Test Count', 'Sample Count', 'User Count', 'month_year']
    feature_cols = [col for col in data_features.columns if col not in exclude_cols and col != target_col]
    # Prepare data for modeling
    X_train = train_data[feature_cols]
    y_train = train_data[target_col]
    X_val = val_data[feature_cols]
    y_val = val_data[target_col]
    # Hyperparameter-tuned quantile regression
    print(f"\nTraining quantile models for {target_col} (with tuning)...")
    quantile_models = train_quantile_models(X_train, y_train, X_val, y_val)
    # Forecast next 7 days
    last_data = data_features.iloc[-1:].copy()
    quantile_forecast_df = quantile_forecast(quantile_models, last_data, feature_cols, target_col, days=7)
    
    # Apply deterministic trend component to forecasts
    base_forecast = quantile_forecast_df.copy()
    # Add trend component to each day's forecast
    for i, day in enumerate(range(1, 8)):  # 7 days forecast
        # Add the cumulative trend for each day
        quantile_forecast_df.loc[i, 'q_0.5'] += daily_growth * day
        quantile_forecast_df.loc[i, 'q_0.025'] += daily_growth * day * 0.8  # Slightly less trend for lower bound
        quantile_forecast_df.loc[i, 'q_0.975'] += daily_growth * day * 1.2  # Slightly more trend for upper bound
    
    # Output
    output_df = quantile_forecast_df.rename(columns={
        'q_0.5': 'median',
        'q_0.025': 'lower_95',
        'q_0.975': 'upper_95'
    })[['date', 'median', 'lower_95', 'upper_95']]
    
    # Fix any anomalies in confidence intervals
    # Ensure upper bound is consistently increasing
    for i in range(1, len(output_df)):
        if output_df.loc[i, 'upper_95'] < output_df.loc[i-1, 'upper_95']:
            output_df.loc[i, 'upper_95'] = output_df.loc[i-1, 'upper_95'] + daily_growth * 1.5
    
    # Ensure lower bound is consistently decreasing or flat
    for i in range(1, len(output_df)):
        if output_df.loc[i, 'lower_95'] > output_df.loc[i-1, 'lower_95']:
            output_df.loc[i, 'lower_95'] = output_df.loc[i-1, 'lower_95'] - daily_growth * 0.2
    
    print("\nMedian Download Speed Forecast with Trend Component (95% CI):")
    print(output_df)
    output_df.to_csv('median_download_speed_forecast_95ci.csv', index=False)
    print("Saved forecast with confidence intervals to 'median_download_speed_forecast_95ci.csv'")
    
    # Visualization
    plt.figure(figsize=(15, 7))
    plt.plot(data['date'], data[target_col], label='Historical Median Download Speed', color='blue')
    plt.plot(output_df['date'], output_df['median'], label='Median Forecast (with Trend)', color='red')
    plt.fill_between(output_df['date'], output_df['lower_95'], output_df['upper_95'], color='orange', alpha=0.2, label='95% CI')
    plt.title('Median Download Speed Forecast with Trend Component and 95% CI')
    plt.xlabel('Date')
    plt.ylabel('Speed (Mbps)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('median_download_speed_forecast_95ci.png')
    plt.show()

if __name__ == "__main__":
    main()

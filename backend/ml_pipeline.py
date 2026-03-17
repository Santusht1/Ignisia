import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')


class SalesPredictor:
    def __init__(self):
        self.models = {
            'Random Forest': RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'XGBoost': XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            ),
            'Prophet': None,
            'Auto ARIMA': None,
            'SARIMA': None
        }

        self.best_model_name = None
        self.best_model = None
        self.features = []
        self.metrics = {}
        self.feature_importance = {}
        self.scaler = StandardScaler()
        self.scale_features = False

    # -------------------- PREPROCESS --------------------
    def preprocess(self, df):
        df = df.copy()  # Avoid modifying original
        df.columns = [c.strip().lower() for c in df.columns]

        # Detect date column
        date_col = next((c for c in df.columns if 'date' in c), None)
        if not date_col:
            raise ValueError("CSV must contain a 'Date' column.")

        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)

        # Target selection
        if 'quantity' in df.columns and 'price' in df.columns:
            df['revenue'] = df['quantity'] * df['price']
            target_col = 'revenue'
        elif 'sales' in df.columns:
            target_col = 'sales'
        elif 'revenue' in df.columns:
            target_col = 'revenue'
        elif 'quantity' in df.columns:
            target_col = 'quantity'
        else:
            raise ValueError("No valid target column found")

        # Fill missing values
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(method='ffill')

        # Aggregate daily
        daily_sales = df.groupby(date_col)[target_col].sum().reset_index()

        # -------- Enhanced Feature Engineering --------
        daily_sales['day_of_week'] = daily_sales[date_col].dt.dayofweek
        daily_sales['month'] = daily_sales[date_col].dt.month
        daily_sales['day'] = daily_sales[date_col].dt.day
        daily_sales['is_weekend'] = (daily_sales['day_of_week'] >= 5).astype(int)
        
        # Quarter and year for seasonality
        daily_sales['quarter'] = daily_sales[date_col].dt.quarter
        daily_sales['year'] = daily_sales[date_col].dt.year
        
        # Week of year for annual patterns
        daily_sales['week_of_year'] = daily_sales[date_col].dt.isocalendar().week

        # Multiple lag features
        for lag in [1, 2, 3, 7, 14]:
            daily_sales[f'lag_{lag}'] = daily_sales[target_col].shift(lag)

        # Multiple rolling features with different windows
        for window in [3, 7, 14, 30]:
            daily_sales[f'rolling_mean_{window}'] = daily_sales[target_col].rolling(window, min_periods=1).mean()
            daily_sales[f'rolling_std_{window}'] = daily_sales[target_col].rolling(window, min_periods=1).std()
            daily_sales[f'rolling_min_{window}'] = daily_sales[target_col].rolling(window, min_periods=1).min()
            daily_sales[f'rolling_max_{window}'] = daily_sales[target_col].rolling(window, min_periods=1).max()

        # Exponential moving average
        daily_sales['ema_7'] = daily_sales[target_col].ewm(span=7, adjust=False).mean()
        daily_sales['ema_14'] = daily_sales[target_col].ewm(span=14, adjust=False).mean()
        
        # Trend: difference from moving average
        daily_sales['trend_7'] = daily_sales[target_col] - daily_sales['rolling_mean_7']
        daily_sales['trend_30'] = daily_sales[target_col] - daily_sales['rolling_mean_30']

        # Fill NaN values from feature creation
        daily_sales = daily_sales.fillna(method='bfill').fillna(method='ffill')

        self.features = [
            'day_of_week', 'month', 'day', 'is_weekend', 'quarter', 'week_of_year',
            'lag_1', 'lag_2', 'lag_3', 'lag_7', 'lag_14',
            'rolling_mean_3', 'rolling_std_3',
            'rolling_mean_7', 'rolling_std_7', 'rolling_min_7', 'rolling_max_7',
            'rolling_mean_14', 'rolling_std_14',
            'rolling_mean_30', 'rolling_std_30',
            'ema_7', 'ema_14',
            'trend_7', 'trend_30'
        ]

        return daily_sales, target_col

    # -------------------- TRAIN --------------------
    def train_and_evaluate(self, df, target_col):
        X = df[self.features]
        y = df[target_col]
        date_series = df[df.columns[0]]

        # Use 80/20 split
        split_idx = int(len(df) * 0.8)

        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Scale features for ML models
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )

        best_r2 = -float('inf')

        for name in self.models:

            try:
                # -------- Prophet --------
                if name == 'Prophet':
                    model = Prophet(
                        daily_seasonality=True,
                        weekly_seasonality=True,
                        yearly_seasonality=True,
                        changepoint_prior_scale=0.05,
                        seasonality_prior_scale=10.0
                    )
                    self.models['Prophet'] = model

                    train_df = pd.DataFrame({
                        'ds': date_series.iloc[:split_idx].values,
                        'y': y_train.values
                    })

                    future_df = pd.DataFrame({
                        'ds': date_series.iloc[split_idx:].values
                    })

                    model.fit(train_df)
                    forecast = model.predict(future_df)
                    preds = forecast['yhat'].values

                # -------- Auto ARIMA --------
                elif name == 'Auto ARIMA':
                    model = auto_arima(
                        y_train,
                        seasonal=True,
                        m=7,
                        max_p=3,
                        max_q=3,
                        max_P=2,
                        max_Q=2,
                        max_d=2,
                        max_D=1,
                        stepwise=True,
                        suppress_warnings=True,
                        error_action='ignore',
                        trace=False
                    )
                    self.models['Auto ARIMA'] = model
                    preds = model.predict(n_periods=len(y_test))

                # -------- SARIMA --------
                elif name == 'SARIMA':
                    model = SARIMAX(
                        y_train,
                        order=(2, 1, 2),  # Improved order
                        seasonal_order=(1, 1, 1, 7),
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    ).fit(disp=False, maxiter=100)

                    self.models['SARIMA'] = model
                    preds = model.forecast(steps=len(y_test))

                # -------- ML Models --------
                else:
                    model = self.models[name]
                    model.fit(X_train_scaled, y_train)
                    preds = model.predict(X_test_scaled)

                # Ensure predictions are positive
                preds = np.maximum(preds, 0)

                # -------- Metrics --------
                rmse = np.sqrt(mean_squared_error(y_test, preds))
                mae = mean_absolute_error(y_test, preds)
                r2 = r2_score(y_test, preds)

                self.metrics[name] = {
                    'RMSE': float(rmse),
                    'MAE': float(mae),
                    'R2': float(r2)
                }

                if r2 > best_r2:
                    best_r2 = r2
                    self.best_model_name = name
                    self.best_model = model

            except Exception as e:
                print(f"Model {name} failed: {str(e)}")
                continue

        # -------- Feature Importance --------
        if self.best_model_name in ['Auto ARIMA', 'SARIMA', 'Prophet']:
            self.feature_importance = {'Time Series Model': 1.0}

        elif hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            self.feature_importance = dict(
                zip(self.features, [float(x) for x in importances])
            )

        elif hasattr(self.best_model, 'coef_'):
            coefs = np.abs(self.best_model.coef_)
            self.feature_importance = dict(
                zip(self.features, [float(x) for x in coefs])
            )

    # -------------------- PREDICT --------------------
    def predict_future(self, df, target_col, days=30):
        future_dates = [
            df.iloc[-1][df.columns[0]] + pd.Timedelta(days=i)
            for i in range(1, days + 1)
        ]

        predictions = []

        # -------- Time Series Models --------
        if self.best_model_name == 'Prophet':
            future_df = pd.DataFrame({'ds': future_dates})
            forecast = self.best_model.predict(future_df)
            pred_vals = forecast['yhat'].values

        elif self.best_model_name == 'Auto ARIMA':
            pred_vals = self.best_model.predict(n_periods=days)

        elif self.best_model_name == 'SARIMA':
            pred_vals = self.best_model.forecast(steps=days)

        # -------- ML Models --------
        else:
            last_known = df.copy()
            pred_vals = []

            for date in future_dates:
                # Calculate all features
                recent_data = last_known[target_col].tail(30)
                
                row = {
                    'day_of_week': date.dayofweek,
                    'month': date.month,
                    'day': date.day,
                    'is_weekend': int(date.dayofweek >= 5),
                    'quarter': (date.month - 1) // 3 + 1,
                    'week_of_year': date.isocalendar()[1],
                }
                
                # Lag features
                for lag in [1, 2, 3, 7, 14]:
                    if len(last_known) >= lag:
                        row[f'lag_{lag}'] = last_known.iloc[-lag][target_col]
                    else:
                        row[f'lag_{lag}'] = last_known.iloc[-1][target_col]
                
                # Rolling features
                for window in [3, 7, 14, 30]:
                    window_data = recent_data.tail(window)
                    row[f'rolling_mean_{window}'] = window_data.mean()
                    row[f'rolling_std_{window}'] = window_data.std() if len(window_data) > 1 else 0
                    row[f'rolling_min_{window}'] = window_data.min()
                    row[f'rolling_max_{window}'] = window_data.max()
                
                # EMA
                row['ema_7'] = recent_data.tail(7).ewm(span=7, adjust=False).mean().iloc[-1]
                row['ema_14'] = recent_data.tail(14).ewm(span=14, adjust=False).mean().iloc[-1]
                
                # Trend
                row['trend_7'] = last_known.iloc[-1][target_col] - row['rolling_mean_7']
                row['trend_30'] = last_known.iloc[-1][target_col] - row['rolling_mean_30']

                X_pred = pd.DataFrame([row])[self.features]
                X_pred_scaled = self.scaler.transform(X_pred)
                pred_val = self.best_model.predict(X_pred_scaled)[0]

                pred_vals.append(pred_val)

                # Add prediction to history for next iteration
                new_row = {df.columns[0]: date, target_col: pred_val}
                last_known = pd.concat([last_known, pd.DataFrame([new_row])], ignore_index=True)

        # -------- Format Output --------
        for date, val in zip(future_dates, pred_vals):
            predictions.append({
                'Date': date.strftime('%Y-%m-%d'),
                'Predicted_Sales': float(max(0, val))
            })

        return predictions

    # -------------------- PIPELINE --------------------
    def run_pipeline(self, df, forecast_days=30):

        if len(df) < 14:
            raise ValueError("Need at least 14 days of data")

        daily_sales, target_col = self.preprocess(df)

        self.train_and_evaluate(daily_sales, target_col)

        predictions = self.predict_future(
            daily_sales, target_col, forecast_days
        )

        # Last 30 days history
        historical = daily_sales[[daily_sales.columns[0], target_col]].tail(30)
        historical.columns = ['Date', 'Sales']
        historical['Date'] = historical['Date'].dt.strftime('%Y-%m-%d')

        return {
            'best_model': self.best_model_name,
            'metrics': self.metrics[self.best_model_name],
            'feature_importance': self.feature_importance,
            'historical': historical.to_dict(orient='records'),
            'predictions': predictions
        }

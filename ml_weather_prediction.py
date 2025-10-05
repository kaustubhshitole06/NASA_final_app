import numpy as np
import pandas as pd
from typing import Dict, List, Any, Union, Tuple
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('ml_weather_prediction')

class WeatherMLPredictor:
    """
    Machine Learning based weather prediction for NASA Weather Probability Dashboard.
    
    This class provides machine learning models to predict weather parameters based on
    historical NASA POWER API data, improving forecast accuracy beyond simple statistical
    probability calculations.
    """
    
    def __init__(self, model_dir: str = "ml_models"):
        """
        Initialize the ML predictor with model storage location.
        
        Args:
            model_dir: Directory to save/load trained models
        """
        # Set model directory - always use absolute path
        if model_dir == "ml_models" or not model_dir:
            # Use absolute path for model directory
            self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml_models")
        else:
            self.model_dir = model_dir
            
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
        # Ensure model directory exists with proper permissions
        try:
            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir, exist_ok=True)
                logger.info(f"Created model directory: {self.model_dir}")
            
            # Test write permissions
            test_file = os.path.join(self.model_dir, "test_write.txt")
            with open(test_file, 'w') as f:
                f.write("Test write permission")
            if os.path.exists(test_file):
                os.remove(test_file)
                logger.info(f"Model directory is writable: {self.model_dir}")
        except Exception as e:
            logger.error(f"Error with model directory: {str(e)}")
            # Continue anyway as the error will be caught later if needed
        self.feature_importance = {}
        
        # Create model directory if it doesn't exist
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
    
    def _prepare_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract time-based features from date for the model.
        
        Args:
            df: DataFrame with 'date' column in format YYYYMMDD
            
        Returns:
            DataFrame with added time features
        """
        df = df.copy()
        
        # Convert date to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
        
        # Extract time features
        df['day_of_year'] = df['date'].dt.dayofyear
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['year'] = df['date'].dt.year
        
        # Create cyclical features for month and day of year to capture seasonality
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        return df
    
    def _prepare_features(self, df: pd.DataFrame, parameter: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and target for training or prediction.
        
        Args:
            df: DataFrame with weather data
            parameter: Weather parameter to predict
            
        Returns:
            Tuple of (X features, y target)
        """
        # Add time features
        df = self._prepare_time_features(df)
        
        # Select features
        features = ['day_of_year', 'month', 'day', 'month_sin', 'month_cos', 'day_sin', 'day_cos']
        
        # Add historical features if available (e.g., lag features)
        if 'value_lag1' in df.columns:
            features.extend(['value_lag1', 'value_lag2', 'value_lag3'])
        
        # Add latitude and longitude if available
        if 'latitude' in df.columns:
            features.extend(['latitude', 'longitude'])
        
        X = df[features].values
        y = df['value'].values if parameter in df.columns else df[parameter].values
        
        return X, y
    
    def _add_lag_features(self, df: pd.DataFrame, lag_days: int = 3) -> pd.DataFrame:
        """
        Add lagged value features for time series prediction.
        
        Args:
            df: DataFrame with weather data
            lag_days: Number of previous days to use as features
            
        Returns:
            DataFrame with added lag features
        """
        df = df.copy()
        
        # Sort by date
        df = df.sort_values('date')
        
        # Add lag features
        for lag in range(1, lag_days + 1):
            df[f'value_lag{lag}'] = df['value'].shift(lag)
        
        # Drop rows with NaN lag values
        df = df.dropna()
        
        return df
    
    def train_model(self, 
                    weather_data: List[Dict[str, Any]], 
                    parameter: str,
                    model_type: str = 'random_forest') -> Dict[str, Any]:
        """
        Train a machine learning model for a specific weather parameter.
        
        Args:
            weather_data: List of weather data dictionaries
            parameter: Parameter to predict (T2M, PRECTOTCORR, etc.)
            model_type: Type of model to train ('random_forest', 'gradient_boosting', 'linear')
            
        Returns:
            Dictionary with training results and metrics
        """
        try:
            # Convert to DataFrame for easier manipulation
            df = pd.DataFrame(weather_data)
            
            # Filter for the target parameter
            param_df = df[df['parameter'] == parameter]
            
            if len(param_df) < 30:
                return {
                    'success': False, 
                    'error': f'Insufficient data for parameter {parameter}. Need at least 30 data points.'
                }
            
            # Add lag features for time series prediction
            param_df = self._add_lag_features(param_df)
            
            # Prepare features
            X, y = self._prepare_features(param_df, parameter)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Select model
            if model_type == 'random_forest':
                model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            elif model_type == 'gradient_boosting':
                model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            elif model_type == 'linear':
                model = LinearRegression()
            else:
                return {'success': False, 'error': f'Unknown model type: {model_type}'}
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Save feature importance if available
            feature_importance = {}
            if hasattr(model, 'feature_importances_'):
                features = ['day_of_year', 'month', 'day', 'month_sin', 'month_cos', 'day_sin', 'day_cos']
                if 'value_lag1' in param_df.columns:
                    features.extend(['value_lag1', 'value_lag2', 'value_lag3'])
                if 'latitude' in param_df.columns:
                    features.extend(['latitude', 'longitude'])
                
                importance = model.feature_importances_
                feature_importance = {feature: float(imp) for feature, imp in zip(features, importance)}
                self.feature_importance[parameter] = feature_importance
            
            # Save model and scaler
            try:
                model_path = os.path.join(self.model_dir, f"{parameter}_{model_type}.joblib")
                scaler_path = os.path.join(self.model_dir, f"{parameter}_{model_type}_scaler.joblib")
                
                logger.info(f"Saving model to {model_path}")
                joblib.dump(model, model_path)
                logger.info(f"Saving scaler to {scaler_path}")
                joblib.dump(scaler, scaler_path)
                logger.info(f"Successfully saved model and scaler for {parameter}")
            except Exception as save_error:
                logger.error(f"Error saving model/scaler: {str(save_error)}")
                raise Exception(f"Failed to save model: {str(save_error)}")
            
            # Store in memory for immediate use
            self.models[parameter] = model
            self.scalers[parameter] = scaler
            
            return {
                'success': True,
                'parameter': parameter,
                'model_type': model_type,
                'metrics': {
                    'mse': float(mse),
                    'rmse': float(np.sqrt(mse)),
                    'r2': float(r2)
                },
                'feature_importance': feature_importance,
                'model_path': model_path,
                'data_points': len(param_df)
            }
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            logger.error(f"Error training model for {parameter}: {str(e)}\n{error_trace}")
            return {'success': False, 'error': str(e), 'traceback': error_trace}
    
    def predict(self, 
                target_date: str,  # Format: MMDD
                latitude: float, 
                longitude: float, 
                parameter: str,
                model_type: str = 'random_forest',
                historical_data: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make predictions for a specific date and location.
        
        Args:
            target_date: Target date in MMDD format
            latitude: Location latitude
            longitude: Location longitude
            parameter: Weather parameter to predict
            model_type: Type of model to use
            historical_data: Optional historical data to enhance prediction
            
        Returns:
            Prediction results
        """
        try:
            # Load model if not already in memory
            if parameter not in self.models:
                model_path = os.path.join(self.model_dir, f"{parameter}_{model_type}.joblib")
                scaler_path = os.path.join(self.model_dir, f"{parameter}_{model_type}_scaler.joblib")
                
                if not os.path.exists(model_path):
                    return {
                        'success': False, 
                        'error': f'Model for {parameter} not found. Train the model first.'
                    }
                
                self.models[parameter] = joblib.load(model_path)
                self.scalers[parameter] = joblib.load(scaler_path)
            
            # Create prediction date for current year
            current_year = datetime.now().year
            month = int(target_date[:2])
            day = int(target_date[2:])
            prediction_date = datetime(current_year, month, day)
            
            # Prepare prediction dataframe
            pred_df = pd.DataFrame({
                'date': [prediction_date],
                'latitude': [latitude],
                'longitude': [longitude]
            })
            
            # Add historical data if available
            if historical_data:
                hist_df = pd.DataFrame(historical_data)
                if 'parameter' in hist_df.columns:
                    # Filter for specific parameter
                    hist_df = hist_df[hist_df['parameter'] == parameter]
                
                # Use the most recent data for lag features
                hist_df = hist_df.sort_values('date', ascending=False)
                
                # Take the 3 most recent values for lag features
                recent_values = hist_df['value'].values[:3]
                
                # Ensure we have 3 lag values (pad with means if needed)
                while len(recent_values) < 3:
                    if len(recent_values) > 0:
                        recent_values = np.append(recent_values, np.mean(recent_values))
                    else:
                        recent_values = np.append(recent_values, 0)  # No data, use 0
                
                # Add lag features to prediction dataframe
                pred_df['value_lag1'] = recent_values[0]
                pred_df['value_lag2'] = recent_values[1]
                pred_df['value_lag3'] = recent_values[2]
            
            # Add time features
            pred_df = self._prepare_time_features(pred_df)
            
            # Prepare features for prediction
            features = ['day_of_year', 'month', 'day', 'month_sin', 'month_cos', 'day_sin', 'day_cos']
            
            # Add lag features if available
            if 'value_lag1' in pred_df.columns:
                features.extend(['value_lag1', 'value_lag2', 'value_lag3'])
            
            # Add location features
            features.extend(['latitude', 'longitude'])
            
            X_pred = pred_df[features].values
            
            # Scale features
            X_pred_scaled = self.scalers[parameter].transform(X_pred)
            
            # Make prediction
            prediction = self.models[parameter].predict(X_pred_scaled)[0]
            
            # Get uncertainty estimate (standard deviation of predictions for RF)
            uncertainty = 0
            if isinstance(self.models[parameter], RandomForestRegressor):
                # Get predictions from all trees in the forest
                predictions = np.array([tree.predict(X_pred_scaled) for tree in self.models[parameter].estimators_])
                uncertainty = float(np.std(predictions))
            
            return {
                'success': True,
                'parameter': parameter,
                'date': target_date,
                'prediction': float(prediction),
                'uncertainty': uncertainty,
                'model_type': model_type
            }
            
        except Exception as e:
            logger.error(f"Error making prediction for {parameter}: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def analyze_extreme_events(self, 
                               predictions: Dict[str, float], 
                               thresholds: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """
        Analyze predictions for extreme weather events based on thresholds.
        
        Args:
            predictions: Dictionary of parameter predictions
            thresholds: Dictionary of thresholds for each parameter
            
        Returns:
            Analysis of extreme events and probability estimates
        """
        results = {}
        
        for param, prediction in predictions.items():
            if param in thresholds:
                param_thresholds = thresholds[param]
                param_results = {}
                
                for condition, threshold in param_thresholds.items():
                    # Determine if condition is met
                    condition_lower = condition.lower()
                    
                    if any(term in condition_lower for term in ["hot", "warm", "wet", "windy", "humid"]):
                        # For conditions where higher values are "worse"
                        is_exceeded = prediction >= threshold
                    elif any(term in condition_lower for term in ["cold", "cool", "dry"]):
                        # For conditions where lower values are "worse"
                        is_exceeded = prediction <= threshold
                    else:
                        # Default to "greater than or equal to"
                        is_exceeded = prediction >= threshold
                    
                    # Add to results
                    param_results[condition] = {
                        'threshold': threshold,
                        'predicted_value': prediction,
                        'threshold_exceeded': is_exceeded
                    }
                
                results[param] = param_results
        
        return results
    
    def calculate_comfort_prediction(self, 
                                     temperature: float,
                                     humidity: float = None,
                                     wind_speed: float = None) -> Dict[str, Any]:
        """
        Calculate comfort prediction based on predicted weather parameters.
        
        Args:
            temperature: Predicted temperature
            humidity: Predicted humidity (if available)
            wind_speed: Predicted wind speed (if available)
            
        Returns:
            Comfort prediction and analysis
        """
        # Temperature comfort (optimal around 20-25°C)
        if 20 <= temperature <= 25:
            temp_score = 10
        elif 15 <= temperature <= 30:
            temp_score = 7
        elif 10 <= temperature <= 35:
            temp_score = 4
        else:
            temp_score = 1
        
        score = temp_score * 0.6  # Temperature weight: 60%
        
        # Humidity comfort (if available)
        if humidity is not None:
            if 40 <= humidity <= 60:
                hum_score = 10
            elif 30 <= humidity <= 70:
                hum_score = 7
            elif 20 <= humidity <= 80:
                hum_score = 4
            else:
                hum_score = 1
            
            score += hum_score * 0.3  # Humidity weight: 30%
        else:
            score += 7 * 0.3  # Default neutral humidity score
        
        # Wind speed comfort (if available)
        if wind_speed is not None:
            if wind_speed <= 5:
                wind_score = 10
            elif wind_speed <= 10:
                wind_score = 7
            elif wind_speed <= 15:
                wind_score = 4
            else:
                wind_score = 1
            
            score += wind_score * 0.1  # Wind weight: 10%
        else:
            score += 7 * 0.1  # Default neutral wind score
        
        # Define comfort categories
        comfort_level = "unknown"
        if score >= 8:
            comfort_level = "very_comfortable"
        elif score >= 6:
            comfort_level = "comfortable"
        elif score >= 4:
            comfort_level = "uncomfortable"
        else:
            comfort_level = "very_uncomfortable"
        
        return {
            'comfort_score': float(score),
            'comfort_level': comfort_level,
            'comfort_components': {
                'temperature_score': temp_score,
                'humidity_score': hum_score if humidity is not None else None,
                'wind_score': wind_score if wind_speed is not None else None
            }
        }
    
    def forecast_multiple_days(self,
                               start_date: datetime,
                               days: int,
                               latitude: float,
                               longitude: float,
                               parameters: List[str],
                               model_type: str = 'random_forest') -> Dict[str, Any]:
        """
        Generate forecasts for multiple days ahead.
        
        Args:
            start_date: Starting date for forecast
            days: Number of days to forecast
            latitude: Location latitude
            longitude: Location longitude
            parameters: List of parameters to forecast
            model_type: Type of model to use
            
        Returns:
            Multi-day forecast results
        """
        forecasts = {}
        
        for day in range(days):
            forecast_date = start_date + timedelta(days=day)
            date_str = forecast_date.strftime("%m%d")
            
            day_forecast = {}
            for param in parameters:
                prediction = self.predict(
                    target_date=date_str,
                    latitude=latitude,
                    longitude=longitude,
                    parameter=param,
                    model_type=model_type
                )
                
                if prediction['success']:
                    day_forecast[param] = prediction
            
            forecasts[forecast_date.strftime("%Y-%m-%d")] = day_forecast
        
        return {
            'success': True,
            'location': {'latitude': latitude, 'longitude': longitude},
            'forecast_period': {
                'start': start_date.strftime("%Y-%m-%d"),
                'end': (start_date + timedelta(days=days-1)).strftime("%Y-%m-%d"),
                'days': days
            },
            'forecasts': forecasts
        }

# Helper functions for integration with NASA Weather Probability API
def prepare_historical_data(nasa_api_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert NASA API data to format suitable for ML training.
    
    Args:
        nasa_api_data: Raw data from NASA POWER API
        
    Returns:
        Processed list of data points
    """
    processed_data = []
    
    if "properties" in nasa_api_data and "parameter" in nasa_api_data["properties"]:
        parameters = nasa_api_data["properties"]["parameter"]
        
        for param_name, param_data in parameters.items():
            for date_str, value in param_data.items():
                if value != -999:  # Skip missing values
                    processed_data.append({
                        "parameter": param_name,
                        "date": pd.to_datetime(date_str),
                        "value": float(value),
                        "year": int(date_str[:4]),
                        "month": int(date_str[4:6]),
                        "day": int(date_str[6:8])
                    })
    
    return processed_data

def integrate_ml_predictions(statistical_results: Dict[str, Any], ml_predictions: Dict[str, Any]) -> Dict[str, Any]:
    """
    Integrate ML predictions with statistical results.
    
    Args:
        statistical_results: Results from statistical analysis
        ml_predictions: Results from ML prediction
        
    Returns:
        Combined results
    """
    integrated_results = statistical_results.copy()
    
    # Add ML predictions to each parameter's results
    for param, prediction in ml_predictions.items():
        if param in integrated_results and prediction['success']:
            if 'ml_prediction' not in integrated_results[param]:
                integrated_results[param]['ml_prediction'] = {}
            
            integrated_results[param]['ml_prediction'] = {
                'value': prediction['prediction'],
                'uncertainty': prediction['uncertainty'],
                'model_type': prediction['model_type']
            }
    
    # Add ML-based summary
    if 'summary' not in integrated_results:
        integrated_results['summary'] = {}
    
    integrated_results['summary']['ml_enhanced'] = True
    integrated_results['summary']['prediction_method'] = 'hybrid'
    
    return integrated_results
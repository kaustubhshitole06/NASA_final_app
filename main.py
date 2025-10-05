from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, RedirectResponse
import httpx
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import json
import io
import os
from pydantic import BaseModel
import asyncio
from nasa_gibs import NASAGIBSClient
from ml_weather_prediction import WeatherMLPredictor, prepare_historical_data, integrate_ml_predictions
import math
from fastapi.security.api_key import APIKey, APIKeyHeader

app = FastAPI(
    title="NASA Weather Probability API",
    description="API for calculating weather probabilities using NASA POWER data",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# NASA POWER API base URL
NASA_POWER_BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

# OpenWeather API configuration
# Note: Replace 'your_api_key_here' with the actual API key provided
OPENWEATHER_API_KEY = "6cc49058c1ff253b14620b285986dbc6"  
OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5"
OPENWEATHER_ONECALL_URL = f"{OPENWEATHER_BASE_URL}/onecall"
OPENWEATHER_FORECAST_URL = f"{OPENWEATHER_BASE_URL}/forecast"

# Flood risk levels
FLOOD_RISK_LEVELS = {
    "LOW": {"threshold": 10, "color": "#28a745", "description": "Low risk of flooding"},
    "MODERATE": {"threshold": 30, "color": "#ffc107", "description": "Moderate risk of flooding"},
    "HIGH": {"threshold": 50, "color": "#fd7e14", "description": "High risk of flooding"},
    "SEVERE": {"threshold": 80, "color": "#dc3545", "description": "Severe risk of flooding"}
}

# Available weather parameters
WEATHER_PARAMETERS = {
    "T2M": "Temperature at 2 Meters (°C)",
    "T2M_MAX": "Maximum Temperature at 2 Meters (°C)", 
    "T2M_MIN": "Minimum Temperature at 2 Meters (°C)",
    "PRECTOTCORR": "Precipitation Corrected (mm/day)",
    "WS2M": "Wind Speed at 2 Meters (m/s)",
    "WS10M": "Wind Speed at 10 Meters (m/s)", 
    "RH2M": "Relative Humidity at 2 Meters (%)",
    "PS": "Surface Pressure (kPa)",
    "QV2M": "Specific Humidity at 2 Meters (g/kg)",
    "GWETPROF": "Profile Soil Moisture (%)",  # Added for flood detection
    "GWETROOT": "Root Zone Soil Moisture (%)",  # Added for flood detection
    "GWETTOP": "Top Soil Moisture (%)",  # Added for flood detection
    "RUNOFF": "Runoff (kg/m^2/day)",  # Added for flood detection
    "WET_DAYS": "Number of Wet Days (days)"  # Added for flood detection
}

class LocationRequest(BaseModel):
    latitude: float
    longitude: float
    start_date: str  # Format: YYYYMMDD
    end_date: str    # Format: YYYYMMDD
    parameters: List[str]
    community: str = "AG"  # Default to Agricultural community

class WeatherProbabilityRequest(BaseModel):
    latitude: float
    longitude: float
    target_date: str  # Format: MMDD (month-day)
    years_back: int = 10  # How many years of historical data to analyze
    parameters: List[str]
    thresholds: Dict[str, Dict[str, float]]  # Parameter -> {"hot": 35, "cold": 0, etc.}
    use_ml: bool = False  # Whether to use ML predictions
    community: str = "AG"  # Data community: AG (Agriculture), RE (Renewable Energy), or SB (Sustainable Buildings)

class MLPredictionRequest(BaseModel):
    latitude: float
    longitude: float
    target_date: str  # Format: MMDD (month-day) 
    parameters: List[str]
    model_type: str = "random_forest"  # Model type to use for prediction
    thresholds: Optional[Dict[str, Dict[str, float]]] = None  # Optional thresholds for analysis
    community: str = "AG"  # Data community: AG (Agriculture), RE (Renewable Energy), or SB (Sustainable Buildings)

# OpenWeather API key header
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# New models for flood detection
class FloodRiskRequest(BaseModel):
    latitude: float
    longitude: float
    forecast_days: int = 5  # How many days ahead to predict flood risk
    include_historical: bool = True  # Whether to include historical data in risk assessment
    elevation: Optional[float] = None  # Optional elevation data if available

class FloodRiskResponse(BaseModel):
    location: Dict[str, float]
    current_risk: Dict[str, Any]
    forecast: List[Dict[str, Any]]
    historical_patterns: Optional[Dict[str, Any]] = None
    contributing_factors: Dict[str, float]
    recommendations: List[str]

# Utility function to get the OpenWeather API key
async def get_api_key(api_key: Optional[str] = Depends(api_key_header)) -> str:
    if api_key:
        return api_key
    return OPENWEATHER_API_KEY

# Function to fetch elevation data from the Open Elevation API
async def get_elevation_data(latitude: float, longitude: float) -> Optional[float]:
    """Fetch elevation data from the Open Elevation API (free and no API key required)"""
    try:
        elevation_api_url = "https://api.open-elevation.com/api/v1/lookup"
        
        payload = {
            "locations": [
                {
                    "latitude": latitude,
                    "longitude": longitude
                }
            ]
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                elevation_api_url, 
                json=payload,
                timeout=10.0  # 10 second timeout
            )
            
        if response.status_code != 200:
            print(f"Failed to get elevation data: {response.status_code}")
            return None
            
        data = response.json()
        if "results" in data and len(data["results"]) > 0:
            return data["results"][0].get("elevation")
            
        return None
    except Exception as e:
        print(f"Error fetching elevation data: {str(e)}")
        return None

# Helper function to determine if a location is in a flood-prone area
async def is_flood_prone(latitude: float, longitude: float, elevation: Optional[float] = None) -> Dict[str, Any]:
    # If elevation is not provided, try to fetch it
    if elevation is None:
        elevation = await get_elevation_data(latitude, longitude)
        
    # Default elevation factor if we still don't have elevation data
    elevation_factor = 0.5
    
    if elevation is not None:
        # Low-lying areas (below 10m) are more flood-prone
        if elevation < 10:
            elevation_factor = 1.0  # Very high risk
        elif elevation < 30:
            elevation_factor = 0.8  # High risk
        elif elevation < 100:
            elevation_factor = 0.6  # Moderate risk
        else:
            elevation_factor = 0.3  # Low risk
    
    # Proximity to water bodies would be ideal to consider here
    # This would require a geographical database of water bodies
    
    # Areas closer to equator tend to have more intense rainfall (simplified)
    latitude_factor = 1.0 - (abs(latitude) / 90) * 0.5
    
    # Calculate an overall flood proneness score (0-1)
    flood_proneness = (elevation_factor * 0.7) + (latitude_factor * 0.3)
    
    return {
        "flood_prone_score": flood_proneness,
        "elevation_factor": elevation_factor,
        "latitude_factor": latitude_factor,
        "elevation": elevation,
        "is_high_risk": flood_proneness > 0.7
    }

@app.get("/")
async def root():
    return {"message": "NASA Weather Probability API", "version": "1.0.0"}

@app.get("/parameters")
async def get_available_parameters():
    """Get list of available weather parameters"""
    return {"parameters": WEATHER_PARAMETERS}

@app.post("/weather/raw")
async def get_raw_weather_data(request: LocationRequest):
    """Fetch raw weather data from NASA POWER API"""
    
    try:
        # Build the NASA POWER API URL
        params = {
            "parameters": ",".join(request.parameters),
            "community": request.community,
            "latitude": request.latitude,
            "longitude": request.longitude,
            "start": request.start_date,
            "end": request.end_date,
            "format": "JSON"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.get(NASA_POWER_BASE_URL, params=params)
            
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail=f"NASA API error: {response.text}")
            
        data = response.json()
        
        return {
            "status": "success",
            "location": {
                "latitude": request.latitude,
                "longitude": request.longitude
            },
            "date_range": {
                "start": request.start_date,
                "end": request.end_date
            },
            "parameters": request.parameters,
            "data": data
        }
        
    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"Request failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.post("/weather/probability")
async def calculate_weather_probability(request: WeatherProbabilityRequest):
    """Calculate probability of weather conditions based on historical data"""
    
    try:
        # Calculate date range for historical data
        current_year = datetime.now().year
        start_year = current_year - request.years_back
        
        # Get historical data for the same day across multiple years
        all_data = []
        
        for year in range(start_year, current_year):
            start_date = f"{year}{request.target_date}"
            end_date = start_date  # Single day
            
            params = {
                "parameters": ",".join(request.parameters),
                "community": request.community,
                "latitude": request.latitude,
                "longitude": request.longitude,
                "start": start_date,
                "end": end_date,
                "format": "JSON"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(NASA_POWER_BASE_URL, params=params)
                
            if response.status_code == 200:
                year_data = response.json()
                if "properties" in year_data and "parameter" in year_data["properties"]:
                    for param in request.parameters:
                        if param in year_data["properties"]["parameter"]:
                            param_data = year_data["properties"]["parameter"][param]
                            if param_data:  # Check if data exists
                                for date, value in param_data.items():
                                    if value != -999:  # NASA uses -999 for missing data
                                        all_data.append({
                                            "year": year,
                                            "date": date,
                                            "parameter": param,
                                            "value": value
                                        })
        
        if not all_data:
            raise HTTPException(status_code=404, detail="No historical data found for the specified location and date")
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(all_data)
        
        # Calculate statistics and probabilities
        results = {}
        
        for param in request.parameters:
            param_df = df[df["parameter"] == param]
            
            if len(param_df) > 0:
                values = param_df["value"].values
                
                stats = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "median": float(np.median(values)),
                    "count": len(values)
                }
                
                # Calculate probabilities based on thresholds
                probabilities = {}
                if param in request.thresholds:
                    thresholds = request.thresholds[param]
                    
                    for condition, threshold_value in thresholds.items():
                        if condition.lower() in ["hot", "very_hot", "extreme_hot"]:
                            probability = np.mean(values >= threshold_value) * 100
                        elif condition.lower() in ["cold", "very_cold", "extreme_cold"]:
                            probability = np.mean(values <= threshold_value) * 100
                        elif condition.lower() in ["wet", "very_wet"]:
                            probability = np.mean(values >= threshold_value) * 100
                        elif condition.lower() in ["windy", "very_windy"]:
                            probability = np.mean(values >= threshold_value) * 100
                        else:
                            probability = np.mean(values >= threshold_value) * 100
                        
                        probabilities[condition] = round(probability, 2)
                
                results[param] = {
                    "statistics": stats,
                    "probabilities": probabilities,
                    "unit": WEATHER_PARAMETERS.get(param, "Unknown"),
                    "historical_values": values.tolist()
                }
        
        # If ML predictions are requested, add them to the response
        if request.use_ml:
            # Initialize ML predictor
            ml_predictor = WeatherMLPredictor()
            ml_predictions = {}
            
            # Process historical data for training
            historical_data = prepare_historical_data({"properties": {"parameter": {p: {} for p in request.parameters}}})
            
            for param in request.parameters:
                # Get ML prediction for this parameter
                prediction = ml_predictor.predict(
                    target_date=request.target_date,
                    latitude=request.latitude,
                    longitude=request.longitude,
                    parameter=param,
                    model_type="random_forest",
                    historical_data=historical_data
                )
                
                if prediction["success"]:
                    ml_predictions[param] = prediction
            
            # If we have ML predictions, integrate them with statistical results
            if ml_predictions:
                results = integrate_ml_predictions(results, ml_predictions)
                
                # Add ML-enhanced flag
                return {
                    "status": "success",
                    "location": {
                        "latitude": request.latitude,
                        "longitude": request.longitude
                    },
                    "target_date": request.target_date,
                    "years_analyzed": request.years_back,
                    "data_points": len(all_data),
                    "results": results,
                    "ml_enhanced": True,
                    "model_type": "random_forest"
                }
            
        return {
            "status": "success",
            "location": {
                "latitude": request.latitude,
                "longitude": request.longitude
            },
            "target_date": request.target_date,
            "years_analyzed": request.years_back,
            "data_points": len(all_data),
            "results": results,
            "ml_enhanced": False
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating probabilities: {str(e)}")

@app.get("/weather/export/csv")
async def export_weather_data_csv(
    latitude: float = Query(..., description="Latitude"),
    longitude: float = Query(..., description="Longitude"), 
    start_date: str = Query(..., description="Start date (YYYYMMDD)"),
    end_date: str = Query(..., description="End date (YYYYMMDD)"),
    parameters: str = Query(..., description="Comma-separated parameter list"),
    community: str = Query("AG", description="NASA POWER data community (AG, RE, SB)")
):
    """Export weather data as CSV"""
    
    try:
        param_list = parameters.split(",")
        
        params = {
            "parameters": parameters,
            "community": community,
            "latitude": latitude,
            "longitude": longitude,
            "start": start_date,
            "end": end_date,
            "format": "JSON"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.get(NASA_POWER_BASE_URL, params=params)
            
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to fetch data from NASA API")
            
        data = response.json()
        
        # Convert to DataFrame
        records = []
        if "properties" in data and "parameter" in data["properties"]:
            for param in param_list:
                if param in data["properties"]["parameter"]:
                    param_data = data["properties"]["parameter"][param]
                    for date, value in param_data.items():
                        records.append({
                            "date": date,
                            "parameter": param,
                            "value": value,
                            "unit": WEATHER_PARAMETERS.get(param, "Unknown")
                        })
        
        df = pd.DataFrame(records)
        
        # Create CSV
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode()),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=weather_data_{latitude}_{longitude}_{start_date}_{end_date}.csv"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

# Initialize NASA GIBS client
gibs_client = NASAGIBSClient()

@app.get("/satellite/layers")
async def get_satellite_layers():
    """Get available satellite imagery layers"""
    layers = gibs_client.get_available_layers()
    categories = gibs_client.get_layer_categories()
    
    return {
        "status": "success",
        "layers": layers,
        "categories": categories
    }

@app.get("/satellite/tile/{z}/{y}/{x}")
async def get_satellite_tile(
    z: int, 
    y: int, 
    x: int, 
    layer: str = Query("true_color", description="Layer ID or shorthand name"),
    date: str = Query(None, description="Date in YYYY-MM-DD format"),
    resolution: str = Query("medium", description="Image resolution")
):
    """Proxy for NASA GIBS satellite imagery tiles"""
    try:
        if date:
            date_obj = datetime.strptime(date, "%Y-%m-%d")
        else:
            date_obj = None
        
        tile_url = gibs_client.get_tile_url(
            date=date_obj,
            layer=layer,
            resolution=resolution,
            z=z, y=y, x=x
        )
        
        return RedirectResponse(url=tile_url)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching satellite tile: {str(e)}")

@app.get("/satellite/wms")
async def get_satellite_wms(
    layer: str = Query("true_color", description="Layer ID or shorthand name"),
    date: str = Query(None, description="Date in YYYY-MM-DD format"),
    bbox: str = Query("-180,-90,180,90", description="Bounding box (minx,miny,maxx,maxy)"),
    width: int = Query(1024, description="Image width in pixels"),
    height: int = Query(512, description="Image height in pixels")
):
    """Proxy for NASA GIBS WMS imagery"""
    try:
        if date:
            date_obj = datetime.strptime(date, "%Y-%m-%d")
        else:
            date_obj = None
            
        wms_url = gibs_client.get_wms_url(
            date=date_obj,
            layer=layer,
            bbox=bbox,
            width=width,
            height=height
        )
        
        return RedirectResponse(url=wms_url)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching WMS image: {str(e)}")

@app.get("/satellite/timerange/{layer}")
async def get_satellite_timerange(layer: str):
    """Get available time range for a satellite layer"""
    try:
        time_range = gibs_client.get_time_range(layer)
        return time_range
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting time range: {str(e)}")

@app.get("/satellite/config")
async def get_leaflet_config():
    """Get Leaflet.js configuration for NASA GIBS"""
    try:
        config = gibs_client.build_leaflet_provider_config()
        return config
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting configuration: {str(e)}")

@app.post("/weather/ml-prediction")
async def get_ml_prediction(request: MLPredictionRequest):
    """Get machine learning based weather prediction"""
    
    try:
        # Get absolute path for model directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(base_dir, "ml_models")
        
        # Initialize ML predictor with explicit model directory
        ml_predictor = WeatherMLPredictor(model_dir=model_dir)
        
        # Get data for the target date from NASA POWER API for previous years
        # This helps improve ML prediction by providing recent historical data
        current_year = datetime.now().year
        all_data = []
        
        # Collect historical data from recent years to help the ML model
        for year in range(current_year - 5, current_year):
            start_date = f"{year}{request.target_date}"
            end_date = start_date  # Single day
            
            params = {
                "parameters": ",".join(request.parameters),
                "community": request.community,
                "latitude": request.latitude,
                "longitude": request.longitude,
                "start": start_date,
                "end": end_date,
                "format": "JSON"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(NASA_POWER_BASE_URL, params=params)
                
            if response.status_code == 200:
                year_data = response.json()
                historical_data = prepare_historical_data(year_data)
                all_data.extend(historical_data)
        
        # Make predictions for each requested parameter
        predictions = {}
        for param in request.parameters:
            prediction = ml_predictor.predict(
                target_date=request.target_date,
                latitude=request.latitude,
                longitude=request.longitude,
                parameter=param,
                model_type=request.model_type,
                historical_data=all_data
            )
            
            if prediction["success"]:
                predictions[param] = {
                    "prediction": prediction["prediction"],
                    "uncertainty": prediction["uncertainty"],
                    "parameter": param,
                    "unit": WEATHER_PARAMETERS.get(param, "Unknown")
                }
        
        # If no successful predictions, return error
        if not predictions:
            raise HTTPException(
                status_code=404, 
                detail="Could not generate predictions. Try training the models first or check parameters."
            )
        
        # If thresholds are provided, analyze the predictions
        extreme_events = None
        if request.thresholds:
            param_predictions = {param: pred["prediction"] for param, pred in predictions.items()}
            extreme_events = ml_predictor.analyze_extreme_events(
                predictions=param_predictions,
                thresholds=request.thresholds
            )
        
        # Calculate comfort prediction if we have the necessary parameters
        comfort_prediction = None
        if "T2M" in predictions:
            temp = predictions["T2M"]["prediction"]
            humidity = predictions["RH2M"]["prediction"] if "RH2M" in predictions else None
            wind_speed = predictions["WS2M"]["prediction"] if "WS2M" in predictions else None
            
            comfort_prediction = ml_predictor.calculate_comfort_prediction(
                temperature=temp,
                humidity=humidity,
                wind_speed=wind_speed
            )
        
        # Build response
        response_data = {
            "status": "success",
            "location": {
                "latitude": request.latitude,
                "longitude": request.longitude
            },
            "target_date": request.target_date,
            "model_type": request.model_type,
            "predictions": predictions
        }
        
        if extreme_events:
            response_data["extreme_events"] = extreme_events
            
        if comfort_prediction:
            response_data["comfort"] = comfort_prediction
            
        return response_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making ML prediction: {str(e)}")

@app.get("/weather/ml-models")
async def check_ml_models(parameters: str = Query(...), model_type: str = Query("random_forest")):
    """Check if ML models exist for specified parameters"""
    try:
        # Get absolute path for model directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(base_dir, "ml_models")
        
        # Initialize ML predictor with explicit model directory
        ml_predictor = WeatherMLPredictor(model_dir=model_dir)
        
        # Get list of parameters
        param_list = parameters.split(',')
        
        # Check if models exist
        all_models_exist = True
        model_status = {}
        
        for param in param_list:
            model_path = os.path.join(ml_predictor.model_dir, f"{param}_{model_type}.joblib")
            scaler_path = os.path.join(ml_predictor.model_dir, f"{param}_{model_type}_scaler.joblib")
            
            # Check if both model and scaler exist
            model_exists = os.path.exists(model_path)
            scaler_exists = os.path.exists(scaler_path)
            
            model_status[param] = {
                "model_exists": model_exists,
                "scaler_exists": scaler_exists,
                "model_path": model_path,
                "ready": model_exists and scaler_exists
            }
            
            if not (model_exists and scaler_exists):
                all_models_exist = False
        
        return {
            "status": "success",
            "all_models_exist": all_models_exist,
            "model_status": model_status
        }
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Error checking ML models: {str(e)}\n{error_trace}"
        )

@app.post("/weather/ml-train")
async def train_ml_models(request: MLPredictionRequest):
    """Train machine learning models for weather prediction"""
    
    try:
        # Get absolute path for model directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(base_dir, "ml_models")
        
        # Initialize ML predictor with explicit model directory
        ml_predictor = WeatherMLPredictor(model_dir=model_dir)
        
        # Get historical data for training
        current_year = datetime.now().year
        start_year = current_year - 10  # Use 10 years for training
        all_data = []
        
        # Get month and day from target_date
        month = request.target_date[:2]
        day = request.target_date[2:]
        
        # Collect data for the requested parameters across multiple years
        # For ML training, we get a wider date range around the target date
        for year in range(start_year, current_year):
            # Create dates for a window around the target date
            center_date = datetime(year, int(month), int(day))
            window_start = center_date - timedelta(days=15)
            window_end = center_date + timedelta(days=15)
            
            start_date = window_start.strftime("%Y%m%d")
            end_date = window_end.strftime("%Y%m%d")
            
            params = {
                "parameters": ",".join(request.parameters),
                "community": request.community,
                "latitude": request.latitude,
                "longitude": request.longitude,
                "start": start_date,
                "end": end_date,
                "format": "JSON"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(NASA_POWER_BASE_URL, params=params)
                
            if response.status_code == 200:
                year_data = response.json()
                historical_data = prepare_historical_data(year_data)
                all_data.extend(historical_data)
        
        if not all_data:
            raise HTTPException(
                status_code=404, 
                detail="No historical data available for training ML models."
            )
        
        # Train models for each parameter
        training_results = {}
        for param in request.parameters:
            result = ml_predictor.train_model(
                weather_data=all_data,
                parameter=param,
                model_type=request.model_type
            )
            
            training_results[param] = result
        
        # Return training results
        return {
            "status": "success",
            "location": {
                "latitude": request.latitude,
                "longitude": request.longitude
            },
            "model_type": request.model_type,
            "parameters": request.parameters,
            "training_results": training_results,
            "data_points_used": len(all_data)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training ML models: {str(e)}")

@app.get("/weather/ml-forecast")
async def get_ml_forecast(
    latitude: float = Query(..., description="Latitude"),
    longitude: float = Query(..., description="Longitude"),
    start_date: str = Query(..., description="Start date (MMDD)"),
    days: int = Query(7, description="Number of days to forecast"),
    parameters: str = Query(..., description="Comma-separated parameter list"),
    model_type: str = Query("random_forest", description="ML model type to use")
):
    """Get ML-based forecast for multiple days"""
    
    try:
        param_list = parameters.split(",")
        
        # Initialize ML predictor
        ml_predictor = WeatherMLPredictor()
        
        # Parse start date
        month = int(start_date[:2])
        day = int(start_date[2:])
        current_year = datetime.now().year
        start_date_obj = datetime(current_year, month, day)
        
        # Get forecast
        forecast = ml_predictor.forecast_multiple_days(
            start_date=start_date_obj,
            days=days,
            latitude=latitude,
            longitude=longitude,
            parameters=param_list,
            model_type=model_type
        )
        
        return forecast
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating ML forecast: {str(e)}")

# Flood risk calculation function
async def calculate_flood_risk(
    latitude: float, 
    longitude: float,
    forecast_days: int = 5,
    include_historical: bool = True,
    elevation: Optional[float] = None,
    api_key: str = OPENWEATHER_API_KEY
) -> Dict[str, Any]:
    """
    Calculate flood risk based on multiple data sources:
    1. Real-time precipitation data from OpenWeather
    2. Forecast data for the next few days
    3. Historical precipitation patterns from NASA POWER
    4. Elevation and terrain information
    """
    flood_risk_results = {
        "current_risk": {
            "level": "UNKNOWN",
            "score": 0,
            "description": "Unable to determine risk level"
        },
        "forecast": [],
        "contributing_factors": {},
        "recommendations": []
    }
    
    try:
        # 1. Get real-time and forecast weather data from OpenWeather
        async with httpx.AsyncClient() as client:
            # Current weather with OneCall API
            params = {
                "lat": latitude,
                "lon": longitude,
                "appid": api_key,
                "units": "metric",
                "exclude": "minutely,hourly,alerts"  # Only get daily forecasts
            }
            
            response = await client.get(OPENWEATHER_ONECALL_URL, params=params)
            
            if response.status_code != 200:
                # If OpenWeather API fails, we can still use NASA POWER data
                weather_data = None
            else:
                weather_data = response.json()
        
        # 2. Get historical data from NASA POWER for the same location
        historical_data = []
        if include_historical:
            current_year = datetime.now().year
            for year in range(current_year - 5, current_year):
                # Get data for this month and previous month to analyze patterns
                current_month = datetime.now().month
                # Calculate start and end dates for historical data
                start_date = f"{year}{max(1, current_month-1):02d}01"  # Previous month start
                end_date = f"{year}{current_month:02d}28"  # Current month
                
                params = {
                    "parameters": "PRECTOTCORR,GWETPROF,GWETROOT,GWETTOP,RUNOFF,WET_DAYS",
                    "community": "AG",
                    "latitude": latitude,
                    "longitude": longitude,
                    "start": start_date,
                    "end": end_date,
                    "format": "JSON"
                }
                
                async with httpx.AsyncClient() as client:
                    response = await client.get(NASA_POWER_BASE_URL, params=params)
                    
                if response.status_code == 200:
                    nasa_data = response.json()
                    if "properties" in nasa_data and "parameter" in nasa_data["properties"]:
                        for param_name, param_values in nasa_data["properties"]["parameter"].items():
                            for date, value in param_values.items():
                                if value != -999:  # NASA uses -999 for missing data
                                    historical_data.append({
                                        "date": date,
                                        "parameter": param_name,
                                        "value": value
                                    })
        
        # 3. Check if the location is in a flood-prone area
        flood_prone_info = await is_flood_prone(latitude, longitude, elevation)
        
        # 4. Calculate current flood risk based on all available data
        # Initialize base risk based on location characteristics
        current_risk_score = flood_prone_info["flood_prone_score"] * 20  # Scale to 0-20
        risk_factors = {
            "terrain": flood_prone_info["flood_prone_score"] * 20
        }
        
        # Add recent precipitation data if available
        recent_precip = 0
        soil_moisture = 0.5  # Default if not available
        
        if weather_data:
            # Get precipitation from current and last day
            current = weather_data.get("current", {})
            daily = weather_data.get("daily", [])
            
            # Current day rain
            recent_precip += current.get("rain", {}).get("1h", 0) * 24  # Extrapolate hourly to daily
            
            # Forecast data
            if daily:
                # Add first day forecast (today)
                today_forecast = daily[0]
                recent_precip += today_forecast.get("rain", 0)
                
                # Process forecast for upcoming days
                for i, day in enumerate(daily):
                    if i >= forecast_days:
                        break
                        
                    forecast_date = datetime.fromtimestamp(day.get("dt", 0)).strftime("%Y-%m-%d")
                    precip = day.get("rain", 0)
                    
                    # Calculate daily risk based on precipitation
                    daily_risk_score = 0
                    if precip < 5:
                        daily_risk_level = "LOW"
                        daily_risk_score = precip * 5  # 0-25
                    elif precip < 15:
                        daily_risk_level = "MODERATE"
                        daily_risk_score = 25 + (precip - 5) * 2.5  # 25-50
                    elif precip < 30:
                        daily_risk_level = "HIGH"
                        daily_risk_score = 50 + (precip - 15) * 2  # 50-80
                    else:
                        daily_risk_level = "SEVERE"
                        daily_risk_score = min(80 + (precip - 30), 100)  # 80-100
                    
                    # Add daily forecast to results
                    flood_risk_results["forecast"].append({
                        "date": forecast_date,
                        "precipitation": precip,
                        "risk_level": daily_risk_level,
                        "risk_score": daily_risk_score,
                        "description": FLOOD_RISK_LEVELS[daily_risk_level]["description"]
                    })
        
        # Add historical data analysis
        if historical_data:
            # Calculate average precipitation and wet days
            precip_values = [item["value"] for item in historical_data if item["parameter"] == "PRECTOTCORR"]
            wet_days = [item["value"] for item in historical_data if item["parameter"] == "WET_DAYS"]
            soil_moisture_values = [
                item["value"] for item in historical_data 
                if item["parameter"] in ["GWETPROF", "GWETROOT", "GWETTOP"]
            ]
            
            avg_precip = sum(precip_values) / len(precip_values) if precip_values else 0
            avg_wet_days = sum(wet_days) / len(wet_days) if wet_days else 0
            avg_soil_moisture = sum(soil_moisture_values) / len(soil_moisture_values) if soil_moisture_values else 0.5
            
            # Use historical data to adjust risk
            historical_risk = 0
            if avg_precip > 10:  # High average precipitation area
                historical_risk += min(avg_precip / 2, 20)
            if avg_wet_days > 15:  # Many wet days historically
                historical_risk += 10
            
            risk_factors["historical_patterns"] = historical_risk
            current_risk_score += historical_risk
            soil_moisture = avg_soil_moisture
        
        # Adjust for recent precipitation
        precip_risk = 0
        if recent_precip > 0:
            if recent_precip < 10:
                precip_risk = recent_precip * 2  # 0-20
            elif recent_precip < 25:
                precip_risk = 20 + (recent_precip - 10)  # 20-35
            else:
                precip_risk = 35 + (recent_precip - 25) * 1.5  # 35-60 max
        
        risk_factors["recent_precipitation"] = precip_risk
        current_risk_score += precip_risk
        
        # Adjust for soil moisture (wetter soil = higher flood risk)
        soil_risk = soil_moisture * 20  # 0-20 scale
        risk_factors["soil_moisture"] = soil_risk
        current_risk_score += soil_risk
        
        # Determine overall current risk level
        if current_risk_score < FLOOD_RISK_LEVELS["LOW"]["threshold"]:
            risk_level = "LOW"
        elif current_risk_score < FLOOD_RISK_LEVELS["MODERATE"]["threshold"]:
            risk_level = "MODERATE"
        elif current_risk_score < FLOOD_RISK_LEVELS["HIGH"]["threshold"]:
            risk_level = "HIGH"
        else:
            risk_level = "SEVERE"
        
        # Update current risk in results
        flood_risk_results["current_risk"] = {
            "level": risk_level,
            "score": current_risk_score,
            "description": FLOOD_RISK_LEVELS[risk_level]["description"],
            "color": FLOOD_RISK_LEVELS[risk_level]["color"]
        }
        
        # Add contributing factors
        flood_risk_results["contributing_factors"] = risk_factors
        
        # Add recommendations based on risk level
        if risk_level == "LOW":
            flood_risk_results["recommendations"] = [
                "No special precautions needed",
                "Monitor weather forecasts during rainy season"
            ]
        elif risk_level == "MODERATE":
            flood_risk_results["recommendations"] = [
                "Keep emergency supplies ready",
                "Monitor water levels in nearby water bodies",
                "Stay informed about weather updates"
            ]
        elif risk_level == "HIGH":
            flood_risk_results["recommendations"] = [
                "Move valuables to higher ground",
                "Prepare for possible evacuation",
                "Avoid areas prone to flash flooding",
                "Keep emergency contacts handy"
            ]
        else:  # SEVERE
            flood_risk_results["recommendations"] = [
                "Evacuate if instructed by authorities",
                "Move to higher ground immediately",
                "Avoid flood waters - just 6 inches of water can knock you down",
                "Do not drive through flooded areas",
                "Stay informed through emergency channels"
            ]
        
        return flood_risk_results
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error calculating flood risk: {str(e)}\n{error_trace}")
        
        # Return default risk assessment
        return flood_risk_results

@app.post("/flood/risk")
async def get_flood_risk(
    request: FloodRiskRequest,
    api_key: str = Depends(get_api_key)
) -> Dict[str, Any]:
    """Calculate flood risk for a location based on multiple data sources"""
    try:
        flood_risk = await calculate_flood_risk(
            latitude=request.latitude,
            longitude=request.longitude,
            forecast_days=request.forecast_days,
            include_historical=request.include_historical,
            elevation=request.elevation,
            api_key=api_key
        )
        
        return {
            "status": "success",
            "location": {
                "latitude": request.latitude,
                "longitude": request.longitude
            },
            "timestamp": datetime.now().isoformat(),
            "flood_risk": flood_risk
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating flood risk: {str(e)}")

@app.get("/flood/thresholds")
async def get_flood_thresholds():
    """Get flood risk level thresholds for reference"""
    return {
        "status": "success",
        "thresholds": FLOOD_RISK_LEVELS
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}
    
@app.get("/weather/ml-debug")
async def ml_debug(param: str = Query("T2M"), lat: float = 40.7, lon: float = -74.0):
    """Debug endpoint for ML testing"""
    try:
        # Get absolute path for model directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(base_dir, "ml_models")
        
        # Initialize ML predictor with explicit model directory
        ml_predictor = WeatherMLPredictor(model_dir=model_dir)
        
        # Check model directory
        model_dir_exists = os.path.exists(ml_predictor.model_dir)
        model_dir_writable = os.access(ml_predictor.model_dir, os.W_OK) if model_dir_exists else False
        
        # Try to create a test file
        test_file_path = os.path.join(ml_predictor.model_dir, "test_write.txt")
        test_write_success = False
        test_write_error = None
        
        try:
            with open(test_file_path, "w") as f:
                f.write("Test write permission")
            test_write_success = True
            if os.path.exists(test_file_path):
                os.remove(test_file_path)
        except Exception as write_error:
            test_write_error = str(write_error)
        
        # Get current year for testing
        current_year = datetime.now().year
        start_year = current_year - 5  # Use 5 years for testing
        
        # Get data for testing
        all_data = []
        for year in range(start_year, current_year):
            start_date = f"{year}0101"
            end_date = f"{year}0131"  # Just use January for test
            
            params = {
                "parameters": param,
                "community": "AG",  # Using default AG for debug endpoint
                "latitude": lat,
                "longitude": lon,
                "start": start_date,
                "end": end_date,
                "format": "JSON"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(NASA_POWER_BASE_URL, params=params)
                
            if response.status_code == 200:
                year_data = response.json()
                historical_data = prepare_historical_data(year_data)
                all_data.extend(historical_data)
        
        data_count = len(all_data)
        
        return {
            "status": "success",
            "model_dir": ml_predictor.model_dir,
            "model_dir_exists": model_dir_exists,
            "model_dir_writable": model_dir_writable,
            "test_write_success": test_write_success,
            "data_count": data_count,
            "data_sample": all_data[:5] if all_data else []
        }
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Debug error: {str(e)}\n{error_trace}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8082)
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, RedirectResponse
import httpx
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Union
import json
import io
from pydantic import BaseModel
import asyncio
from nasa_gibs import NASAGIBSClient
from nasa_client import NASAPowerClient

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
    "QV2M": "Specific Humidity at 2 Meters (g/kg)"
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

class HourlyDataRequest(BaseModel):
    latitude: float
    longitude: float
    start_date: str  # Format: YYYYMMDD
    end_date: str    # Format: YYYYMMDD
    parameters: List[str]
    community: str = "SB"  # Default to Science/Buildings community

class DailyDataRequest(BaseModel):
    latitude: float
    longitude: float
    start_date: str  # Format: YYYYMMDD
    end_date: str    # Format: YYYYMMDD
    parameters: List[str]
    community: str = "SB"  # Default to Science/Buildings community

class MonthlyDataRequest(BaseModel):
    latitude: float
    longitude: float
    start_month: str  # Format: YYYYMM
    end_month: str    # Format: YYYYMM
    parameters: List[str]
    community: str = "SB"  # Default to Science/Buildings community

class ClimatologyDataRequest(BaseModel):
    latitude: float
    longitude: float
    month: str  # Format: MM (01-12) or "all"
    parameters: List[str]
    community: str = "SB"  # Default to Science/Buildings community

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
                "community": "AG",
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
        
        return {
            "status": "success",
            "location": {
                "latitude": request.latitude,
                "longitude": request.longitude
            },
            "target_date": request.target_date,
            "years_analyzed": request.years_back,
            "data_points": len(all_data),
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating probabilities: {str(e)}")

@app.get("/weather/export/csv")
async def export_weather_data_csv(
    latitude: float = Query(..., description="Latitude"),
    longitude: float = Query(..., description="Longitude"), 
    start_date: str = Query(..., description="Start date (YYYYMMDD)"),
    end_date: str = Query(..., description="End date (YYYYMMDD)"),
    parameters: str = Query(..., description="Comma-separated parameter list")
):
    """Export weather data as CSV"""
    
    try:
        param_list = parameters.split(",")
        
        params = {
            "parameters": parameters,
            "community": "AG",
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

@app.post("/weather/hourly")
async def get_hourly_weather_data(request: HourlyDataRequest):
    """Fetch hourly weather data from NASA POWER API"""
    
    try:
        async with NASAPowerClient() as client:
            data = await client.get_hourly_data(
                latitude=request.latitude,
                longitude=request.longitude,
                start_date=request.start_date,
                end_date=request.end_date,
                parameters=request.parameters,
                community=request.community
            )
            
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
        raise HTTPException(status_code=500, detail=f"Error fetching hourly data: {str(e)}")

@app.post("/weather/daily")
async def get_daily_weather_data(request: DailyDataRequest):
    """Fetch daily weather data from NASA POWER API"""
    
    try:
        async with NASAPowerClient() as client:
            data = await client.get_daily_data(
                latitude=request.latitude,
                longitude=request.longitude,
                start_date=request.start_date,
                end_date=request.end_date,
                parameters=request.parameters,
                community=request.community
            )
            
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
        raise HTTPException(status_code=500, detail=f"Error fetching daily data: {str(e)}")

@app.post("/weather/monthly")
async def get_monthly_weather_data(request: MonthlyDataRequest):
    """Fetch monthly weather data from NASA POWER API"""
    
    try:
        async with NASAPowerClient() as client:
            data = await client.get_monthly_data(
                latitude=request.latitude,
                longitude=request.longitude,
                start_year_month=request.start_month,
                end_year_month=request.end_month,
                parameters=request.parameters,
                community=request.community
            )
            
        return {
            "status": "success",
            "location": {
                "latitude": request.latitude,
                "longitude": request.longitude
            },
            "date_range": {
                "start": request.start_month,
                "end": request.end_month
            },
            "parameters": request.parameters,
            "data": data
        }
        
    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"Request failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching monthly data: {str(e)}")

@app.post("/weather/climatology")
async def get_climatology_weather_data(request: ClimatologyDataRequest):
    """Fetch climatology weather data from NASA POWER API"""
    
    try:
        async with NASAPowerClient() as client:
            data = await client.get_climatology_data(
                latitude=request.latitude,
                longitude=request.longitude,
                month=request.month,
                parameters=request.parameters,
                community=request.community
            )
            
        return {
            "status": "success",
            "location": {
                "latitude": request.latitude,
                "longitude": request.longitude
            },
            "month": request.month,
            "parameters": request.parameters,
            "data": data
        }
        
    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"Request failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching climatology data: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8082)
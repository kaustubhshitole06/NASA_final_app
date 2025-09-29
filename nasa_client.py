import httpx
import asyncio
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union

class NASAPowerClient:
    """Client for interacting with NASA POWER API"""
    
    # Base URLs for different temporal endpoints
    BASE_URLS = {
        "hourly": "https://power.larc.nasa.gov/api/temporal/hourly/point",
        "daily": "https://power.larc.nasa.gov/api/temporal/daily/point",
        "monthly": "https://power.larc.nasa.gov/api/temporal/monthly/point", 
        "climatology": "https://power.larc.nasa.gov/api/temporal/climatology/point"
    }
    
    def __init__(self):
        self.session = None
    
    async def __aenter__(self):
        self.session = httpx.AsyncClient()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.aclose()
    
    async def get_data(
        self,
        temporal_api: str,  # 'hourly', 'daily', 'monthly', or 'climatology'
        latitude: float, 
        longitude: float,
        parameters: List[str],
        start: Optional[str] = None,  # Format depends on temporal_api
        end: Optional[str] = None,    # Format depends on temporal_api
        community: str = "SB",        # SB = Science/Buildings community
        format_type: str = "JSON"
    ) -> Dict[str, Any]:
        """
        Fetch data from NASA POWER API for different temporal resolutions
        
        Args:
            temporal_api: Type of temporal API ('hourly', 'daily', 'monthly', 'climatology')
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            parameters: List of weather parameters to fetch
            start: Start date in format appropriate for the temporal_api:
                   - hourly, daily: YYYYMMDD
                   - monthly: YYYYMM
                   - climatology: MM (month number 01-12)
            end: End date in format appropriate for the temporal_api
            community: Data community (SB, AG, RE, SSE)
            format_type: Response format (JSON, CSV, ASCII, NETCDF)
        
        Returns:
            Dict containing API response data
        """
        
        # Validate temporal_api
        if temporal_api not in self.BASE_URLS:
            raise ValueError(f"Invalid temporal_api: {temporal_api}. Valid options are: {list(self.BASE_URLS.keys())}")
        
        # Build base parameters
        params = {
            "parameters": ",".join(parameters),
            "community": community,
            "latitude": latitude,
            "longitude": longitude,
            "format": format_type
        }
        
        # Add start/end parameters based on temporal_api
        if temporal_api != "climatology":
            if not start or not end:
                raise ValueError(f"Both start and end are required for {temporal_api} API")
                
            # For monthly API, the parameter names are different
            if temporal_api == "monthly":
                params["start"] = start
                params["end"] = end
            else:
                params["start"] = start
                params["end"] = end
        else:
            # Climatology only needs month
            if start:
                params["month"] = start
        
        if not self.session:
            raise RuntimeError("Client not initialized. Use 'async with' context manager.")
        
        # Make API request
        response = await self.session.get(self.BASE_URLS[temporal_api], params=params)
        response.raise_for_status()
        
        return response.json()
    
    # Convenience methods for each temporal resolution
    
    async def get_hourly_data(
        self,
        latitude: float,
        longitude: float,
        start_date: str,  # YYYYMMDD
        end_date: str,    # YYYYMMDD
        parameters: List[str],
        community: str = "SB"
    ) -> Dict[str, Any]:
        """Get hourly weather data"""
        return await self.get_data(
            temporal_api="hourly",
            latitude=latitude,
            longitude=longitude,
            parameters=parameters,
            start=start_date,
            end=end_date,
            community=community
        )
    
    async def get_daily_data(
        self,
        latitude: float,
        longitude: float,
        start_date: str,  # YYYYMMDD
        end_date: str,    # YYYYMMDD
        parameters: List[str],
        community: str = "SB"
    ) -> Dict[str, Any]:
        """Get daily weather data"""
        return await self.get_data(
            temporal_api="daily",
            latitude=latitude,
            longitude=longitude,
            parameters=parameters,
            start=start_date,
            end=end_date,
            community=community
        )
    
    async def get_monthly_data(
        self,
        latitude: float,
        longitude: float,
        start_year_month: str,  # YYYYMM
        end_year_month: str,    # YYYYMM
        parameters: List[str],
        community: str = "SB"
    ) -> Dict[str, Any]:
        """Get monthly weather data"""
        return await self.get_data(
            temporal_api="monthly",
            latitude=latitude,
            longitude=longitude,
            parameters=parameters,
            start=start_year_month,
            end=end_year_month,
            community=community
        )
    
    async def get_climatology_data(
        self,
        latitude: float,
        longitude: float,
        month: str,  # MM (01-12, or "all" for all months)
        parameters: List[str],
        community: str = "SB"
    ) -> Dict[str, Any]:
        """Get climatology data for specified month(s)"""
        return await self.get_data(
            temporal_api="climatology",
            latitude=latitude,
            longitude=longitude,
            parameters=parameters,
            start=month,
            end=None,
            community=community
        )
    
    async def get_historical_range(
        self,
        latitude: float,
        longitude: float, 
        target_month_day: str,  # Format: MMDD
        parameters: list,
        years_back: int = 10
    ):
        """
        Get historical data for the same date across multiple years
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            target_month_day: Target date in MMDD format (e.g., "0315" for March 15)
            years_back: Number of years to go back
            parameters: List of weather parameters
            
        Returns:
            List of data points across years
        """
        
        current_year = datetime.now().year
        start_year = current_year - years_back
        
        all_data = []
        
        for year in range(start_year, current_year):
            date_str = f"{year}{target_month_day}"
            
            try:
                data = await self.get_weather_data(
                    latitude=latitude,
                    longitude=longitude,
                    start_date=date_str,
                    end_date=date_str,
                    parameters=parameters
                )
                
                if "properties" in data and "parameter" in data["properties"]:
                    for param in parameters:
                        if param in data["properties"]["parameter"]:
                            param_data = data["properties"]["parameter"][param]
                            for date, value in param_data.items():
                                if value != -999:  # Skip missing data
                                    all_data.append({
                                        "year": year,
                                        "date": date,
                                        "parameter": param,
                                        "value": value
                                    })
                                    
            except Exception as e:
                print(f"Warning: Failed to fetch data for year {year}: {e}")
                continue
        
        return all_data

# Weather threshold definitions
WEATHER_THRESHOLDS = {
    "T2M": {
        "very_hot": 35.0,       # Above 35°C
        "hot": 30.0,            # Above 30°C  
        "cold": 10.0,           # Below 10°C
        "very_cold": 0.0        # Below 0°C
    },
    "T2M_MAX": {
        "very_hot": 40.0,       # Above 40°C max
        "hot": 35.0,            # Above 35°C max
        "extreme_hot": 45.0     # Above 45°C max
    },
    "T2M_MIN": {
        "cold": 5.0,            # Below 5°C min
        "very_cold": -5.0,      # Below -5°C min  
        "extreme_cold": -10.0   # Below -10°C min
    },
    "PRECTOTCORR": {
        "wet": 10.0,            # Above 10mm/day
        "very_wet": 25.0,       # Above 25mm/day
        "extreme_wet": 50.0     # Above 50mm/day
    },
    "WS2M": {
        "windy": 10.0,          # Above 10 m/s
        "very_windy": 15.0,     # Above 15 m/s
        "extreme_windy": 20.0   # Above 20 m/s
    },
    "WS10M": {
        "windy": 12.0,          # Above 12 m/s at 10m
        "very_windy": 18.0,     # Above 18 m/s at 10m
        "extreme_windy": 25.0   # Above 25 m/s at 10m
    },
    "RH2M": {
        "very_humid": 80.0,     # Above 80% humidity
        "humid": 70.0,          # Above 70% humidity
        "dry": 30.0,            # Below 30% humidity
        "very_dry": 20.0        # Below 20% humidity
    }
}

def get_default_thresholds(parameter: str) -> dict:
    """Get default weather thresholds for a parameter"""
    return WEATHER_THRESHOLDS.get(parameter, {})

# Example usage
async def example_usage():
    """Example of how to use the NASA POWER client"""
    
    async with NASAPowerClient() as client:
        # Get current weather data
        data = await client.get_weather_data(
            latitude=12.97,
            longitude=77.59,
            start_date="20240101",
            end_date="20241231", 
            parameters=["T2M", "PRECTOTCORR", "WS2M"]
        )
        
        print("Current year data:", json.dumps(data, indent=2))
        
        # Get historical data for March 15th
        historical = await client.get_historical_range(
            latitude=12.97,
            longitude=77.59,
            target_month_day="0315",
            years_back=5,
            parameters=["T2M", "PRECTOTCORR"]
        )
        
        print(f"Found {len(historical)} historical data points")

if __name__ == "__main__":
    asyncio.run(example_usage())
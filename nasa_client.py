import httpx
import asyncio
import json
from datetime import datetime, timedelta

class NASAPowerClient:
    """Client for interacting with NASA POWER API"""
    
    BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
    
    def __init__(self):
        self.session = None
    
    async def __aenter__(self):
        self.session = httpx.AsyncClient()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.aclose()
    
    async def get_weather_data(
        self, 
        latitude: float, 
        longitude: float,
        start_date: str, 
        end_date: str,
        parameters: list,
        community: str = "AG"
    ):
        """
        Fetch weather data from NASA POWER API
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate  
            start_date: Start date in YYYYMMDD format
            end_date: End date in YYYYMMDD format
            parameters: List of weather parameters to fetch
            community: Data community (AG, RE, SB)
        
        Returns:
            Dict containing API response data
        """
        
        params = {
            "parameters": ",".join(parameters),
            "community": community,
            "latitude": latitude,
            "longitude": longitude,
            "start": start_date,
            "end": end_date,
            "format": "JSON"
        }
        
        if not self.session:
            raise RuntimeError("Client not initialized. Use 'async with' context manager.")
        
        response = await self.session.get(self.BASE_URL, params=params)
        response.raise_for_status()
        
        return response.json()
    
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
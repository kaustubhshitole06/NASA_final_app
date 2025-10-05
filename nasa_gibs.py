"""
NASA GIBS (Global Imagery Browse Services) Client
Provides access to NASA satellite imagery for the frontend map.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json

class NASAGIBSClient:
    """Client for NASA GIBS satellite imagery services"""
    
    # Base URLs for NASA GIBS services
    WMTS_BASE_URL = "https://gibs.earthdata.nasa.gov/wmts"
    WMS_BASE_URL = "https://gibs.earthdata.nasa.gov/wms"
    
    # Available projections
    PROJECTIONS = {
        "geographic": "epsg4326",
        "web_mercator": "epsg3857",
        "north_polar": "epsg3413",
        "south_polar": "epsg3031"
    }
    
    # Common image layers (subset)
    DEFAULT_LAYERS = {
        "true_color": "MODIS_Terra_CorrectedReflectance_TrueColor",
        "night_lights": "VIIRS_SNPP_DayNightBand_ENCC",
        "cloud_cover": "MODIS_Terra_Cloud_Top_Temp_Day",
        "temperature": "MODIS_Terra_Land_Surface_Temp_Day",
        "precipitation": "GPM_IMERG_Precipitation_Rate",
        "snow_cover": "MODIS_Terra_Snow_Cover",
        "vegetation": "MODIS_Terra_NDVI",
        "aerosol": "MODIS_Terra_Aerosol"
    }
    
    # Resolution levels
    RESOLUTIONS = {
        "very_high": "250m",
        "high": "500m",
        "medium": "1km",
        "low": "2km"
    }
    
    def __init__(self):
        """Initialize the NASA GIBS client"""
        self.default_projection = "geographic"  # epsg4326
        self.default_resolution = "medium"      # 1km
        self.default_layer = "true_color"       # MODIS True Color
    
    def get_tile_url(
        self,
        date: Optional[datetime] = None,
        layer: str = None,
        projection: str = None,
        resolution: str = None,
        x: int = 0,
        y: int = 0,
        z: int = 0
    ) -> str:
        """
        Generate a URL for a specific GIBS map tile
        
        Args:
            date: Date for the imagery (defaults to yesterday)
            layer: Layer type (from DEFAULT_LAYERS or direct layer ID)
            projection: Map projection to use
            resolution: Resolution level
            x, y, z: Tile coordinates and zoom level
            
        Returns:
            URL string for the requested tile
        """
        # Set defaults
        if date is None:
            # Default to yesterday as most recent imagery may not be processed
            date = datetime.now() - timedelta(days=1)
        
        date_str = date.strftime("%Y-%m-%d")
        
        # Handle layer selection
        selected_layer = self.DEFAULT_LAYERS.get(
            layer or self.default_layer, 
            layer or self.DEFAULT_LAYERS[self.default_layer]
        )
        
        # Handle projection
        selected_projection = self.PROJECTIONS.get(
            projection or self.default_projection,
            projection or self.PROJECTIONS[self.default_projection]
        )
        
        # Handle resolution
        selected_resolution = self.RESOLUTIONS.get(
            resolution or self.default_resolution,
            resolution or self.RESOLUTIONS[self.default_resolution]
        )
        
        # Build the tile URL
        url = (
            f"{self.WMTS_BASE_URL}/{selected_projection}/best/{selected_layer}/default/"
            f"{date_str}/{selected_resolution}/{z}/{y}/{x}.png"
        )
        
        return url
    
    def get_wms_url(
        self,
        date: Optional[datetime] = None,
        layer: str = None,
        bbox: str = "-180,-90,180,90",
        width: int = 1024,
        height: int = 512
    ) -> str:
        """
        Generate a URL for a WMS GetMap request
        
        Args:
            date: Date for the imagery
            layer: Layer name
            bbox: Bounding box (minx,miny,maxx,maxy)
            width: Image width in pixels
            height: Image height in pixels
            
        Returns:
            URL for WMS GetMap request
        """
        if date is None:
            date = datetime.now() - timedelta(days=1)
        
        date_str = date.strftime("%Y-%m-%d")
        
        selected_layer = self.DEFAULT_LAYERS.get(
            layer or self.default_layer, 
            layer or self.DEFAULT_LAYERS[self.default_layer]
        )
        
        url = (
            f"{self.WMS_BASE_URL}/epsg4326/wms.cgi?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap"
            f"&LAYERS={selected_layer}&STYLES=&FORMAT=image/png&CRS=EPSG:4326"
            f"&TIME={date_str}&WIDTH={width}&HEIGHT={height}&BBOX={bbox}"
        )
        
        return url
    
    def get_available_layers(self) -> List[Dict[str, str]]:
        """
        Get list of available imagery layers with descriptions
        
        Returns:
            List of layer information dictionaries
        """
        # This is a subset of available layers - a full implementation would query
        # the GIBS capabilities document
        return [
            {
                "id": "MODIS_Terra_CorrectedReflectance_TrueColor",
                "name": "True Color (MODIS Terra)",
                "description": "Corrected reflectance imagery showing land surface, oceans, and clouds",
                "category": "base_layer" 
            },
            {
                "id": "VIIRS_SNPP_DayNightBand_ENCC",
                "name": "Night Lights (VIIRS)",
                "description": "Earth at night showing city lights and other illuminated areas",
                "category": "night"
            },
            {
                "id": "MODIS_Terra_Cloud_Top_Temp_Day",
                "name": "Cloud Top Temperature",
                "description": "Temperature at the top of clouds, useful for identifying storm systems",
                "category": "clouds"
            },
            {
                "id": "MODIS_Terra_Land_Surface_Temp_Day",
                "name": "Land Surface Temperature",
                "description": "Daytime land surface temperature observations",
                "category": "temperature"
            },
            {
                "id": "GPM_IMERG_Precipitation_Rate",
                "name": "Precipitation Rate",
                "description": "Recent precipitation measurements from multiple satellites",
                "category": "precipitation"
            },
            {
                "id": "MODIS_Terra_Snow_Cover",
                "name": "Snow Cover",
                "description": "Areas covered by snow, shown as white on a blue background",
                "category": "cryosphere"
            },
            {
                "id": "MODIS_Terra_NDVI",
                "name": "Vegetation Index (NDVI)",
                "description": "Normalized Difference Vegetation Index showing plant health and density",
                "category": "land"
            },
            {
                "id": "MODIS_Terra_Aerosol",
                "name": "Aerosol Optical Depth",
                "description": "Airborne particulate matter concentrations",
                "category": "air_quality"
            }
        ]
    
    def get_layer_categories(self) -> Dict[str, List[str]]:
        """
        Get layer categories for organization in UI
        
        Returns:
            Dictionary of category names and associated layer IDs
        """
        return {
            "Base Layers": ["MODIS_Terra_CorrectedReflectance_TrueColor"],
            "Weather": [
                "MODIS_Terra_Cloud_Top_Temp_Day",
                "GPM_IMERG_Precipitation_Rate",
                "MODIS_Terra_Land_Surface_Temp_Day"
            ],
            "Environment": [
                "MODIS_Terra_NDVI",
                "MODIS_Terra_Snow_Cover",
                "MODIS_Terra_Aerosol"
            ],
            "Night": ["VIIRS_SNPP_DayNightBand_ENCC"]
        }
    
    def get_time_range(self, layer: str) -> Dict[str, Any]:
        """
        Get available time range for a specific layer
        
        Args:
            layer: Layer ID or shorthand name
            
        Returns:
            Dictionary with time range information
        """
        # In a full implementation, this would query the capabilities document
        # Here we return fixed ranges for demo purposes
        
        selected_layer = self.DEFAULT_LAYERS.get(layer, layer)
        
        if selected_layer in ["VIIRS_SNPP_DayNightBand_ENCC"]:
            start_date = datetime(2012, 4, 1)
        elif selected_layer.startswith("MODIS_Terra"):
            start_date = datetime(2000, 2, 24)
        else:
            start_date = datetime(2015, 1, 1)
            
        end_date = datetime.now() - timedelta(days=1)
        
        return {
            "layer": selected_layer,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "period": "daily",
            "has_time": True
        }
    
    def build_leaflet_provider_config(self) -> Dict[str, Any]:
        """
        Build configuration for Leaflet.js TileLayer
        
        Returns:
            Dictionary with configuration for Leaflet.TileLayer
        """
        return {
            "url": f"{self.WMTS_BASE_URL}/epsg3857/best/{{layer}}/default/{{time}}/{{resolution}}/{{z}}/{{y}}/{{x}}.png",
            "options": {
                "layer": "MODIS_Terra_CorrectedReflectance_TrueColor",
                "time": datetime.now().strftime("%Y-%m-%d"),
                "resolution": "250m",
                "tileSize": 256,
                "attribution": "NASA GIBS",
                "bounds": [[-85.0511287776, -180], [85.0511287776, 180]]
            }
        }

# Usage examples
if __name__ == "__main__":
    client = NASAGIBSClient()
    
    # Get a tile URL for yesterday's true color imagery
    tile_url = client.get_tile_url()
    print(f"Example tile URL: {tile_url}")
    
    # Get a WMS URL for a specific area
    wms_url = client.get_wms_url(layer="precipitation", bbox="-100,30,-80,50")
    print(f"Example WMS URL: {wms_url}")
    
    # Get available layers
    layers = client.get_available_layers()
    print(f"Available layers: {len(layers)}")
    
    # Get Leaflet configuration
    leaflet_config = client.build_leaflet_provider_config()
    print(f"Leaflet config: {json.dumps(leaflet_config)}")
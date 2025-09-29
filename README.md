# NASA Weather Probability Dashboard

A comprehensive web application that helps users determine the likelihood of adverse weather conditions for specific locations and times using NASA POWER Earth observation data.

## 🌟 Features

- **Interactive Location Selection**: Click on map or search for locations
- **Historical Weather Analysis**: Analyze 5-20 years of historical data
- **Multiple Weather Parameters**: Temperature, precipitation, wind speed, humidity
- **Probability Calculations**: Calculate odds of extreme weather conditions
- **NASA Satellite Imagery**: Integrated NASA GIBS satellite visualization
- **Visual Analytics**: Interactive charts and probability displays
- **Comfort Index**: Combined weather comfort scoring
- **Data Export**: Download results as CSV files
- **Trend Analysis**: Detect climate trends over time

## 🛠️ Technology Stack

### Backend
- **FastAPI**: Modern Python web framework
- **NASA POWER API**: Earth observation data source
- **pandas/numpy**: Statistical analysis
- **httpx**: Async HTTP client

### Frontend  
- **HTML5/CSS3/JavaScript**: Core web technologies
- **Leaflet.js**: Interactive maps
- **Chart.js**: Data visualizations
- **Bootstrap 5**: Responsive UI framework

## 📊 NASA Data Integrations

### NASA POWER API Integration

This application integrates with NASA's POWER (Prediction Of Worldwide Energy Resources) API to access:

- **T2M**: Temperature at 2 meters
- **T2M_MAX/MIN**: Maximum/minimum temperatures
- **PRECTOTCORR**: Precipitation corrected data
- **WS2M/WS10M**: Wind speed at 2m/10m height  
- **RH2M**: Relative humidity at 2 meters
- **PS**: Surface pressure
- **QV2M**: Specific humidity

#### Example API Endpoint
```
https://power.larc.nasa.gov/api/temporal/daily/point?parameters=T2M&community=AG&latitude=12.97&longitude=77.59&start=20200101&end=20201231&format=JSON
```

### NASA Worldview / GIBS Integration

The application also integrates NASA's Global Imagery Browse Services (GIBS) to provide satellite visualization:

- **True Color Imagery**: Natural color satellite images from MODIS
- **Cloud Coverage**: Cloud top temperature visualization
- **Precipitation**: Global precipitation data visualization
- **Temperature**: Land surface temperature imagery
- **Vegetation**: NDVI (Normalized Difference Vegetation Index)
- **Night Lights**: City lights and illumination at night
- **Snow Cover**: Snow and ice visualization

#### Example GIBS Endpoint
```
https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/MODIS_Terra_CorrectedReflectance_TrueColor/default/2020-07-01/250m/0/0/0.png
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Modern web browser

### Installation

1. **Clone/Download the project**
   ```bash
   cd "C:\Users\User\Desktop\NASA"
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the FastAPI server**
   ```bash
   python main.py
   ```
   Or using uvicorn:
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

4. **Open the frontend**
   - Open `index.html` in your web browser
   - Or serve it using a simple HTTP server:
     ```bash
     python -m http.server 8080
     ```
   - Then visit `http://localhost:8080`

### API Documentation
- FastAPI server: `http://localhost:8000`
- Interactive API docs: `http://localhost:8000/docs`
- OpenAPI schema: `http://localhost:8000/openapi.json`

## 📋 How to Use

### 1. Select Location
- **Click on the map** to select your desired location
- **Use current location** button for GPS coordinates  
- **Search** for locations (basic implementation provided)

### 2. Choose Date & Parameters
- **Target Date**: Select the date you want to analyze
- **Years of History**: Choose how many years of historical data to analyze (5-20 years)
- **Weather Parameters**: Select which conditions to analyze:
  - 🌡️ Temperature
  - 🌧️ Precipitation  
  - 💨 Wind Speed
  - 💧 Humidity
  
### 3. Explore Satellite Imagery
- **Base Layers**: Switch between map and satellite imagery
- **Weather Overlays**: Add cloud cover, precipitation, temperature visualizations
- **Environmental Layers**: View vegetation, snow cover, air quality
- **Date Selection**: Choose different dates for satellite imagery

### 4. Analyze Probabilities
- Click **"Analyze Weather Probability"**
- View probability percentages for different weather conditions:
  - Very Hot/Cold conditions
  - Wet/Very Wet conditions  
  - Windy/Very Windy conditions
  - Humid/Dry conditions

### 5. View Results
- **Probability Cards**: Shows likelihood percentages
- **Historical Trends**: Line charts of historical data
- **Distribution Charts**: Comparison across parameters
- **Comfort Index**: Overall weather comfort analysis

### 6. Export Data
- Click **"Export Data (CSV)"** to download results
- Includes metadata and source information

## 🔧 API Endpoints

### Core Endpoints

#### `GET /`
Basic API information

#### `GET /parameters`
List available weather parameters

#### `POST /weather/raw`
Fetch raw weather data from NASA POWER API

**Request Body:**
```json
{
  "latitude": 12.97,
  "longitude": 77.59,
  "start_date": "20200101",
  "end_date": "20201231", 
  "parameters": ["T2M", "PRECTOTCORR"],
  "community": "AG"
}
```

#### `POST /weather/probability`
Calculate weather probabilities based on historical data

**Request Body:**
```json
{
  "latitude": 12.97,
  "longitude": 77.59,
  "target_date": "0315",
  "years_back": 10,
  "parameters": ["T2M", "PRECTOTCORR", "WS2M"],
  "thresholds": {
    "T2M": {"hot": 35, "cold": 10},
    "PRECTOTCORR": {"wet": 10, "very_wet": 25}
  }
}
```

#### `GET /weather/export/csv`
Export weather data as CSV file

**Query Parameters:**
- `latitude`: Latitude coordinate
- `longitude`: Longitude coordinate
- `start_date`: Start date (YYYYMMDD)
- `end_date`: End date (YYYYMMDD)
- `parameters`: Comma-separated parameter list

## 📊 Weather Thresholds

### Temperature (°C)
- **Very Hot**: > 35°C
- **Hot**: > 30°C  
- **Cold**: < 10°C
- **Very Cold**: < 0°C

### Precipitation (mm/day)
- **Wet**: > 10mm
- **Very Wet**: > 25mm
- **Extreme Wet**: > 50mm

### Wind Speed (m/s)
- **Windy**: > 10 m/s
- **Very Windy**: > 15 m/s
- **Extreme Windy**: > 20 m/s

### Humidity (%)
- **Humid**: > 70%
- **Very Humid**: > 80%
- **Dry**: < 30%
- **Very Dry**: < 20%

## 🧮 Statistical Analysis

The application provides comprehensive statistical analysis:

### Basic Statistics
- Mean, median, standard deviation
- Min/max values, quartiles
- Data completeness metrics

### Probability Calculations
- Threshold exceedance probabilities
- Historical trend analysis
- Extreme event return periods

### Comfort Index
- Multi-parameter comfort scoring
- Temperature, humidity, wind combination
- Comfort probability categories

## 🗂️ Project Structure

```
NASA/
├── main.py                 # FastAPI application
├── nasa_client.py          # NASA POWER API client
├── nasa_gibs.py            # NASA GIBS satellite imagery client
├── weather_analytics.py    # Statistical analysis functions
├── requirements.txt        # Python dependencies
├── index.html              # Frontend interface
└── README.md               # Documentation
```

## 🔍 Example Use Cases

### 1. Vacation Planning
- Check probability of rain during summer vacation
- Analyze temperature trends for beach destinations
- Plan outdoor activities with comfort index

### 2. Event Planning
- Wedding venue weather likelihood
- Festival planning with wind/rain probabilities
- Sports event weather conditions

### 3. Agricultural Planning
- Planting season weather analysis
- Irrigation planning with precipitation data
- Growing season temperature trends

### 4. Research & Analysis
- Climate trend analysis over decades
- Extreme weather event frequency
- Regional weather pattern studies

## 🌐 NASA Data Sources

This application leverages NASA's comprehensive Earth observation data:

- **MERRA-2**: Modern-Era Retrospective analysis
- **GEOS-5**: Goddard Earth Observing System  
- **Satellite Data**: Various NASA Earth observation missions
- **Ground Observations**: Weather station data integration

## 🚧 Future Enhancements

### Planned Features
- [ ] Advanced location search with geocoding
- [ ] Multi-location comparison analysis
- [ ] Weather alerts and notifications
- [ ] Mobile-responsive progressive web app
- [ ] Historical event correlation analysis
- [ ] Custom threshold configuration
- [ ] Seasonal analysis patterns
- [ ] Social sharing of results

### Technical Improvements
- [ ] Data caching for performance
- [ ] Background data processing
- [ ] Advanced statistical models
- [ ] Real-time data integration
- [ ] User authentication and saved locations
- [ ] API rate limiting and optimization

## 📄 License

This project is developed for the NASA Space Apps Challenge and uses NASA's public Earth observation data APIs.

## 🤝 Contributing

1. Fork the repository
2. Create feature branches
3. Submit pull requests with clear descriptions
4. Follow coding standards and documentation

## 📞 Support

For questions about NASA POWER data: [larc-power-project@mail.nasa.gov](mailto:larc-power-project@mail.nasa.gov)

For application issues: Create GitHub issues with detailed descriptions

---

**Built with NASA Earth observation data for better outdoor planning! 🛰️🌍**
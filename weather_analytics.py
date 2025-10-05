import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from datetime import datetime

class WeatherStatistics:
    """Statistical analysis functions for weather probability calculations"""
    
    @staticmethod
    def calculate_basic_stats(values: List[float]) -> Dict[str, float]:
        """Calculate basic statistical measures"""
        if not values:
            return {}
        
        values_array = np.array(values)
        
        return {
            "mean": float(np.mean(values_array)),
            "median": float(np.median(values_array)),
            "std": float(np.std(values_array)),
            "min": float(np.min(values_array)),
            "max": float(np.max(values_array)),
            "q25": float(np.percentile(values_array, 25)),
            "q75": float(np.percentile(values_array, 75)),
            "count": len(values)
        }
    
    @staticmethod
    def calculate_threshold_probabilities(
        values: List[float], 
        thresholds: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate probability of exceeding/falling below thresholds"""
        if not values:
            return {}
        
        values_array = np.array(values)
        probabilities = {}
        
        for condition, threshold in thresholds.items():
            condition_lower = condition.lower()
            
            if any(term in condition_lower for term in ["hot", "warm", "wet", "windy", "humid"]):
                # For conditions where higher values are "worse"
                probability = np.mean(values_array >= threshold) * 100
            elif any(term in condition_lower for term in ["cold", "cool", "dry"]):
                # For conditions where lower values are "worse"  
                probability = np.mean(values_array <= threshold) * 100
            else:
                # Default to "greater than or equal to"
                probability = np.mean(values_array >= threshold) * 100
            
            probabilities[condition] = round(probability, 2)
        
        return probabilities
    
    @staticmethod
    def calculate_comfort_index(
        temperature: List[float],
        humidity: List[float] = None,
        wind_speed: List[float] = None
    ) -> Dict[str, Any]:
        """Calculate weather comfort index combining multiple factors"""
        
        if not temperature:
            return {}
        
        comfort_scores = []
        
        for i, temp in enumerate(temperature):
            score = 0
            
            # Temperature comfort (optimal around 20-25°C)
            if 20 <= temp <= 25:
                temp_score = 10
            elif 15 <= temp <= 30:
                temp_score = 7
            elif 10 <= temp <= 35:
                temp_score = 4
            else:
                temp_score = 1
            
            score += temp_score * 0.6  # Temperature weight: 60%
            
            # Humidity comfort (if available)
            if humidity and i < len(humidity):
                hum = humidity[i]
                if 40 <= hum <= 60:
                    hum_score = 10
                elif 30 <= hum <= 70:
                    hum_score = 7
                elif 20 <= hum <= 80:
                    hum_score = 4
                else:
                    hum_score = 1
                
                score += hum_score * 0.3  # Humidity weight: 30%
            else:
                score += 7 * 0.3  # Default neutral humidity score
            
            # Wind speed comfort (if available) 
            if wind_speed and i < len(wind_speed):
                wind = wind_speed[i]
                if wind <= 5:
                    wind_score = 10
                elif wind <= 10:
                    wind_score = 7
                elif wind <= 15:
                    wind_score = 4
                else:
                    wind_score = 1
                
                score += wind_score * 0.1  # Wind weight: 10%
            else:
                score += 7 * 0.1  # Default neutral wind score
            
            comfort_scores.append(score)
        
        comfort_array = np.array(comfort_scores)
        
        # Define comfort categories
        very_comfortable = np.mean(comfort_array >= 8) * 100
        comfortable = np.mean(comfort_array >= 6) * 100
        uncomfortable = np.mean(comfort_array < 4) * 100
        very_uncomfortable = np.mean(comfort_array < 2) * 100
        
        return {
            "comfort_statistics": {
                "mean_score": float(np.mean(comfort_array)),
                "min_score": float(np.min(comfort_array)),
                "max_score": float(np.max(comfort_array))
            },
            "comfort_probabilities": {
                "very_comfortable": round(very_comfortable, 2),
                "comfortable": round(comfortable, 2), 
                "uncomfortable": round(uncomfortable, 2),
                "very_uncomfortable": round(very_uncomfortable, 2)
            },
            "daily_scores": comfort_scores
        }
    
    @staticmethod
    def detect_trends(values: List[float], years: List[int]) -> Dict[str, Any]:
        """Detect trends in weather data over time"""
        if len(values) != len(years) or len(values) < 3:
            return {"trend": "insufficient_data"}
        
        # Simple linear regression to detect trend
        x = np.array(years) - min(years)  # Normalize years
        y = np.array(values)
        
        # Calculate slope
        n = len(x)
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - (np.sum(x))**2)
        
        # Calculate correlation coefficient
        correlation = np.corrcoef(x, y)[0, 1] if not np.isnan(np.corrcoef(x, y)[0, 1]) else 0
        
        # Determine trend strength and direction
        if abs(correlation) < 0.3:
            trend = "no_trend"
            strength = "weak"
        elif abs(correlation) < 0.7:
            strength = "moderate" 
        else:
            strength = "strong"
        
        if correlation > 0.3:
            trend = "increasing"
        elif correlation < -0.3:
            trend = "decreasing"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "strength": strength,
            "slope": round(slope, 4),
            "correlation": round(correlation, 3),
            "change_per_year": round(slope, 4)
        }
    
    @staticmethod
    def calculate_extreme_event_probability(
        values: List[float],
        return_periods: List[int] = [2, 5, 10, 20, 50]
    ) -> Dict[str, float]:
        """Calculate probability of extreme events using return periods"""
        if not values or len(values) < 5:
            return {}
        
        values_sorted = np.sort(values)[::-1]  # Sort descending for extreme high values
        n = len(values)
        
        extreme_values = {}
        
        for period in return_periods:
            if period <= n:
                # Calculate the value exceeded once every 'period' years
                rank = n / period
                if rank <= n:
                    index = int(rank - 1)
                    extreme_values[f"{period}_year_return"] = round(float(values_sorted[index]), 2)
        
        return extreme_values

class WeatherAnalyzer:
    """Main class for comprehensive weather analysis"""
    
    def __init__(self):
        self.stats = WeatherStatistics()
    
    def analyze_weather_data(
        self,
        data: List[Dict[str, Any]],
        parameter: str,
        thresholds: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """Comprehensive analysis of weather data for a single parameter"""
        
        # Extract values and years
        values = [item["value"] for item in data if item["parameter"] == parameter]
        years = [item["year"] for item in data if item["parameter"] == parameter]
        
        if not values:
            return {"error": "No data available for parameter"}
        
        # Basic statistics
        basic_stats = self.stats.calculate_basic_stats(values)
        
        # Threshold probabilities
        threshold_probs = {}
        if thresholds:
            threshold_probs = self.stats.calculate_threshold_probabilities(values, thresholds)
        
        # Trend analysis
        trend_analysis = self.stats.detect_trends(values, years)
        
        # Extreme event analysis
        extreme_events = self.stats.calculate_extreme_event_probability(values)
        
        return {
            "parameter": parameter,
            "basic_statistics": basic_stats,
            "threshold_probabilities": threshold_probs,
            "trend_analysis": trend_analysis,
            "extreme_events": extreme_events,
            "data_quality": {
                "total_points": len(values),
                "years_covered": len(set(years)) if years else 0,
                "completeness": round(len(values) / len(set(years)) * 100, 1) if years else 0
            }
        }
    
    def analyze_multi_parameter(
        self,
        data: List[Dict[str, Any]],
        parameters: List[str],
        include_comfort_index: bool = True
    ) -> Dict[str, Any]:
        """Analyze multiple weather parameters together"""
        
        results = {}
        
        # Analyze each parameter individually
        for param in parameters:
            param_data = [item for item in data if item["parameter"] == param]
            if param_data:
                results[param] = self.analyze_weather_data(data, param)
        
        # Calculate comfort index if temperature and humidity are available
        if include_comfort_index and "T2M" in parameters:
            temp_values = [item["value"] for item in data if item["parameter"] == "T2M"]
            hum_values = [item["value"] for item in data if item["parameter"] == "RH2M"]
            wind_values = [item["value"] for item in data if item["parameter"] == "WS2M"]
            
            if temp_values:
                comfort_analysis = self.stats.calculate_comfort_index(
                    temperature=temp_values,
                    humidity=hum_values if hum_values else None,
                    wind_speed=wind_values if wind_values else None
                )
                results["comfort_index"] = comfort_analysis
        
        return results
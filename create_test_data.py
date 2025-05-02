import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_test_data(output_file='data/hourly_trips_with_weather.csv', n_samples=200):
    """
    Create a synthetic dataset for testing the trip prediction model
    
    Args:
        output_file: Path to save the CSV file
        n_samples: Number of hourly samples to generate
    """
    # Generate datetime range
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(hours=i) for i in range(n_samples)]
    
    # Create base trip counts with daily and weekly seasonality
    base_counts = []
    for i in range(n_samples):
        hour = dates[i].hour
        day_of_week = dates[i].weekday()
        
        # Daily pattern: more trips during commute hours
        hour_factor = 1.0
        if 7 <= hour <= 9:  # Morning rush
            hour_factor = 2.0
        elif 16 <= hour <= 18:  # Evening rush
            hour_factor = 2.5
        elif 0 <= hour <= 5:  # Night
            hour_factor = 0.3
            
        # Weekly pattern: fewer trips on weekends
        day_factor = 1.0
        if day_of_week >= 5:  # Weekend
            day_factor = 0.7
            
        # Base count with some randomness
        base_count = 50 * hour_factor * day_factor
        noise = np.random.normal(0, base_count * 0.2)  # 20% noise
        base_counts.append(max(0, base_count + noise))
    
    # Create weather data
    temp = []
    humidity = []
    wind_speed = []
    clouds = []
    rain_1h = []
    snow_1h = []
    weather_main = []
    
    weather_options = ['Clear', 'Clouds', 'Rain', 'Snow', 'Mist']
    weather_probs = [0.5, 0.3, 0.1, 0.05, 0.05]
    
    for i in range(n_samples):
        # Temperature follows daily cycle and adds week-to-week variation
        hour = dates[i].hour
        day_of_year = dates[i].timetuple().tm_yday
        
        # Daily temperature cycle
        hour_temp = 15 + 5 * np.sin(np.pi * (hour - 14) / 12)
        
        # Add some weekly variation
        week_temp = 3 * np.sin(2 * np.pi * day_of_year / 365)
        
        # Final temperature with noise
        t = hour_temp + week_temp + np.random.normal(0, 2)
        temp.append(t)
        
        # Humidity inverse to temperature
        h = 60 - t/2 + np.random.normal(0, 10)
        h = max(10, min(100, h))
        humidity.append(h)
        
        # Wind speed
        w = 5 + np.random.exponential(3)
        wind_speed.append(w)
        
        # Cloud cover percentage
        c = np.random.beta(2, 3) * 100
        clouds.append(c)
        
        # Weather type (weighted random)
        weather = np.random.choice(weather_options, p=weather_probs)
        weather_main.append(weather)
        
        # Rain amount (mostly zero, sometimes positive)
        if weather == 'Rain':
            r = np.random.exponential(2)
        else:
            r = 0
        rain_1h.append(r)
        
        # Snow amount (mostly zero, sometimes positive)
        if weather == 'Snow':
            s = np.random.exponential(1)
        else:
            s = 0
        snow_1h.append(s)
    
    # Create final dataframe
    df = pd.DataFrame({
        'datetime': dates,
        'trip_count': base_counts,
        'temp': temp,
        'humidity': humidity,
        'wind_speed': wind_speed,
        'clouds': clouds,
        'rain_1h': rain_1h,
        'snow_1h': snow_1h,
        'weather_main': weather_main
    })
    
    # Add a few more weather variables
    df['feels_like'] = df['temp'] - 3 * np.sqrt(df['wind_speed']) / 10
    df['pressure'] = 1013 + np.random.normal(0, 5, size=n_samples)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Created test dataset with {n_samples} hourly samples at: {output_file}")
    
    return df

if __name__ == "__main__":
    import os
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Create test data
    create_test_data(n_samples=500)
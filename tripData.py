import pandas as pd
import numpy as np
import glob
from datetime import datetime, timedelta
import requests
import time
import os

# Function to load and process all Bluebikes trip CSV files
def load_trip_data(data_directory):
    # Get all CSV files in directory
    all_files = glob.glob(os.path.join(data_directory, "*.csv"))
    
    # Create empty list to store dataframes
    dfs = []
    
    # Read each file into a dataframe and append to list
    for file in all_files:
        try:
            # Read the file
            df = pd.read_csv(file)
            
            # Determine the date columns based on header format
            if 'started_at' in df.columns:
                start_time_col = 'started_at'
                end_time_col = 'ended_at'
            elif 'starttime' in df.columns:
                start_time_col = 'starttime'
                end_time_col = 'stoptime'
            else:
                # If neither format is found, skip this file
                print(f"Warning: {file} has unknown format, skipping")
                continue
            
            # Add file source for tracking
            df['source_file'] = os.path.basename(file)
            
            # Standardize columns for time data
            df['standardized_start_time'] = pd.to_datetime(df[start_time_col])
            df['standardized_end_time'] = pd.to_datetime(df[end_time_col])
            
            # Extract date components using standardized columns
            df['start_date'] = df['standardized_start_time'].dt.date
            df['start_hour'] = df['standardized_start_time'].dt.hour
            df['start_day_of_week'] = df['standardized_start_time'].dt.dayofweek
            df['start_month'] = df['standardized_start_time'].dt.month
            
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    if not dfs:
        raise ValueError("No valid CSV files were found or processed")
    
    # Concatenate all dataframes
    try:
        # Try to concatenate directly first
        combined_df = pd.concat(dfs, ignore_index=True)
    except ValueError as e:
        # If direct concatenation fails, try a more flexible approach
        print(f"Warning: Column mismatch detected, using a more flexible concatenation approach")
        combined_df = pd.concat(dfs, ignore_index=True, sort=False)
    
    return combined_df

# Aggregate data by day
def aggregate_daily_trips(trips_df):
    # Group by date and count trips
    daily_trips = trips_df.groupby('start_date').size().reset_index()
    daily_trips.columns = ['date', 'trip_count']
    
    # Add additional time features
    daily_trips['day_of_week'] = pd.to_datetime(daily_trips['date']).dt.dayofweek
    daily_trips['month'] = pd.to_datetime(daily_trips['date']).dt.month
    daily_trips['is_weekend'] = daily_trips['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    return daily_trips

# For hourly data if needed
def aggregate_hourly_trips(trips_df):
    # Create datetime key for hour-level grouping using the standardized column
    trips_df['hour_key'] = trips_df['standardized_start_time'].dt.floor('H')
    
    # Group by hour and count trips
    hourly_trips = trips_df.groupby(['hour_key']).size().reset_index()
    hourly_trips.columns = ['datetime', 'trip_count']
    
    # Add time features
    hourly_trips['hour'] = hourly_trips['datetime'].dt.hour
    hourly_trips['day'] = hourly_trips['datetime'].dt.day
    hourly_trips['day_of_week'] = hourly_trips['datetime'].dt.dayofweek
    hourly_trips['month'] = hourly_trips['datetime'].dt.month
    hourly_trips['is_weekend'] = hourly_trips['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Determine station ID column names based on available columns
    if 'start_station_id' in trips_df.columns:
        start_station_id_col = 'start_station_id'
        start_lat_col = 'start_lat'
        start_lng_col = 'start_lng'
    elif 'start station id' in trips_df.columns:
        start_station_id_col = 'start station id'
        start_lat_col = 'start station latitude'
        start_lng_col = 'start station longitude'
    else:
        # If neither format is found, use placeholder values
        print("Warning: Could not find station ID columns, using placeholder values for lat/lng")
        hourly_trips['latitude'] = np.nan
        hourly_trips['longitude'] = np.nan
        return hourly_trips
    
    # Get representative lat/lng for stations (use most common station for simplicity)
    try:
        most_common_station = trips_df[start_station_id_col].value_counts().index[0]
        station_info = trips_df[trips_df[start_station_id_col] == most_common_station].iloc[0]
        
        hourly_trips['latitude'] = station_info[start_lat_col]
        hourly_trips['longitude'] = station_info[start_lng_col]
    except (IndexError, KeyError) as e:
        print(f"Warning: Error getting station coordinates: {e}")
        hourly_trips['latitude'] = np.nan
        hourly_trips['longitude'] = np.nan
    
    return hourly_trips

# Function to create aggregations with error handling
def create_aggregations(trips_df):
    try:
        # Create daily aggregation
        daily_trips = aggregate_daily_trips(trips_df)
        print(f"Created daily aggregation with {len(daily_trips)} rows")
        
        # Create hourly aggregation
        hourly_trips = aggregate_hourly_trips(trips_df)
        print(f"Created hourly aggregation with {len(hourly_trips)} rows")
        
        return daily_trips, hourly_trips
    except Exception as e:
        print(f"Error creating aggregations: {e}")
        return None, None

# Load all trip data
trips_df = load_trip_data("./trips")
print(f"Loaded {len(trips_df)} trips")
print(f"Trips:{trips_df}" )

# Print column names to verify
print("Available columns:", trips_df.columns.tolist())

# Create aggregations
daily_trips, hourly_trips = create_aggregations(trips_df)
#print(f"Daily Trips:{daily_trips}")


#print(f"Hourly Trips:{hourly_trips}")
# generate a csv with hourly trips
if hourly_trips is not None:
    # Save the hourly trips to a CSV file
    hourly_trips.to_csv('hourly_trips_aggregated.csv', index=False)
    print("Hourly trips aggregation saved to 'hourly_trips_aggregated.csv'")
    

def get_weather_data(api_key, lat, lon, datetime_list):
    """
    Retrieve historical weather data using OpenWeatherMap API
    
    Parameters:
    - api_key: Your OpenWeatherMap API key
    - lat, lon: Latitude and longitude coordinates
    - datetime_list: List of datetime objects for which to retrieve weather data
    
    Returns:
    - DataFrame with weather data for each datetime
    """
    base_url = "https://api.openweathermap.org/data/3.0/onecall/timemachine"
    weather_data_list = []
    
    # Track unique timestamps to avoid redundant API calls
    processed_timestamps = set()
    
    for dt in datetime_list:
        # Convert datetime to Unix timestamp
        timestamp = int(dt.timestamp())
        
        # Skip if we've already processed this timestamp
        if timestamp in processed_timestamps:
            continue
        
        processed_timestamps.add(timestamp)
        
        # Build API request URL
        params = {
            'lat': lat,
            'lon': lon,
            'dt': timestamp,
            'appid': api_key
            #'units': 'metric'  # Get temperature in Celsius
        }
        
        try:
            response = requests.get(base_url, params=params)
            # print the request URL for debugging
            print(f"Request URL: {response.url}")  # For debugging purposes
            if response.status_code != 200:
                print(f"Error with API request: {response.status_code} - {response.text}")
                continue
                
            data = response.json()
            
            if 'data' in data and len(data['data']) > 0:
                weather = data['data'][0]
                
                # Extract all relevant weather information
                weather_dict = {
                    'timestamp': dt,
                    'temp': weather.get('temp'), # Temperature in Kelvin
                    'feels_like': weather.get('feels_like'),
                    'pressure': weather.get('pressure'),
                    'humidity': weather.get('humidity'),
                    'dew_point': weather.get('dew_point'),
                    'uvi': weather.get('uvi'),
                    'clouds': weather.get('clouds'),
                    'visibility': weather.get('visibility'),
                    'wind_speed': weather.get('wind_speed'),
                    'wind_deg': weather.get('wind_deg'),
                    'weather_id': weather.get('weather', [{}])[0].get('id'),
                    'weather_main': weather.get('weather', [{}])[0].get('main'),
                    'weather_description': weather.get('weather', [{}])[0].get('description'),
                    'weather_icon': weather.get('weather', [{}])[0].get('icon')
                }
                
                # Add sunrise and sunset if available
                if 'sunrise' in weather:
                    weather_dict['sunrise'] = datetime.fromtimestamp(weather['sunrise'])
                if 'sunset' in weather:
                    weather_dict['sunset'] = datetime.fromtimestamp(weather['sunset'])
                
                # Add precipitation if available
                if 'rain' in weather:
                    if '1h' in weather['rain']:
                        weather_dict['rain_1h'] = weather['rain']['1h']
                    elif isinstance(weather['rain'], (int, float)):
                        weather_dict['rain_1h'] = weather['rain']
                    else:
                        weather_dict['rain_1h'] = 0
                else:
                    weather_dict['rain_1h'] = 0
                    
                # Add snow if available
                if 'snow' in weather:
                    if '1h' in weather['snow']:
                        weather_dict['snow_1h'] = weather['snow']['1h']
                    elif isinstance(weather['snow'], (int, float)):
                        weather_dict['snow_1h'] = weather['snow']
                    else:
                        weather_dict['snow_1h'] = 0
                else:
                    weather_dict['snow_1h'] = 0
                
                # Add timezone information
                weather_dict['timezone'] = data.get('timezone')
                weather_dict['timezone_offset'] = data.get('timezone_offset')
                
                weather_data_list.append(weather_dict)
                print(f"Retrieved weather data for {dt}")
            else:
                print(f"No weather data available for {dt}")
            
            # Sleep to respect API rate limits
            time.sleep(1.1)  # Adjust based on your API plan
            
        except Exception as e:
            print(f"Error retrieving weather for {dt}: {e}")
    
    weather_df = pd.DataFrame(weather_data_list)
    return weather_df

def add_weather_to_hourly_trips():
    """
    Function to read hourly trips data, add weather data, and save to a new CSV
    """
    # Read the hourly trips data
    if not os.path.exists('hourly_trips_aggregated.csv'):
        print("Error: hourly_trips_aggregated.csv not found")
        return
    
    hourly_trips = pd.read_csv('hourly_trips_aggregated.csv')
    print(f"Loaded {len(hourly_trips)} hourly trip records")
    
    # Convert the datetime column to datetime objects
    hourly_trips['datetime'] = pd.to_datetime(hourly_trips['datetime'])
    
    # Extract unique datetimes for weather API calls
    # Since the API provides hourly data, we can just use the unique datetime values
    unique_datetimes = sorted(hourly_trips['datetime'].unique())
    
    # Sample data for development/testing - use only a subset of dates to avoid excessive API calls
    # Comment this out for production use
    #unique_datetimes = unique_datetimes[:24]  # First 24 hours for testing
    
    # Use latitude and longitude from the first record (or you can take the mean)
    first_record = hourly_trips.iloc[0]
    lat = first_record['latitude']
    lon = first_record['longitude']
    
    # Replace with your actual API key
    api_key = "375b6f0262558a074127469469d64b83"
    
    # Retrieve weather data
    print(f"Retrieving weather data for {len(unique_datetimes)} unique timestamps...")
    weather_df = get_weather_data(api_key, lat, lon, unique_datetimes)
    
    if len(weather_df) == 0:
        print("Error: No weather data retrieved")
        return
    
    print(f"Retrieved weather data for {len(weather_df)} timestamps")
    
    # Merge weather data with hourly trips
    # Convert timestamp to datetime for merging
    weather_df['datetime'] = pd.to_datetime(weather_df['timestamp'])
    
    # Merge on datetime column
    result = pd.merge(hourly_trips, weather_df, on='datetime', how='left')
    
    # Drop the redundant timestamp column
    if 'timestamp' in result.columns:
        result = result.drop(columns=['timestamp'])
    
    # Save the result to a new CSV
    result.to_csv('hourly_trips_with_weather.csv', index=False)
    print(f"Saved {len(result)} records to hourly_trips_with_weather.csv")
    
    # Also save a summary of the weather data for reference
    if len(weather_df) > 0:
        weather_df.to_csv('weather_data_summary.csv', index=False)
        print(f"Saved {len(weather_df)} records to weather_data_summary.csv")
    
    return result


add_weather_to_hourly_trips()

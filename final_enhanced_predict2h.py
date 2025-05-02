import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import holidays
import requests
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, Bidirectional, Concatenate
from tensorflow.keras.layers import GlobalAveragePooling1D, Flatten, RepeatVector, Permute
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.regularizers import l2, l1, l1_l2
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError, Huber
import tensorflow as tf
import os
from datetime import datetime, timedelta
import argparse

# Define SMAPE loss function
def smape_loss(y_true, y_pred):
    """
    Symmetric Mean Absolute Percentage Error
    Scale-invariant metric for regression problems
    """
    epsilon = 1e-10  # Small value to avoid division by zero
    return tf.reduce_mean(2.0 * tf.abs(y_pred - y_true) / (tf.abs(y_true) + tf.abs(y_pred) + epsilon))

class TripPrediction2Hour:
    def __init__(self, seq_length=12, ensemble=False, lr=0.0005, batch_size=32, 
                 dropout_rate=0.4, reg_type='l2', reg_value=0.001, 
                 optimizer_type='adam', loss_type='mse', skip_feature_importance=False):
        self.seq_length = seq_length
        self.prediction_horizon = 2  # 2 hours ahead
        self.model = None
        self.scaler_X = RobustScaler()  # RobustScaler handles outliers better
        self.scaler_y = MinMaxScaler()
        self.feature_names = None
        self.ensemble = ensemble
        self.models = {}  # For ensemble models
        self.weather_categories = None  # Store weather categories for consistent one-hot encoding
        self.skip_feature_importance = skip_feature_importance  # New parameter to skip feature importance
        
        # Store parameters for visualization
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.reg_type = reg_type
        self.reg_value = reg_value
        self.optimizer_type = optimizer_type
        self.loss_type = loss_type
        self.initial_lr = lr
        
        self.us_holidays = holidays.US()  # US holiday calendar
        self.X_train = None  # Store for feature importance analysis
        
        # Configure regularizer
        if reg_type == 'l2':
            self.regularizer = l2(reg_value)
        elif reg_type == 'l1':
            self.regularizer = l1(reg_value)
        elif reg_type == 'l1_l2':
            self.regularizer = l1_l2(l1=reg_value, l2=reg_value)
        else:
            self.regularizer = None
            
        # Configure optimizer
        if optimizer_type == 'adam':
            self.optimizer = Adam(learning_rate=lr, clipnorm=1.0)
        elif optimizer_type == 'rmsprop':
            self.optimizer = RMSprop(learning_rate=lr, clipnorm=1.0)
        elif optimizer_type == 'sgd':
            self.optimizer = SGD(learning_rate=lr, clipnorm=1.0, momentum=0.9)
            
        # Configure loss function
        if loss_type == 'mse':
            self.loss = 'mse'
        elif loss_type == 'mae':
            self.loss = 'mae'
        elif loss_type == 'huber':
            self.loss = Huber(delta=1.0)
        elif loss_type == 'smape':
            self.loss = smape_loss
            
    def preprocess_data(self, df, add_lags=True):
        # Create copy to avoid modifying the original
        data = df.copy()
        
        # First, let's check for NaN values and print column statistics
        print(f"Dataset shape before preprocessing: {data.shape}")
        nan_count = data.isna().sum().sum()
        print(f"Total NaN values before preprocessing: {nan_count}")
        
        # Convert datetime if it exists (with explicit format)
        if 'datetime' in data.columns:
            try:
                # Try to parse datetime with automatic detection
                data['datetime'] = pd.to_datetime(data['datetime'])
            except:
                # If that fails, try a specific format
                print("Automatic datetime parsing failed, using explicit format")
                data['datetime'] = pd.to_datetime(data['datetime'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        
        # Handle missing or infinite values before creating new features
        # Replace infinities with NaN first
        data = data.replace([np.inf, -np.inf], np.nan)
        
        # Print columns with NaN values
        nan_cols = data.columns[data.isna().any()].tolist()
        print(f"Columns with NaN values: {nan_cols}")
        
        # Handle missing values in numeric columns
        numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            # Check if column has NaN values
            nan_count = data[col].isna().sum()
            if nan_count > 0:
                print(f"Filling {nan_count} NaN values in column '{col}'")
                # If all values are NaN, fill with 0
                if nan_count == len(data):
                    data[col] = 0
                else:
                    # Otherwise use median
                    data[col] = data[col].fillna(data[col].median())
        
        # ---------- ENHANCED FEATURE ENGINEERING ----------
        
        # 1. Add lag features (for previous hours' trip counts)
        if add_lags and 'trip_count' in data.columns:
            print("Adding lag features...")
            # Sort by datetime to ensure correct lag creation
            if 'datetime' in data.columns:
                data = data.sort_values('datetime')
            
            # Create lag features with more focus on recent hours since we're predicting 2 hours ahead
            for lag in [1, 2, 3, 6, 9, 12, 24]:  # Modified lags for 2-hour prediction
                data[f'trip_count_lag_{lag}'] = data['trip_count'].shift(lag)
                
            # Calculate rolling statistics (moving averages)
            data['trip_count_ma_2h'] = data['trip_count'].rolling(window=2).mean()  # Added 2-hour MA
            data['trip_count_ma_3h'] = data['trip_count'].rolling(window=3).mean()
            data['trip_count_ma_6h'] = data['trip_count'].rolling(window=6).mean()
            data['trip_count_ma_12h'] = data['trip_count'].rolling(window=12).mean()
            data['trip_count_ma_24h'] = data['trip_count'].rolling(window=24).mean()
            
            # Add rolling standard deviations (captures trip count volatility)
            data['trip_count_std_2h'] = data['trip_count'].rolling(window=2).std()  # Added 2-hour STD
            data['trip_count_std_3h'] = data['trip_count'].rolling(window=3).std()
            data['trip_count_std_6h'] = data['trip_count'].rolling(window=6).std()
            data['trip_count_std_12h'] = data['trip_count'].rolling(window=12).std()
            data['trip_count_std_24h'] = data['trip_count'].rolling(window=24).std()
            
            # Add diff features (rate of change)
            data['trip_count_diff_1'] = data['trip_count'].diff()
            data['trip_count_diff_2'] = data['trip_count'].diff(2)   # Added 2-hour diff
            data['trip_count_diff_3'] = data['trip_count'].diff(3)
            data['trip_count_diff_6'] = data['trip_count'].diff(6)
            data['trip_count_diff_12'] = data['trip_count'].diff(12)
            data['trip_count_diff_24'] = data['trip_count'].diff(24)
        
        # 2. Create holiday indicators
        if 'datetime' in data.columns:
            print("Adding holiday and time-based features...")
            data['date'] = data['datetime'].dt.date
            data['is_holiday'] = data['date'].apply(lambda x: 1 if x in self.us_holidays else 0)
            
            # 3. Enhanced time-based features
            # Extract more granular time features
            data['hour_of_day'] = data['datetime'].dt.hour
            data['is_rush_hour_am'] = ((data['hour_of_day'] >= 7) & (data['hour_of_day'] <= 9)).astype(int)
            data['is_rush_hour_pm'] = ((data['hour_of_day'] >= 16) & (data['hour_of_day'] <= 19)).astype(int)
            data['is_business_hours'] = ((data['hour_of_day'] >= 9) & (data['hour_of_day'] <= 17)).astype(int)
            data['is_night'] = ((data['hour_of_day'] >= 22) | (data['hour_of_day'] <= 5)).astype(int)
            
            # Day type features (more granular than just weekend/weekday)
            data['day_of_week'] = data['datetime'].dt.dayofweek
            data['is_weekday'] = (data['day_of_week'] < 5).astype(int)
            data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
            data['is_monday'] = (data['day_of_week'] == 0).astype(int)
            data['is_friday'] = (data['day_of_week'] == 4).astype(int)
            
            # Month and season features
            data['month'] = data['datetime'].dt.month
            data['day'] = data['datetime'].dt.day
            data['is_summer'] = ((data['month'] >= 6) & (data['month'] <= 8)).astype(int)
            data['is_winter'] = ((data['month'] == 12) | (data['month'] <= 2)).astype(int)
            data['is_spring'] = ((data['month'] >= 3) & (data['month'] <= 5)).astype(int)
            data['is_fall'] = ((data['month'] >= 9) & (data['month'] <= 11)).astype(int)
            
            # Day of month features
            data['is_start_of_month'] = (data['day'] <= 5).astype(int)
            data['is_end_of_month'] = (data['day'] >= 25).astype(int)
            
            # Add features for weekday/hour combinations
            for day in range(7):
                for hour in [8, 12, 17, 20]:  # Key hours of the day
                    data[f'is_day{day}_hour{hour}'] = ((data['day_of_week'] == day) & 
                                                      (data['hour_of_day'] == hour)).astype(int)
        
        # 4. Weather interaction features
        if all(col in data.columns for col in ['temp', 'is_weekend']):
            # Interaction between temperature and weekend
            data['temp_weekend_interaction'] = data['temp'] * data['is_weekend']
            
        if all(col in data.columns for col in ['rain_1h', 'is_rush_hour_am']):
            # Interaction between rain and rush hour
            data['rain_rush_hour_interaction'] = data['rain_1h'] * data['is_rush_hour_am']
        
        # 5. Cyclical encoding of time features
        if 'hour_of_day' in data.columns:
            data['hour_sin'] = np.sin(2 * np.pi * data['hour_of_day']/24)
            data['hour_cos'] = np.cos(2 * np.pi * data['hour_of_day']/24)
        elif 'hour' in data.columns:
            data['hour_sin'] = np.sin(2 * np.pi * data['hour']/24)
            data['hour_cos'] = np.cos(2 * np.pi * data['hour']/24)
        
        if 'day' in data.columns:
            data['day_sin'] = np.sin(2 * np.pi * data['day']/31)
            data['day_cos'] = np.cos(2 * np.pi * data['day']/31)
        
        if 'month' in data.columns:
            data['month_sin'] = np.sin(2 * np.pi * data['month']/12)
            data['month_cos'] = np.cos(2 * np.pi * data['month']/12)
        
        if 'day_of_week' in data.columns:
            data['day_of_week_sin'] = np.sin(2 * np.pi * data['day_of_week']/7)
            data['day_of_week_cos'] = np.cos(2 * np.pi * data['day_of_week']/7)
        
        # Handle one-hot encoding for weather categories
        if 'weather_main' in data.columns:
            # Fill missing values with most common category
            data['weather_main'] = data['weather_main'].fillna(data['weather_main'].mode()[0])
            
            # Store the weather categories from training data if this is the first call
            if not hasattr(self, 'weather_categories') or self.weather_categories is None:
                self.weather_categories = data['weather_main'].unique().tolist()
                print(f"Weather categories found: {self.weather_categories}")
            
            # Create one-hot encoding for all known weather categories
            for category in self.weather_categories:
                col_name = f'weather_{category}'
                data[col_name] = (data['weather_main'] == category).astype(int)
            
            # If this is prediction data, ensure we have columns for all training categories
            if hasattr(self, 'weather_categories') and self.weather_categories is not None:
                for category in self.weather_categories:
                    col_name = f'weather_{category}'
                    if col_name not in data.columns:
                        print(f"Adding missing weather category column: {col_name}")
                        data[col_name] = 0
        else:
            # If the weather_main column is missing but we know the categories from training
            if hasattr(self, 'weather_categories') and self.weather_categories is not None:
                print("weather_main column not found. Adding placeholder weather category columns.")
                for category in self.weather_categories:
                    col_name = f'weather_{category}'
                    data[col_name] = 0
        
        # Drop unnecessary columns
        cols_to_drop = ['weather_main', 'weather_description', 'weather_icon', 
                        'sunrise', 'sunset', 'timezone', 'datetime', 'date']
        data = data.drop([col for col in cols_to_drop if col in data.columns], axis=1)
        
        # Final check for any remaining NaN values
        remaining_nan = data.isna().sum().sum()
        if remaining_nan > 0:
            print(f"WARNING: {remaining_nan} NaN values remain after preprocessing")
            print(data.isna().sum()[data.isna().sum() > 0])
            
            # Fill remaining NaNs with appropriate values
            data = data.fillna(data.median())
            print(f"Filled remaining NaNs with median values. New NaN count: {data.isna().sum().sum()}")
        
        return data
    
    def create_sequences(self, X, y):
        """
        Modified to create sequences targeting 2 hours ahead
        """
        # Check for NaN values
        if np.isnan(X).any():
            print(f"WARNING: X contains {np.isnan(X).sum()} NaN values before sequence creation")
            # Replace with column means as a last resort
            col_means = np.nanmean(X, axis=0)
            # Replace NaN with column means
            X = np.where(np.isnan(X), np.tile(col_means, (X.shape[0], 1)), X)
            
        if np.isnan(y).any():
            print(f"WARNING: y contains {np.isnan(y).sum()} NaN values before sequence creation")
            # Replace with mean as a last resort
            y_mean = np.nanmean(y)
            y = np.where(np.isnan(y), y_mean, y)
            
        Xs, ys = [], []
        
        # Modified to target the value 2 hours ahead
        for i in range(len(X) - self.seq_length - self.prediction_horizon):
            seq_x = X[i:(i + self.seq_length)]
            # Target the value 2 hours ahead
            target_y = y[i + self.seq_length + self.prediction_horizon - 1]
            
            # Only add sequence if it contains no NaN values
            if not np.isnan(seq_x).any() and not np.isnan(target_y):
                Xs.append(seq_x)
                ys.append(target_y)
        
        if len(Xs) == 0:
            raise ValueError("No valid sequences could be created due to NaN values")
            
        return np.array(Xs), np.array(ys)
    
    def build_model(self, input_shape):
        # Define input layer
        input_layer = Input(shape=input_shape)
        
        # Bidirectional LSTM layers with regularization
        lstm1 = Bidirectional(LSTM(128, 
                               activation='relu', 
                               return_sequences=True,
                               kernel_initializer='he_normal',
                               recurrent_initializer='orthogonal',
                               kernel_regularizer=self.regularizer,
                               recurrent_regularizer=self.regularizer))(input_layer)
        
        batch_norm1 = BatchNormalization()(lstm1)
        dropout1 = Dropout(self.dropout_rate)(batch_norm1)
        
        lstm2 = Bidirectional(LSTM(64, 
                               activation='relu', 
                               return_sequences=True,
                               kernel_initializer='he_normal',
                               recurrent_initializer='orthogonal',
                               kernel_regularizer=self.regularizer,
                               recurrent_regularizer=self.regularizer))(dropout1)
        
        batch_norm2 = BatchNormalization()(lstm2)
        dropout2 = Dropout(self.dropout_rate)(batch_norm2)
        
        # Simplified attention mechanism using only Keras layers
        # Compute attention weights
        attention = Dense(1, activation='tanh')(dropout2)
        attention = Flatten()(attention)
        attention = Dense(input_shape[0], activation='softmax')(attention)
        attention = RepeatVector(lstm2.shape[2])(attention)
        attention = Permute([2, 1])(attention)
        
        # Apply attention weights
        weighted = Concatenate()([dropout2, attention])
        weighted = Dense(lstm2.shape[2], activation='relu')(weighted)
        
        # Global pooling
        context_vector = GlobalAveragePooling1D()(weighted)
        
        # Dense layers with regularization
        dense1 = Dense(32, activation='relu', kernel_initializer='he_normal', kernel_regularizer=self.regularizer)(context_vector)
        batch_norm3 = BatchNormalization()(dense1)
        dropout3 = Dropout(self.dropout_rate * 0.75)(batch_norm3)  # Slightly lower dropout
        
        # Output layer
        output = Dense(1, activation='linear')(dropout3)
        
        # Create model
        model = Model(inputs=input_layer, outputs=output)
        
        # Compile with selected optimizer and loss
        model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=['mae']
        )
        
        return model
    
    def get_param_subtitle(self):
        """Create a subtitle with current parameters for plots"""
        return f"Params: Batch={self.batch_size}, {self.optimizer_type.capitalize()} LR={self.initial_lr}, {self.reg_type.upper()}={self.reg_value}, Dropout={self.dropout_rate}, Loss={self.loss_type}"
    
    def fit(self, df, test_size=0.2, val_size=0.25, epochs=50, batch_size=None):
        # Update batch size if provided
        if batch_size is not None:
            self.batch_size = batch_size
            
        # Preprocess data
        processed_data = self.preprocess_data(df)
        
        # Check if trip_count exists
        if 'trip_count' not in processed_data.columns:
            raise ValueError("Dataset must contain 'trip_count' column")
        
        # Identify weather categories if present in the data
        if 'weather_main' in df.columns:
            self.weather_categories = df['weather_main'].dropna().unique().tolist()
            print(f"Found weather categories: {self.weather_categories}")
            # Ensure we have weather_* columns for each category
            for category in self.weather_categories:
                col_name = f'weather_{category}'
                if col_name not in processed_data.columns:
                    print(f"Adding missing weather category column: {col_name}")
                    processed_data[col_name] = 0
        
        # Store feature names
        self.feature_names = [col for col in processed_data.columns if col != 'trip_count']
        
        if self.ensemble:
            # Train separate models for weekdays and weekends
            return self.fit_ensemble(processed_data, test_size, val_size, epochs, self.batch_size)
        
        # Split features and target
        X = processed_data.drop('trip_count', axis=1)
        y = processed_data['trip_count']
        
        # Print statistics before scaling
        print("\nData statistics before scaling:")
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        print(f"X contains NaN: {X.isna().any().any()}")
        print(f"y contains NaN: {y.isna().any()}")
        
        # Scale features and target
        try:
            X_scaled = self.scaler_X.fit_transform(X)
            y_scaled = self.scaler_y.fit_transform(y.values.reshape(-1, 1))
            
            # Check for NaN values after scaling
            if np.isnan(X_scaled).any():
                print("WARNING: NaN values detected after scaling X. Replacing with 0.")
                X_scaled = np.nan_to_num(X_scaled)
                
            if np.isnan(y_scaled).any():
                print("WARNING: NaN values detected after scaling y. Replacing with 0.")
                y_scaled = np.nan_to_num(y_scaled)
                
        except Exception as e:
            print(f"Error during scaling: {e}")
            # Try to identify problematic columns
            for col in X.columns:
                if X[col].isna().any():
                    print(f"Column {col} has {X[col].isna().sum()} NaN values")
            raise
        
        # Create sequences
        try:
            X_seq, y_seq = self.create_sequences(X_scaled, y_scaled)
            print(f"Created sequences: X_seq shape: {X_seq.shape}, y_seq shape: {y_seq.shape}")
        except Exception as e:
            print(f"Error creating sequences: {e}")
            raise
        
        # Split into train, validation, and test sets
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X_seq, y_seq, test_size=test_size, shuffle=False)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size, shuffle=False)
        
        # Store train and test data for later use
        self.X_train = X_train
        self.X_test = X_test
        self.y_test = y_test
        
        # Final check for NaN values
        for name, data in [("X_train", X_train), ("y_train", y_train), 
                          ("X_val", X_val), ("y_val", y_val),
                          ("X_test", X_test), ("y_test", y_test)]:
            if np.isnan(data).any():
                print(f"WARNING: {name} contains {np.isnan(data).sum()} NaN values")
                raise ValueError(f"NaN values in {name}")
        
        # Build model
        input_shape = (X_train.shape[1], X_train.shape[2])
        self.model = self.build_model(input_shape)
        
        # Early stopping with more patience
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        # Model checkpoint
        checkpoint = ModelCheckpoint(
            './best_2hour_prediction_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        
        # Learning rate reduction on plateau
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        )
        
        callbacks = [early_stopping, checkpoint, reduce_lr]
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate on test data
        test_loss, test_mae = self.model.evaluate(X_test, y_test)
        print(f"Test Loss ({self.loss_type.upper()}): {test_loss}")
        print(f"Test MAE: {test_mae}")
        
        # Make predictions on test set
        y_pred_scaled = self.model.predict(X_test)
        
        # Check for NaNs in predictions
        if np.isnan(y_pred_scaled).any():
            print(f"WARNING: Predictions contain {np.isnan(y_pred_scaled).sum()} NaN values")
            # Replace NaNs with 0 as a last resort
            y_pred_scaled = np.nan_to_num(y_pred_scaled)
        
        # Inverse transform
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        y_true = self.scaler_y.inverse_transform(y_test.reshape(-1, 1))
        
        # Calculate metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        print(f"Mean Absolute Error: {mae:.2f}")
        print(f"Root Mean Squared Error: {rmse:.2f}")
        print(f"R² Score: {r2:.4f}")
        
        # Plot results
        self.plot_training_history(history)
        self.plot_predictions(y_true, y_pred)
        
        # Only analyze feature importance if model trained successfully and not skipped
        if not self.skip_feature_importance:
            try:
                self.analyze_feature_importance(X_train, self.feature_names)
            except Exception as e:
                print(f"Skipping feature importance analysis due to error: {e}")
        else:
            print("Skipping feature importance analysis as requested")
        
        return history
    
    def fit_ensemble(self, processed_data, test_size=0.2, val_size=0.25, epochs=50, batch_size=32):
        """Train separate models for weekdays and weekends"""
        print("Training ensemble of models...")
        
        # Assume is_weekday and is_weekend columns exist from preprocessing
        if 'is_weekday' not in processed_data.columns or 'is_weekend' not in processed_data.columns:
            raise ValueError("Ensemble requires 'is_weekday' and 'is_weekend' columns")
        
        # Split data into weekday and weekend
        weekday_data = processed_data[processed_data['is_weekday'] == 1].copy()
        weekend_data = processed_data[processed_data['is_weekend'] == 1].copy()
        
        print(f"Weekday data shape: {weekday_data.shape}")
        print(f"Weekend data shape: {weekend_data.shape}")
        
        # Verify we have enough data for both subsets
        if len(weekday_data) < self.seq_length * 10 or len(weekend_data) < self.seq_length * 10:
            print("Warning: Not enough data for ensemble model. Falling back to single model.")
            return self.fit(processed_data, test_size, val_size, epochs, batch_size)
        
        # Initialize models dictionary if it doesn't exist
        if not hasattr(self, 'models') or self.models is None:
            self.models = {}
        
        # Train weekday model
        print("\n===== Training WEEKDAY Model =====")
        self.models['weekday'] = TripPrediction2Hour(
            seq_length=self.seq_length, 
            ensemble=False,
            lr=self.initial_lr,
            batch_size=batch_size,
            dropout_rate=self.dropout_rate,
            reg_type=self.reg_type,
            reg_value=self.reg_value,
            optimizer_type=self.optimizer_type,
            loss_type=self.loss_type,
            skip_feature_importance=self.skip_feature_importance
        )
        # Don't add lags again since they're already added
        weekday_history = self.models['weekday'].fit(
            weekday_data, 
            test_size=test_size, 
            val_size=val_size, 
            epochs=epochs, 
            batch_size=batch_size
        )
        
        # Train weekend model
        print("\n===== Training WEEKEND Model =====")
        self.models['weekend'] = TripPrediction2Hour(
            seq_length=self.seq_length, 
            ensemble=False,
            lr=self.initial_lr,
            batch_size=batch_size,
            dropout_rate=self.dropout_rate,
            reg_type=self.reg_type,
            reg_value=self.reg_value,
            optimizer_type=self.optimizer_type,
            loss_type=self.loss_type,
            skip_feature_importance=self.skip_feature_importance
        )
        weekend_history = self.models['weekend'].fit(
            weekend_data, 
            test_size=test_size, 
            val_size=val_size, 
            epochs=epochs, 
            batch_size=batch_size
        )
        
        # Evaluate the ensemble model on a test set
        print("\n===== Evaluating Ensemble Performance =====")
        # Combine weekday and weekend test sets
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        import numpy as np
        
        try:
            # Calculate the number of test samples for each day type
            weekday_test_samples = int(len(weekday_data) * test_size)
            weekend_test_samples = int(len(weekend_data) * test_size)
            
            print(f"Test set: {weekday_test_samples + weekend_test_samples} samples "
                  f"({weekday_test_samples} weekday, {weekend_test_samples} weekend)")
            
            # Get predictions from each model for its respective data
            weekday_y_true = []
            weekday_y_pred = []
            weekend_y_true = []
            weekend_y_pred = []
            
            # Get predictions for weekday test data
            if hasattr(self.models['weekday'], 'X_test') and hasattr(self.models['weekday'], 'y_test'):
                weekday_preds = self.models['weekday'].model.predict(self.models['weekday'].X_test)
                weekday_y_true = self.models['weekday'].scaler_y.inverse_transform(
                    self.models['weekday'].y_test.reshape(-1, 1))
                weekday_y_pred = self.models['weekday'].scaler_y.inverse_transform(weekday_preds)
            
            # Get predictions for weekend test data
            if hasattr(self.models['weekend'], 'X_test') and hasattr(self.models['weekend'], 'y_test'):
                weekend_preds = self.models['weekend'].model.predict(self.models['weekend'].X_test)
                weekend_y_true = self.models['weekend'].scaler_y.inverse_transform(
                    self.models['weekend'].y_test.reshape(-1, 1))
                weekend_y_pred = self.models['weekend'].scaler_y.inverse_transform(weekend_preds)
            
            # Combine predictions
            y_true = np.vstack([weekday_y_true, weekend_y_true]) if len(weekday_y_true) > 0 and len(weekend_y_true) > 0 else \
                     weekday_y_true if len(weekday_y_true) > 0 else weekend_y_true
            y_pred = np.vstack([weekday_y_pred, weekend_y_pred]) if len(weekday_y_pred) > 0 and len(weekend_y_pred) > 0 else \
                     weekday_y_pred if len(weekday_y_pred) > 0 else weekend_y_pred
            
            if len(y_true) > 0 and len(y_pred) > 0:
                # Calculate metrics
                mae = mean_absolute_error(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                r2 = r2_score(y_true, y_pred)
                
                print(f"Ensemble Model Performance:")
                print(f"Mean Absolute Error: {mae:.2f}")
                print(f"Root Mean Squared Error: {rmse:.2f}")
                print(f"R² Score: {r2:.4f}")
                
                # Store metrics for the ensemble
                self.ensemble_metrics = {
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2
                }
                
                # Plot combined predictions
                self.plot_ensemble_predictions(y_true, y_pred)
            else:
                print("Warning: Could not evaluate ensemble performance - no test data available")
        
        except Exception as e:
            print(f"Error evaluating ensemble performance: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n===== Ensemble Training Complete =====")
        
        # Save X_test and y_test for weekday and weekend models in the parent model for later use
        self.X_test_weekday = getattr(self.models['weekday'], 'X_test', None)
        self.y_test_weekday = getattr(self.models['weekday'], 'y_test', None)
        self.X_test_weekend = getattr(self.models['weekend'], 'X_test', None)
        self.y_test_weekend = getattr(self.models['weekend'], 'y_test', None)
        
        # Return combined history (just for reference)
        return {
            'weekday': weekday_history,
            'weekend': weekend_history
        }
        
    def plot_ensemble_predictions(self, y_true, y_pred, n_samples=200):
        """Plot the combined predictions from the ensemble model"""
        # Only plot if we have data
        if len(y_true) == 0 or len(y_pred) == 0:
            print("Warning: No data available for ensemble prediction plot")
            return
            
        # Make sure we don't try to plot more samples than we have
        n_samples = min(n_samples, len(y_true))
        
        # Time series plot of actual vs predicted values
        plt.figure(figsize=(15, 6))
        plt.plot(y_true[:n_samples], 'b-', label='Actual')
        plt.plot(y_pred[:n_samples], 'orange', label='Predicted')
        plt.title(f'Ensemble Model: Actual vs Predicted Trip Count (First {n_samples} samples)\n{self.get_param_subtitle()}')
        plt.xlabel('Sample Index')
        plt.ylabel('Trip Count')
        plt.legend()
        # Add parameters to filename
        filename = f'./ensemble_prediction_results_{self.optimizer_type}_{self.reg_type}_{self.dropout_rate}_{self.loss_type}.png'
        plt.savefig(filename)
        plt.show()
        
        # Residual plot
        plt.figure(figsize=(15, 6))
        residuals = y_true[:n_samples].flatten() - y_pred[:n_samples].flatten()
        plt.scatter(y_pred[:n_samples], residuals, color='#1f77b4')  # Use default blue color
        plt.axhline(y=0, color='r', linestyle='-')
        plt.title(f'Ensemble Model: Residual Plot\n{self.get_param_subtitle()}')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        # Add parameters to filename
        filename = f'./ensemble_residual_plot_{self.optimizer_type}_{self.reg_type}_{self.dropout_rate}_{self.loss_type}.png'
        plt.savefig(filename)
        plt.show()
    
    def plot_training_history(self, history):
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'Loss\n{self.get_param_subtitle()}')
        plt.xlabel('Epochs')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title(f'Mean Absolute Error\n{self.get_param_subtitle()}')
        plt.xlabel('Epochs')
        plt.legend()
        
        plt.tight_layout()
        # Add parameters to filename
        filename = f'./training_history_2h_{self.optimizer_type}_{self.reg_type}_{self.dropout_rate}_{self.loss_type}.png'
        plt.savefig(filename)
        plt.show()
    
    def plot_predictions(self, y_true, y_pred, n_samples=200):
        """Plot the actual vs predicted trip counts and residuals"""
        # Time series plot of actual vs predicted values
        plt.figure(figsize=(15, 6))
        plt.plot(y_true[:n_samples], 'b-', label='Actual')
        plt.plot(y_pred[:n_samples], 'orange', label='Predicted')
        plt.title(f'Trip Count: Actual vs Predicted (First {n_samples} hours)\n{self.get_param_subtitle()}')
        plt.xlabel('Hours')
        plt.ylabel('Trip Count')
        plt.legend()
        # Add parameters to filename
        filename = f'./prediction_results_2h_{self.optimizer_type}_{self.reg_type}_{self.dropout_rate}_{self.loss_type}.png'
        plt.savefig(filename)
        plt.show()
        
        # Residual plot
        plt.figure(figsize=(15, 6))
        residuals = y_true[:n_samples].flatten() - y_pred[:n_samples].flatten()
        plt.scatter(y_pred[:n_samples], residuals, color='#1f77b4')  # Use default blue color
        plt.axhline(y=0, color='r', linestyle='-')
        plt.title(f'Residual Plot\n{self.get_param_subtitle()}')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        # Add parameters to filename
        filename = f'./residual_plot_2h_{self.optimizer_type}_{self.reg_type}_{self.dropout_rate}_{self.loss_type}.png'
        plt.savefig(filename)
        plt.show()
    
    def analyze_feature_importance(self, X_train, feature_names, n_top=15):
        """Analyze feature importance using the permutation method"""
        print("Analyzing feature importance for 2-hour prediction... (this may take some time)")
        baseline_preds = self.model.predict(X_train)
        importances = []
        
        for i in range(X_train.shape[2]):
            X_perturbed = X_train.copy()
            X_perturbed[:, :, i] = np.random.permutation(X_perturbed[:, :, i])
            
            perturbed_preds = self.model.predict(X_perturbed)
            importance = np.mean(np.abs(baseline_preds - perturbed_preds))
            importances.append(importance)
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        # Plot feature importance as horizontal bars
        plt.figure(figsize=(15, 10))
        top_features = importance_df.head(n_top)
        sns.barplot(data=top_features, x='Importance', y='Feature')
        plt.title(f'Top {n_top} Feature Importances\n{self.get_param_subtitle()}')
        plt.tight_layout()
        # Add parameters to filename
        filename = f'./feature_importance_2h_{self.optimizer_type}_{self.reg_type}_{self.dropout_rate}_{self.loss_type}.png'
        plt.savefig(filename)
        plt.show()
        
        print(f"Top 10 most important features for 2-hour prediction:")
        print(importance_df.head(10))
        
        return importance_df
    
    def predict(self, new_data):
        """
        Make predictions for 2 hours ahead using new data
        
        Args:
            new_data: DataFrame with same format as training data
                     Must have enough historical data (at least seq_length)
        
        Returns:
            Predicted trip count 2 hours in the future
        """
        # Check if new_data exists
        if new_data is None or len(new_data) == 0:
            raise ValueError("No data provided for prediction")
            
        # If using ensemble models
        if self.ensemble and self.models:
            # Determine if weekday or weekend for the prediction point (2 hours ahead)
            if 'datetime' in new_data.columns:
                # Convert to datetime if it's not already
                if not pd.api.types.is_datetime64_any_dtype(new_data['datetime']):
                    new_data['datetime'] = pd.to_datetime(new_data['datetime'], errors='coerce')
                
                # Get the last timestamp
                last_time = new_data['datetime'].iloc[-1]
                
                # Calculate the time 2 hours ahead
                future_time = last_time + timedelta(hours=2)
                
                # Determine if the future time is a weekend
                is_weekend = future_time.dayofweek >= 5
                
                if is_weekend:
                    print("Using weekend model for prediction (future time is on a weekend)")
                    return self.models['weekend'].predict(new_data)
                else:
                    print("Using weekday model for prediction (future time is on a weekday)")
                    return self.models['weekday'].predict(new_data)
            else:
                # If no datetime column, use the last day_of_week value
                # (Since 2 hours is not enough to change the day in most cases)
                if 'day_of_week' in new_data.columns:
                    last_day = new_data['day_of_week'].iloc[-1]
                    is_weekend = last_day >= 5
                    
                    if is_weekend:
                        print("Using weekend model for prediction (based on day_of_week)")
                        return self.models['weekend'].predict(new_data)
                    else:
                        print("Using weekday model for prediction (based on day_of_week)")
                        return self.models['weekday'].predict(new_data)
                else:
                    # Fall back to using weekday model
                    print("Warning: Cannot determine if weekend or weekday for future prediction. Using weekday model.")
                    return self.models['weekday'].predict(new_data)
        
        # Single model prediction
        # Preprocess data
        processed_data = self.preprocess_data(new_data)
        
        # Make sure 'trip_count' is dropped before scaling if it exists
        if 'trip_count' in processed_data.columns:
            print("Dropping 'trip_count' column before prediction")
            processed_data = processed_data.drop('trip_count', axis=1)
        
        # Check for feature consistency with training data
        if hasattr(self, 'feature_names') and self.feature_names is not None:
            missing_features = [f for f in self.feature_names if f not in processed_data.columns]
            unexpected_features = [f for f in processed_data.columns if f not in self.feature_names]
            
            if missing_features:
                print(f"Warning: Missing features in prediction data: {missing_features}")
                # Add missing columns with zeros
                for col in missing_features:
                    processed_data[col] = 0
                    
            if unexpected_features:
                print(f"Warning: Unexpected features in prediction data: {unexpected_features}")
                # Drop unexpected columns
                processed_data = processed_data.drop(unexpected_features, axis=1)
            
            # Ensure columns are in the exact same order as during training
            processed_data = processed_data[self.feature_names]
        
        # Scale data
        try:
            X_scaled = self.scaler_X.transform(processed_data)
        except ValueError as e:
            print(f"Error during scaling: {e}")
            print(f"Processed data shape: {processed_data.shape}, columns: {processed_data.columns}")
            print(f"Trained on feature names: {self.feature_names}")
            raise
        
        # Check for NaN values after scaling
        if np.isnan(X_scaled).any():
            print("WARNING: NaN values detected in prediction data. Replacing with 0.")
            X_scaled = np.nan_to_num(X_scaled)
        
        # Check if we have enough data for a sequence
        if len(X_scaled) < self.seq_length:
            raise ValueError(f"Not enough data for a sequence. Need at least {self.seq_length} rows.")
        
        # Create sequence
        sequence = X_scaled[-self.seq_length:].reshape(1, self.seq_length, X_scaled.shape[1])
        
        # Make prediction
        prediction_scaled = self.model.predict(sequence)
        
        # Check for NaN in prediction
        if np.isnan(prediction_scaled).any():
            print("WARNING: NaN values in prediction. Using 0 instead.")
            prediction_scaled = np.nan_to_num(prediction_scaled)
            
        prediction = self.scaler_y.inverse_transform(prediction_scaled)
        
        return prediction[0][0]
    
    def save(self, filepath):
        """Save the model(s) and scalers."""
        # For ensemble models
        if self.ensemble and self.models:
            # Print debug info about models
            print(f"Saving ensemble model. Available model keys: {list(self.models.keys())}")
            
            try:
                # Save each sub-model separately
                for model_type, model_obj in self.models.items():
                    # Check if this is a model instance or just a TripPrediction2Hour instance
                    if hasattr(model_obj, 'model') and model_obj.model is not None:
                        # It's a TripPrediction2Hour instance with model
                        # Make sure filepath has the correct extension
                        model_path = f"{filepath}_{model_type}.keras"
                        print(f"Saving sub-model for {model_type} to {model_path}")
                        model_obj.model.save(model_path)
                        
                        # Save the scalers specific to this sub-model
                        with open(f"{filepath}_{model_type}_scalers.pkl", "wb") as f:
                            import pickle
                            pickle.dump({
                                'scaler_X': model_obj.scaler_X,
                                'scaler_y': model_obj.scaler_y,
                                'feature_names': model_obj.feature_names,
                                'seq_length': model_obj.seq_length,
                                'prediction_horizon': model_obj.prediction_horizon,
                                'batch_size': model_obj.batch_size,
                                'dropout_rate': model_obj.dropout_rate,
                                'reg_type': model_obj.reg_type,
                                'reg_value': model_obj.reg_value,
                                'optimizer_type': model_obj.optimizer_type,
                                'loss_type': model_obj.loss_type,
                                'skip_feature_importance': model_obj.skip_feature_importance
                            }, f)
                    else:
                        print(f"Warning: Sub-model for {model_type} doesn't have a valid model attribute")
                
                # Save ensemble configuration
                import pickle
                with open(f"{filepath}_ensemble_config.pkl", "wb") as f:
                    pickle.dump({
                        'ensemble': True,
                        'seq_length': self.seq_length,
                        'prediction_horizon': self.prediction_horizon,
                        'initial_lr': self.initial_lr,
                        'batch_size': self.batch_size,
                        'dropout_rate': self.dropout_rate,
                        'reg_type': self.reg_type,
                        'reg_value': self.reg_value,
                        'optimizer_type': self.optimizer_type,
                        'loss_type': self.loss_type,
                        'skip_feature_importance': self.skip_feature_importance
                    }, f)
                
                print(f"Ensemble model configuration saved to {filepath}_ensemble_config.pkl")
                
            except Exception as e:
                print(f"Error saving ensemble model: {e}")
                import traceback
                traceback.print_exc()
                print("Attempting to save main model as fallback...")
                # If ensemble saving fails, try to save the main model as fallback
                if self.model is not None:
                    self.model.save(filepath + "_fallback.keras")
                    print(f"Main model saved to {filepath}_fallback.keras as fallback")
            
            return
        
        # Save single model
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
            
        # Make sure filepath has the correct extension
        if not (filepath.endswith('.keras') or filepath.endswith('.h5')):
            filepath = filepath + '.keras'
            
        # Save the model
        self.model.save(filepath)
        
        # Save the scalers and parameters
        import pickle
        with open(f"{filepath.replace('.keras','').replace('.h5','')}_scalers.pkl", "wb") as f:
            pickle.dump({
                'scaler_X': self.scaler_X,
                'scaler_y': self.scaler_y,
                'seq_length': self.seq_length,
                'prediction_horizon': self.prediction_horizon,
                'feature_names': self.feature_names,
                'ensemble': False,
                'batch_size': self.batch_size,
                'dropout_rate': self.dropout_rate,
                'reg_type': self.reg_type,
                'reg_value': self.reg_value,
                'optimizer_type': self.optimizer_type,
                'loss_type': self.loss_type,
                'skip_feature_importance': self.skip_feature_importance
            }, f)
        
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Load the model(s) and scalers."""
        import pickle
        import os
        
        # Check if this is an ensemble model
        if os.path.exists(f"{filepath}_ensemble_config.pkl"):
            with open(f"{filepath}_ensemble_config.pkl", "rb") as f:
                config = pickle.load(f)
            
            # Create instance with ensemble flag and all parameters
            instance = cls(
                seq_length=config['seq_length'], 
                ensemble=True,
                lr=config.get('initial_lr', 0.0005),
                batch_size=config.get('batch_size', 32),
                dropout_rate=config.get('dropout_rate', 0.4),
                reg_type=config.get('reg_type', 'l2'),
                reg_value=config.get('reg_value', 0.001),
                optimizer_type=config.get('optimizer_type', 'adam'),
                loss_type=config.get('loss_type', 'mse'),
                skip_feature_importance=config.get('skip_feature_importance', False)
            )
            instance.prediction_horizon = config.get('prediction_horizon', 2)
            
            # Load each model
            instance.models = {
                'weekday': cls.load(f"{filepath}_weekday"),
                'weekend': cls.load(f"{filepath}_weekend")
            }
            
            print(f"Loaded ensemble models from {filepath}_weekday and {filepath}_weekend")
            return instance
        
        # Load single model
        model = load_model(filepath, custom_objects={'smape_loss': smape_loss})
        
        # Load scalers and parameters
        scaler_path = f"{filepath.replace('.keras','').replace('.h5','')}_scalers.pkl"
        with open(scaler_path, "rb") as f:
            saved_data = pickle.load(f)
        
        # Create instance and set attributes
        instance = cls(
            seq_length=saved_data['seq_length'],
            ensemble=saved_data.get('ensemble', False),
            lr=saved_data.get('initial_lr', 0.0005),
            batch_size=saved_data.get('batch_size', 32),
            dropout_rate=saved_data.get('dropout_rate', 0.4),
            reg_type=saved_data.get('reg_type', 'l2'),
            reg_value=saved_data.get('reg_value', 0.001),
            optimizer_type=saved_data.get('optimizer_type', 'adam'),
            loss_type=saved_data.get('loss_type', 'mse'),
            skip_feature_importance=saved_data.get('skip_feature_importance', False)
        )
        instance.prediction_horizon = saved_data.get('prediction_horizon', 2)
        instance.model = model
        instance.scaler_X = saved_data['scaler_X']
        instance.scaler_y = saved_data['scaler_y']
        instance.feature_names = saved_data['feature_names']
        
        print(f"Model loaded from {filepath}")
        return instance


# Function to analyze feature importance for 2-hour predictions
def analyze_features_for_2h_prediction(model, df):
    """
    Perform comprehensive feature importance analysis for 2-hour predictions
    """
    # Skip if feature importance analysis is turned off
    if model.skip_feature_importance:
        print("Skipping feature importance analysis as requested")
        return None, None
        
    # Ensure we have a model with feature importance data
    if model.model is None or model.feature_names is None:
        raise ValueError("Model must be trained first")
    
    # Preprocess data for analysis
    processed_data = model.preprocess_data(df)
    X = processed_data.drop('trip_count', axis=1)
    
    # Get feature importances from the model
    importance_df = model.analyze_feature_importance(
        model.X_train, model.feature_names)
    
    # Categorize features
    feature_categories = {
        'Weather': ['temp', 'feels_like', 'humidity', 'rain_1h', 'snow_1h', 'wind_speed', 'clouds'],
        'Time': ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos', 
                'day_of_week_sin', 'day_of_week_cos'],
        'Special Periods': ['is_weekend', 'is_holiday', 'is_rush_hour_am', 'is_rush_hour_pm', 
                           'is_business_hours', 'is_night'],
        'Historical': ['trip_count_lag_1', 'trip_count_lag_2', 'trip_count_lag_3', 
                      'trip_count_ma_2h', 'trip_count_ma_3h', 'trip_count_ma_6h',
                      'trip_count_std_2h', 'trip_count_std_3h', 'trip_count_std_6h',
                      'trip_count_diff_1', 'trip_count_diff_2', 'trip_count_diff_3']
    }
    
    # Calculate importance by category
    category_importance = {}
    for category, features in feature_categories.items():
        # Find all column names that contain these features
        category_cols = []
        for feature in features:
            matching_cols = [col for col in importance_df['Feature'] if feature in col]
            category_cols.extend(matching_cols)
        
        # Calculate total importance for the category
        category_importance[category] = importance_df[
            importance_df['Feature'].isin(category_cols)]['Importance'].sum()
    
    # Create visualizations
    # 1. Category importance pie chart
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.pie(category_importance.values(), labels=category_importance.keys(), 
            autopct='%1.1f%%', startangle=90)
    plt.title(f'Feature Importance by Category (2-hour Prediction)\n{model.get_param_subtitle()}')
    
    # 2. Top features bar chart
    plt.subplot(1, 2, 2)
    top_n = 10
    sns.barplot(data=importance_df.head(top_n), y='Feature', x='Importance')
    plt.title(f'Top {top_n} Most Important Features (2-hour Prediction)\n{model.get_param_subtitle()}')
    plt.tight_layout()
    # Add parameters to filename
    filename = f'./feature_importance_analysis_2h_{model.optimizer_type}_{model.reg_type}_{model.dropout_rate}_{model.loss_type}.png'
    plt.savefig(filename)
    plt.show()
    
    # 3. Correlation heatmap of top features with trip count
    top_features = importance_df.head(15)['Feature'].tolist()
    correlation_data = processed_data[top_features + ['trip_count']]
    plt.figure(figsize=(12, 10))
    correlation_matrix = correlation_data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(f'Correlation Between Top Features and Trip Count (2-hour Prediction)\n{model.get_param_subtitle()}')
    plt.tight_layout()
    # Add parameters to filename
    filename = f'./feature_correlation_heatmap_2h_{model.optimizer_type}_{model.reg_type}_{model.dropout_rate}_{model.loss_type}.png'
    plt.savefig(filename)
    plt.show()
    
    # Print insights
    print("\n=== 2-HOUR PREDICTION FEATURE IMPORTANCE ANALYSIS ===")
    print("\nTop 10 Individual Features:")
    print(importance_df.head(10))
    
    print("\nFeature Importance by Category:")
    for category, importance in sorted(category_importance.items(), 
                                      key=lambda x: x[1], reverse=True):
        print(f"{category}: {importance:.4f} ({importance/sum(category_importance.values())*100:.1f}%)")
    
    return importance_df, category_importance


def run_with_parameters(params, data=None, experiment_name=None):
    """Run the prediction pipeline with given parameters"""
    print(f"\nRunning 2-Hour Prediction with parameters: {params}")
    
    # Load data if not provided
    if data is None:
        print("Loading dataset...")
        df = pd.read_csv('hourly_trips_with_weather.csv')
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    else:
        df = data
        
    # Generate a descriptive model name
    if experiment_name:
        model_name = f"{experiment_name}"
    else:
        is_ensemble = "ensemble" if params.get('ensemble', False) else "single"
        model_name = f"{is_ensemble}_{params['optimizer_type']}_{params['reg_type']}"
        model_name += f"_b{params['batch_size']}_d{params['dropout_rate']}_r{params['reg_value']}"
        
    model_filepath = f'./trip_prediction_2hour_model_{model_name}'
    
    # Print experiment configuration
    print("\n===== EXPERIMENT CONFIGURATION =====")
    print(f"Model type: {'Ensemble' if params.get('ensemble', False) else 'Single'}")
    print(f"Optimizer: {params['optimizer_type']}, Learning rate: {params['lr']}")
    print(f"Regularization: {params['reg_type']} = {params['reg_value']}")
    print(f"Dropout rate: {params['dropout_rate']}")
    print(f"Loss function: {params['loss_type']}")
    print(f"Batch size: {params['batch_size']}")
    print(f"Training epochs: {params['epochs']}")
    print(f"Skip feature importance: {params.get('skip_feature_importance', False)}")
    print(f"Output model path: {model_filepath}")
    print("==================================")
    
    try:
        # Create model with specified parameters
        model_2h = TripPrediction2Hour(
            seq_length=12, 
            ensemble=params.get('ensemble', False),
            lr=params.get('lr', 0.0005),
            batch_size=params.get('batch_size', 32),
            dropout_rate=params.get('dropout_rate', 0.4),
            reg_type=params.get('reg_type', 'l2'),
            reg_value=params.get('reg_value', 0.001),
            optimizer_type=params.get('optimizer_type', 'adam'),
            loss_type=params.get('loss_type', 'mse'),
            skip_feature_importance=params.get('skip_feature_importance', False)
        )
        
        # Train model
        history = model_2h.fit(
            df, 
            epochs=params.get('epochs', 20), 
            batch_size=params.get('batch_size', 32)
        )
        
        # Save the model
        model_2h.save(model_filepath)
        
        # Analyze feature importance if not an ensemble model and not skipped
        # (For ensemble models, this is already done for each sub-model)
        if not params.get('ensemble', False) and not params.get('skip_feature_importance', False):
            analyze_features_for_2h_prediction(model_2h, df)
        
        # Test prediction on last 24 hours
        test_data = df.tail(24).copy()  # Just need the last 24 hours for prediction
        prediction = model_2h.predict(test_data)
        print(f"\nPredicted trip count 2 hours ahead: {prediction:.2f}")
        
        print(f"\nComplete! 2-hour prediction model has been trained and saved to {model_filepath}")
        return model_2h
        
    except Exception as e:
        print(f"Error during model training: {e}")
        import traceback
        traceback.print_exc()
        print("Model training failed. Please check the error message above.")
        return None
    

def interactive_parameter_selection():
    """Allow user to interactively select parameters"""
    print("\n===== Bike Trip Prediction Parameter Selection =====")
    
    # Define parameter options
    optimizer_options = {
        '1': {'name': 'adam', 'description': 'Adam (adaptive learning rate)'},
        '2': {'name': 'rmsprop', 'description': 'RMSprop (good for RNNs)'},
        '3': {'name': 'sgd', 'description': 'SGD with momentum'}
    }
    
    regularizer_options = {
        '1': {'name': 'l2', 'description': 'L2 regularization (weight decay)'},
        '2': {'name': 'l1', 'description': 'L1 regularization (sparse weights)'},
        '3': {'name': 'l1_l2', 'description': 'Combined L1 and L2 regularization'}
    }
    
    loss_options = {
        '1': {'name': 'mse', 'description': 'Mean Squared Error (standard)'},
        '2': {'name': 'mae', 'description': 'Mean Absolute Error (less sensitive to outliers)'},
        '3': {'name': 'huber', 'description': 'Huber Loss (combines MAE and MSE benefits)'},
        '4': {'name': 'smape', 'description': 'Symmetric Mean Absolute Percentage Error (scale-invariant)'}
    }
    
    # Print options
    print("\nOptimizer options:")
    for key, val in optimizer_options.items():
        print(f"{key}. {val['description']}")
        
    opt_choice = input("Select optimizer (default=1): ").strip() or '1'
    optimizer_type = optimizer_options[opt_choice]['name']
    
    print("\nRegularizer options:")
    for key, val in regularizer_options.items():
        print(f"{key}. {val['description']}")
        
    reg_choice = input("Select regularizer (default=1): ").strip() or '1'
    reg_type = regularizer_options[reg_choice]['name']
    
    print("\nLoss function options:")
    for key, val in loss_options.items():
        print(f"{key}. {val['description']}")
        
    loss_choice = input("Select loss function (default=1): ").strip() or '1'
    loss_type = loss_options[loss_choice]['name']
    
    # Get numeric parameters
    batch_size = int(input("\nBatch size (default=32): ").strip() or '32')
    dropout_rate = float(input("Dropout rate (0.0-0.9, default=0.4): ").strip() or '0.4')
    reg_value = float(input("Regularization strength (default=0.001): ").strip() or '0.001')
    learning_rate = float(input("Learning rate (default=0.0005): ").strip() or '0.0005')
    epochs = int(input("Training epochs (default=20): ").strip() or '20')
    
    # Ask about ensemble
    ensemble = input("Use ensemble model for weekday/weekend? (y/n, default=n): ").strip().lower() == 'y'
    
    # Ask about skipping feature importance
    skip_feature_importance = input("Skip feature importance analysis? (y/n, default=n): ").strip().lower() == 'y'
    
    # Return parameters as dictionary
    return {
        'optimizer_type': optimizer_type,
        'reg_type': reg_type,
        'reg_value': reg_value,
        'loss_type': loss_type,
        'batch_size': batch_size,
        'dropout_rate': dropout_rate,
        'lr': learning_rate,
        'epochs': epochs,
        'ensemble': ensemble,
        'skip_feature_importance': skip_feature_importance
    }


# Example parameter configurations
default_params = {
    'optimizer_type': 'adam',
    'reg_type': 'l2',
    'reg_value': 0.001,
    'loss_type': 'mse',
    'batch_size': 32,
    'dropout_rate': 0.4,
    'lr': 0.0005,
    'epochs': 20,
    'ensemble': False,
    'skip_feature_importance': False
}

rmsprop_params = {
    'optimizer_type': 'rmsprop',
    'reg_type': 'l2',
    'reg_value': 0.001,
    'loss_type': 'mse',
    'batch_size': 64,
    'dropout_rate': 0.4,
    'lr': 0.001,
    'epochs': 20,
    'ensemble': False,
    'skip_feature_importance': False
}

l1_params = {
    'optimizer_type': 'adam',
    'reg_type': 'l1',
    'reg_value': 0.0005,
    'loss_type': 'mse', 
    'batch_size': 32,
    'dropout_rate': 0.3,
    'lr': 0.0005,
    'epochs': 20,
    'ensemble': False,
    'skip_feature_importance': False
}

huber_params = {
    'optimizer_type': 'adam',
    'reg_type': 'l2',
    'reg_value': 0.001,
    'loss_type': 'huber',
    'batch_size': 32,
    'dropout_rate': 0.4,
    'lr': 0.0005,
    'epochs': 20,
    'ensemble': False,
    'skip_feature_importance': False
}

smape_params = {
    'optimizer_type': 'adam',
    'reg_type': 'l2',
    'reg_value': 0.001,
    'loss_type': 'smape',
    'batch_size': 32,
    'dropout_rate': 0.4,
    'lr': 0.0005,
    'epochs': 20,
    'ensemble': False,
    'skip_feature_importance': False
}


# Main function for command line usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train bike trip prediction model with custom parameters')
    parser.add_argument('--mode', type=str, default='interactive', 
                        choices=['interactive', 'default', 'rmsprop', 'l1', 'huber', 'smape', 'ensemble'],
                        help='Mode to run: interactive or one of the predefined parameter sets')
    parser.add_argument('--experiment', type=str, default=None,
                        help='Name for the experiment (used in output filenames)')
    parser.add_argument('--ensemble', action='store_true',
                        help='Train separate models for weekday and weekend data')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size parameter')
    parser.add_argument('--dropout', type=float, default=None,
                        help='Override dropout rate parameter')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of training epochs')
    parser.add_argument('--skip_feature_importance', action='store_true',
                        help='Skip the feature importance analysis (saves time)')
    
    args = parser.parse_args()
    
    # Load data once for efficiency
    print("Loading dataset...")
    df = pd.read_csv('hourly_trips_with_weather.csv')
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Get base parameters based on selected mode
    if args.mode == 'interactive':
        params = interactive_parameter_selection()
    elif args.mode == 'default':
        params = default_params
    elif args.mode == 'rmsprop':
        params = rmsprop_params
    elif args.mode == 'l1':
        params = l1_params
    elif args.mode == 'huber':
        params = huber_params
    elif args.mode == 'smape':
        params = smape_params
    elif args.mode == 'ensemble':
        # Predefined ensemble parameters
        params = {
            'optimizer_type': 'adam',
            'reg_type': 'l2',
            'reg_value': 0.001,
            'loss_type': 'mse',
            'batch_size': 64,
            'dropout_rate': 0.4,
            'lr': 0.0005,
            'epochs': 20,
            'ensemble': True,
            'skip_feature_importance': False
        }
    else:
        print("Invalid mode selected. Using default parameters.")
        params = default_params
    
    # Override parameters with command-line arguments if provided
    if args.ensemble:
        params['ensemble'] = True
        
    if args.batch_size is not None:
        params['batch_size'] = args.batch_size
        
    if args.dropout is not None:
        params['dropout_rate'] = args.dropout
        
    if args.epochs is not None:
        params['epochs'] = args.epochs
        
    if args.skip_feature_importance:
        params['skip_feature_importance'] = True
    
    # Run with parameters
    model = run_with_parameters(params, df, experiment_name=args.experiment)
    
    # Summary of results
    if model is not None:
        if hasattr(model, 'ensemble_metrics') and model.ensemble_metrics:
            print("\n===== ENSEMBLE MODEL RESULTS =====")
            print(f"MAE: {model.ensemble_metrics['mae']:.2f}")
            print(f"RMSE: {model.ensemble_metrics['rmse']:.2f}")
            print(f"R²: {model.ensemble_metrics['r2']:.4f}")
        elif hasattr(model, 'model') and model.model is not None:
            print("\n===== MODEL RESULTS =====")
            test_loss = model.model.evaluate(model.X_test, model.y_test)[0]
            print(f"Test Loss: {test_loss:.4f}")
        
        print("\nExperiment completed successfully!")
    else:
        print("\nExperiment failed. Please check the error messages above.")
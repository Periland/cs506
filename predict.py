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
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import os
from datetime import datetime, timedelta

class ImprovedTripPredictionModel:
    def __init__(self, seq_length=12, ensemble=False, lr=0.0005):
        self.seq_length = seq_length
        self.model = None
        self.scaler_X = RobustScaler()  # RobustScaler handles outliers better
        self.scaler_y = MinMaxScaler()
        self.feature_names = None
        self.ensemble = ensemble
        self.models = {}  # For ensemble models
        self.initial_lr = lr
        self.us_holidays = holidays.US()  # US holiday calendar
        self.X_train = None  # Store for feature importance analysis
    
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
            
            # Create lag features for the past 1, 2, 3, 6, 12, and 24 hours
            for lag in [1, 2, 3, 6, 12, 24]:
                data[f'trip_count_lag_{lag}'] = data['trip_count'].shift(lag)
                
            # Calculate rolling statistics (moving averages)
            data['trip_count_ma_3h'] = data['trip_count'].rolling(window=3).mean()
            data['trip_count_ma_6h'] = data['trip_count'].rolling(window=6).mean()
            data['trip_count_ma_12h'] = data['trip_count'].rolling(window=12).mean()
            data['trip_count_ma_24h'] = data['trip_count'].rolling(window=24).mean()
            
            # Add rolling standard deviations (captures trip count volatility)
            data['trip_count_std_24h'] = data['trip_count'].rolling(window=24).std()
            
            # Add diff features (rate of change)
            data['trip_count_diff_1'] = data['trip_count'].diff()
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
        
        # 4. Weather interaction features
        if all(col in data.columns for col in ['temp', 'is_weekend']):
            # Interaction between temperature and weekend
            data['temp_weekend_interaction'] = data['temp'] * data['is_weekend']
            
        if all(col in data.columns for col in ['rain_1h', 'is_rush_hour_am']):
            # Interaction between rain and rush hour
            data['rain_rush_hour_interaction'] = data['rain_1h'] * data['is_rush_hour_am']
        
        # 5. Cyclical encoding of time features (better than the original implementation)
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
        
        # Convert categorical weather_main to one-hot encoding
        if 'weather_main' in data.columns:
            # Fill missing values with most common category
            data['weather_main'] = data['weather_main'].fillna(data['weather_main'].mode()[0])
            weather_dummies = pd.get_dummies(data['weather_main'], prefix='weather')
            data = pd.concat([data, weather_dummies], axis=1)
        
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
        for i in range(len(X) - self.seq_length):
            seq_x = X[i:(i + self.seq_length)]
            # Only add sequence if it contains no NaN values
            if not np.isnan(seq_x).any() and not np.isnan(y[i + self.seq_length]):
                Xs.append(seq_x)
                ys.append(y[i + self.seq_length])
        
        if len(Xs) == 0:
            raise ValueError("No valid sequences could be created due to NaN values")
            
        return np.array(Xs), np.array(ys)
    
    def build_model(self, input_shape):
        # ---------- ENHANCED MODEL ARCHITECTURE ----------
        
        # 1. Define input layer
        input_layer = Input(shape=input_shape)
        
        # 2. Bidirectional LSTM layers with L2 regularization
        lstm1 = Bidirectional(LSTM(128, 
                               activation='relu', 
                               return_sequences=True,
                               kernel_initializer='he_normal',
                               recurrent_initializer='orthogonal',
                               kernel_regularizer=l2(0.001),  # L2 regularization
                               recurrent_regularizer=l2(0.001)))(input_layer)
        
        batch_norm1 = BatchNormalization()(lstm1)
        dropout1 = Dropout(0.4)(batch_norm1)  # Increased dropout rate
        
        lstm2 = Bidirectional(LSTM(64, 
                               activation='relu', 
                               return_sequences=True,
                               kernel_initializer='he_normal',
                               recurrent_initializer='orthogonal',
                               kernel_regularizer=l2(0.001),
                               recurrent_regularizer=l2(0.001)))(dropout1)
        
        batch_norm2 = BatchNormalization()(lstm2)
        dropout2 = Dropout(0.4)(batch_norm2)
        
        # 3. Simplified attention mechanism using only Keras layers
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
        
        # 4. Dense layers with regularization
        dense1 = Dense(32, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(0.001))(context_vector)
        batch_norm3 = BatchNormalization()(dense1)
        dropout3 = Dropout(0.3)(batch_norm3)
        
        # Output layer
        output = Dense(1, activation='linear')(dropout3)
        
        # Create model
        model = Model(inputs=input_layer, outputs=output)
        
        # Compile with gradient clipping and a lower learning rate
        optimizer = Adam(learning_rate=self.initial_lr, clipnorm=1.0)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def fit(self, df, test_size=0.2, val_size=0.25, epochs=50, batch_size=32):
        # Preprocess data
        processed_data = self.preprocess_data(df)
        
        # Check if trip_count exists
        if 'trip_count' not in processed_data.columns:
            raise ValueError("Dataset must contain 'trip_count' column")
        
        # Store feature names
        self.feature_names = [col for col in processed_data.columns if col != 'trip_count']
        
        if self.ensemble:
            # Train separate models for weekdays and weekends
            return self.fit_ensemble(processed_data, test_size, val_size, epochs, batch_size)
        
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
        
        # Store X_train for feature importance analysis
        self.X_train = X_train
        
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
        
        # ---------- ENHANCED CALLBACKS ----------
        
        # Early stopping with more patience
        early_stopping = EarlyStopping(
            monitor='val_loss',
            #patience=15,  # Increased patience
            patience=50,
            restore_best_weights=True,
            verbose=1
        )
        
        # Model checkpoint
        checkpoint = ModelCheckpoint(
            'best_trip_prediction_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        
        # Learning rate reduction on plateau
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            #patience=5,
            patience=50,
            min_lr=0.00001,
            verbose=1
        )
        
        callbacks = [early_stopping, checkpoint, reduce_lr]
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate on test data
        test_loss, test_mae = self.model.evaluate(X_test, y_test)
        print(f"Test Loss (MSE): {test_loss}")
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
        y_true = self.scaler_y.inverse_transform(y_test)
        
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
        
        # Only analyze feature importance if model trained successfully
        try:
            self.analyze_feature_importance(X_train, self.feature_names)
        except Exception as e:
            print(f"Skipping feature importance analysis due to error: {e}")
        
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
        
        # Train weekday model
        print("\n===== Training WEEKDAY Model =====")
        self.models['weekday'] = ImprovedTripPredictionModel(
            seq_length=self.seq_length, 
            ensemble=False,
            lr=self.initial_lr
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
        self.models['weekend'] = ImprovedTripPredictionModel(
            seq_length=self.seq_length, 
            ensemble=False,
            lr=self.initial_lr
        )
        weekend_history = self.models['weekend'].fit(
            weekend_data, 
            test_size=test_size, 
            val_size=val_size, 
            epochs=epochs, 
            batch_size=batch_size
        )
        
        print("\n===== Ensemble Training Complete =====")
        
        # Return combined history (just for reference)
        return {
            'weekday': weekday_history,
            'weekend': weekend_history
        }
    
    def plot_training_history(self, history):
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title('Mean Absolute Error')
        plt.xlabel('Epochs')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()
    
    def plot_predictions(self, y_true, y_pred, n_samples=200):
        plt.figure(figsize=(15, 6))
        plt.plot(y_true[:n_samples], label='Actual')
        plt.plot(y_pred[:n_samples], label='Predicted')
        plt.title(f'Trip Count: Actual vs Predicted (First {n_samples} hours)')
        plt.xlabel('Hours')
        plt.ylabel('Trip Count')
        plt.legend()
        plt.savefig('prediction_results.png')
        plt.show()
        
        # Add residual plot
        plt.figure(figsize=(15, 6))
        residuals = y_true[:n_samples] - y_pred[:n_samples]
        plt.scatter(y_pred[:n_samples], residuals)
        plt.axhline(y=0, color='r', linestyle='-')
        plt.title('Residual Plot')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.savefig('residual_plot.png')
        plt.show()
    
    def analyze_feature_importance(self, X_train, feature_names, n_top=15):
        print("Analyzing feature importance... (this may take some time)")
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
        
        plt.figure(figsize=(12, 8))
        sns.barplot(data=importance_df.head(n_top), x='Importance', y='Feature')
        plt.title(f'Top {n_top} Feature Importances')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.show()
        
        print(f"Top 10 most important features:")
        print(importance_df.head(10))
        
        return importance_df
    
    def predict(self, new_data):
        """
        Make predictions for new data using either a single model or the ensemble.
        
        Args:
            new_data: DataFrame with same format as training data
        
        Returns:
            Predicted trip counts
        """
        # Check if new_data exists
        if new_data is None or len(new_data) == 0:
            raise ValueError("No data provided for prediction")
            
        # If using ensemble models
        if self.ensemble and self.models:
            # Determine if weekday or weekend for the prediction point
            if 'datetime' in new_data.columns:
                # Convert to datetime if it's not already
                if not pd.api.types.is_datetime64_any_dtype(new_data['datetime']):
                    new_data['datetime'] = pd.to_datetime(new_data['datetime'], errors='coerce')
                
                # Get the day of week for the last row (prediction point)
                last_day = new_data['datetime'].iloc[-1].dayofweek
                is_weekend = last_day >= 5  # 5 and 6 are weekend days (Saturday and Sunday)
                
                if is_weekend:
                    return self.models['weekend'].predict(new_data)
                else:
                    return self.models['weekday'].predict(new_data)
            else:
                # If we can't determine day type, use the last row's is_weekend column if available
                if 'is_weekend' in new_data.columns:
                    is_weekend = new_data['is_weekend'].iloc[-1] == 1
                    if is_weekend:
                        return self.models['weekend'].predict(new_data)
                    else:
                        return self.models['weekday'].predict(new_data)
                else:
                    # Fall back to using weekday model
                    print("Warning: Cannot determine if weekend or weekday. Using weekday model.")
                    return self.models['weekday'].predict(new_data)
        
        # Single model prediction
        # Preprocess data
        processed_data = self.preprocess_data(new_data)
        
        # Scale data
        X_scaled = self.scaler_X.transform(processed_data)
        
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
            # Save each model separately
            for model_type, model in self.models.items():
                model.save(f"{filepath}_{model_type}")
            
            # Save ensemble configuration
            import pickle
            with open(f"{filepath}_ensemble_config.pkl", "wb") as f:
                pickle.dump({
                    'ensemble': True,
                    'seq_length': self.seq_length,
                    'initial_lr': self.initial_lr
                }, f)
            
            print(f"Ensemble models saved to {filepath}_weekday and {filepath}_weekend")
            return
        
        # Save single model
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
            
        # Save the model
        self.model.save(filepath)
        
        # Save the scalers and parameters
        import pickle
        with open(f"{filepath}_scalers.pkl", "wb") as f:
            pickle.dump({
                'scaler_X': self.scaler_X,
                'scaler_y': self.scaler_y,
                'seq_length': self.seq_length,
                'feature_names': self.feature_names,
                'ensemble': False
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
            
            # Create instance with ensemble flag
            instance = cls(
                seq_length=config['seq_length'], 
                ensemble=True,
                lr=config.get('initial_lr', 0.0005)
            )
            
            # Load each model
            instance.models = {
                'weekday': cls.load(f"{filepath}_weekday"),
                'weekend': cls.load(f"{filepath}_weekend")
            }
            
            print(f"Loaded ensemble models from {filepath}_weekday and {filepath}_weekend")
            return instance
        
        # Load single model
        model = load_model(filepath)
        
        # Load scalers and parameters
        with open(f"{filepath}_scalers.pkl", "rb") as f:
            saved_data = pickle.load(f)
        
        # Create instance and set attributes
        instance = cls(
            seq_length=saved_data['seq_length'],
            ensemble=saved_data.get('ensemble', False),
            lr=saved_data.get('initial_lr', 0.0005)
        )
        instance.model = model
        instance.scaler_X = saved_data['scaler_X']
        instance.scaler_y = saved_data['scaler_y']
        instance.feature_names = saved_data['feature_names']
        
        print(f"Model loaded from {filepath}")
        return instance


# --- New functionality for in-depth feature analysis ---

def analyze_features_in_depth(model, df):
    """
    Perform comprehensive feature importance analysis
    """
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
                      'trip_count_ma_3h', 'trip_count_ma_6h', 'trip_count_ma_12h']
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
    plt.title('Feature Importance by Category')
    
    # 2. Top features bar chart
    plt.subplot(1, 2, 2)
    top_n = 10
    sns.barplot(data=importance_df.head(top_n), y='Feature', x='Importance')
    plt.title(f'Top {top_n} Most Important Features')
    plt.tight_layout()
    plt.savefig('feature_importance_analysis.png')
    plt.show()
    
    # 3. Correlation heatmap of top features with trip count
    top_features = importance_df.head(15)['Feature'].tolist()
    correlation_data = processed_data[top_features + ['trip_count']]
    plt.figure(figsize=(12, 10))
    correlation_matrix = correlation_data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Between Top Features and Trip Count')
    plt.tight_layout()
    plt.savefig('feature_correlation_heatmap.png')
    plt.show()
    
    # Print insights
    print("\n=== FEATURE IMPORTANCE ANALYSIS ===")
    print("\nTop 10 Individual Features:")
    print(importance_df.head(10))
    
    print("\nFeature Importance by Category:")
    for category, importance in sorted(category_importance.items(), 
                                      key=lambda x: x[1], reverse=True):
        print(f"{category}: {importance:.4f} ({importance/sum(category_importance.values())*100:.1f}%)")
    
    # Find features with unexpected importance
    print("\nFeatures with Unexpectedly High Importance:")
    # Example: features with high importance but low correlation
    for feature in importance_df.head(15)['Feature']:
        correlation = abs(correlation_matrix.loc[feature, 'trip_count'])
        importance = importance_df[importance_df['Feature'] == feature]['Importance'].values[0]
        if importance > np.mean(importance_df['Importance']) and correlation < 0.3:
            print(f"- {feature}: High importance ({importance:.4f}) but low correlation ({correlation:.2f})")
            
    return importance_df, category_importance


# --- Weather forecast data fetcher for real-time predictions ---

def fetch_weather_forecast(api_key, lat, lon, hours_ahead=24):
    """
    Fetch weather forecast from OpenWeatherMap API
    
    Args:
        api_key: OpenWeatherMap API key
        lat, lon: Location coordinates
        hours_ahead: Number of hours to forecast
        
    Returns:
        DataFrame with forecast data formatted for the model
    """
    # OpenWeatherMap 5-day forecast endpoint (3-hour intervals)
    url = "https://api.openweathermap.org/data/2.5/forecast"
    
    params = {
        'lat': lat,
        'lon': lon,
        'appid': api_key,
        'units': 'metric'  # Use metric units
    }
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        if response.status_code != 200:
            print(f"Error fetching forecast: {data.get('message', 'Unknown error')}")
            return None
            
        # Process forecast data
        forecast_data = []
        current_time = datetime.now()
        
        for item in data['list']:
            forecast_time = datetime.fromtimestamp(item['dt'])
            
            # Only include future forecasts within requested hours
            time_diff = (forecast_time - current_time).total_seconds() / 3600
            if 0 <= time_diff <= hours_ahead:
                forecast_data.append({
                    'datetime': forecast_time,
                    'temp': item['main']['temp'],
                    'feels_like': item['main']['feels_like'],
                    'pressure': item['main']['pressure'],
                    'humidity': item['main']['humidity'],
                    'clouds': item['clouds']['all'],
                    'wind_speed': item['wind']['speed'],
                    'wind_deg': item['wind']['deg'],
                    'weather_main': item['weather'][0]['main'],
                    'weather_description': item['weather'][0]['description'],
                    'weather_icon': item['weather'][0]['icon'],
                    'rain_1h': item['rain'].get('1h', 0) if 'rain' in item else 0,
                    'snow_1h': item['snow'].get('1h', 0) if 'snow' in item else 0
                })
        
        # Convert to DataFrame
        if not forecast_data:
            print("No forecast data available for the requested time period")
            return None
            
        forecast_df = pd.DataFrame(forecast_data)
        
        # Add required columns that might be missing
        if 'dew_point' not in forecast_df.columns:
            forecast_df['dew_point'] = forecast_df['temp'] - ((100 - forecast_df['humidity']) / 5)
        
        # Add basic time features needed by the model
        forecast_df['hour'] = forecast_df['datetime'].dt.hour
        forecast_df['day'] = forecast_df['datetime'].dt.day
        forecast_df['day_of_week'] = forecast_df['datetime'].dt.dayofweek
        forecast_df['month'] = forecast_df['datetime'].dt.month
        forecast_df['is_weekend'] = (forecast_df['day_of_week'] >= 5).astype(int)
        
        # Add other required columns with default values
        required_cols = ['uvi', 'visibility', 'weather_id', 'timezone_offset']
        for col in required_cols:
            if col not in forecast_df.columns:
                forecast_df[col] = 0
        
        return forecast_df
        
    except Exception as e:
        print(f"Error fetching weather forecast: {e}")
        return None


def predict_trips_with_forecast(model_path, api_key, lat, lon, hours_ahead=24):
    """
    Predict trip counts using weather forecast
    
    Args:
        model_path: Path to saved model
        api_key: OpenWeatherMap API key
        lat, lon: Location coordinates
        hours_ahead: Number of hours to forecast
        
    Returns:
        DataFrame with predicted trip counts
    """
    # Load the model
    model = ImprovedTripPredictionModel.load(model_path)
    
    # Fetch weather forecast
    forecast_df = fetch_weather_forecast(api_key, lat, lon, hours_ahead)
    
    if forecast_df is None or len(forecast_df) == 0:
        print("No forecast data available")
        return None
    
    # Get historical data for context (last 24 hours)
    # In a real system, you would fetch this from your database
    # Here we'll just use dummy data
    historical_data = []
    start_time = forecast_df['datetime'].min() - timedelta(hours=model.seq_length)
    for i in range(model.seq_length):
        historical_data.append({
            'datetime': start_time + timedelta(hours=i),
            'trip_count': 0,  # This will be ignored for prediction
            'temp': 20,  # Placeholder values
            'feels_like': 20,
            'pressure': 1013,
            'humidity': 70,
            'clouds': 50,
            'wind_speed': 5,
            'wind_deg': 180,
            'weather_main': 'Clear',
            'weather_description': 'clear sky',
            'weather_icon': '01d',
            'rain_1h': 0,
            'snow_1h': 0,
            'hour': [(start_time + timedelta(hours=i)).hour for i in range(model.seq_length)],
            'day': [(start_time + timedelta(hours=i)).day for i in range(model.seq_length)],
            'day_of_week': [(start_time + timedelta(hours=i)).weekday() for i in range(model.seq_length)],
            'month': [(start_time + timedelta(hours=i)).month for i in range(model.seq_length)],
            'is_weekend': [1 if (start_time + timedelta(hours=i)).weekday() >= 5 else 0 for i in range(model.seq_length)]
        })
    
    historical_df = pd.DataFrame(historical_data)
    
    # Combine historical and forecast data
    combined_data = pd.concat([historical_df, forecast_df])
    
    # Make predictions for each hour
    predictions = []
    for i in range(len(forecast_df)):
        # Get the window of data needed for this prediction
        window_start = i
        window_end = i + model.seq_length
        prediction_data = combined_data.iloc[window_start:window_end].copy()
        
        # Make prediction
        trip_count = model.predict(prediction_data)
        
        # Store result
        predictions.append({
            'datetime': forecast_df.iloc[i]['datetime'],
            'predicted_trips': trip_count,
            'temp': forecast_df.iloc[i]['temp'],
            'weather_main': forecast_df.iloc[i]['weather_main'],
            'is_weekend': forecast_df.iloc[i]['is_weekend']
        })
    
    return pd.DataFrame(predictions)


# --- Multi-step prediction model ---

class MultiStepTripPredictionModel(ImprovedTripPredictionModel):
    def __init__(self, seq_length=12, prediction_horizon=6, ensemble=False, lr=0.0005):
        super().__init__(seq_length=seq_length, ensemble=ensemble, lr=lr)
        self.prediction_horizon = prediction_horizon  # Number of future steps to predict
        self.prediction_metrics = None
    
    def create_sequences(self, X, y):
        """Modified to create sequences with multiple target values"""
        # Check for NaN values and replace them, same as parent class
        if np.isnan(X).any():
            print(f"WARNING: X contains {np.isnan(X).sum()} NaN values before sequence creation")
            col_means = np.nanmean(X, axis=0)
            X = np.where(np.isnan(X), np.tile(col_means, (X.shape[0], 1)), X)
            
        if np.isnan(y).any():
            print(f"WARNING: y contains {np.isnan(y).sum()} NaN values before sequence creation")
            y_mean = np.nanmean(y)
            y = np.where(np.isnan(y), y_mean, y)
        
        Xs, ys = [], []
        
        # Need additional steps at the end for targets
        for i in range(len(X) - self.seq_length - self.prediction_horizon + 1):
            # Input sequence
            seq_x = X[i:(i + self.seq_length)]
            
            # Target sequence (multiple future values)
            seq_y = y[i + self.seq_length:i + self.seq_length + self.prediction_horizon]
            
            # Only add if no NaN values
            if not np.isnan(seq_x).any() and not np.isnan(seq_y).any():
                Xs.append(seq_x)
                ys.append(seq_y)
        
        if len(Xs) == 0:
            raise ValueError("No valid sequences could be created due to NaN values")
            
        return np.array(Xs), np.array(ys)
    
    def build_model(self, input_shape):
        """Modified to predict multiple time steps"""
        # Input layer
        input_layer = Input(shape=input_shape)
        
        # LSTM layers (similar to original model)
        lstm1 = Bidirectional(LSTM(128, 
                               activation='relu', 
                               return_sequences=True,
                               kernel_initializer='he_normal',
                               recurrent_initializer='orthogonal',
                               kernel_regularizer=l2(0.001),
                               recurrent_regularizer=l2(0.001)))(input_layer)
        
        batch_norm1 = BatchNormalization()(lstm1)
        dropout1 = Dropout(0.4)(batch_norm1)
        
        lstm2 = Bidirectional(LSTM(64, 
                               activation='relu', 
                               return_sequences=True,
                               kernel_initializer='he_normal',
                               recurrent_initializer='orthogonal',
                               kernel_regularizer=l2(0.001),
                               recurrent_regularizer=l2(0.001)))(dropout1)
        
        batch_norm2 = BatchNormalization()(lstm2)
        dropout2 = Dropout(0.4)(batch_norm2)
        
        # Attention mechanism
        attention = Dense(1, activation='tanh')(dropout2)
        attention = Flatten()(attention)
        attention = Dense(input_shape[0], activation='softmax')(attention)
        attention = RepeatVector(lstm2.shape[2])(attention)
        attention = Permute([2, 1])(attention)
        
        weighted = Concatenate()([dropout2, attention])
        weighted = Dense(lstm2.shape[2], activation='relu')(weighted)
        
        context_vector = GlobalAveragePooling1D()(weighted)
        
        # Dense layers for shared features
        dense1 = Dense(32, activation='relu')(context_vector)
        batch_norm3 = BatchNormalization()(dense1)
        shared_features = Dropout(0.3)(batch_norm3)
        
        # Output layers - one for each time step in prediction horizon
        outputs = []
        for i in range(self.prediction_horizon):
            output_layer = Dense(1, name=f'output_{i}')(shared_features)
            outputs.append(output_layer)
        
        # Create model with multiple outputs
        model = Model(inputs=input_layer, outputs=outputs)
        
        # Loss and metrics dictionaries
        loss_dict = {f'output_{i}': 'mse' for i in range(self.prediction_horizon)}
        metrics_dict = {f'output_{i}': 'mae' for i in range(self.prediction_horizon)}
        
        # Compile with gradient clipping
        optimizer = Adam(learning_rate=self.initial_lr, clipnorm=1.0)
        model.compile(
            optimizer=optimizer,
            loss=loss_dict,
            metrics=metrics_dict
        )
        
        return model
    
    def fit(self, df, test_size=0.2, val_size=0.25, epochs=50, batch_size=32):
        """Modified to handle multiple output predictions"""
        # Most preprocessing steps are the same as the parent class
        processed_data = self.preprocess_data(df)
        
        if 'trip_count' not in processed_data.columns:
            raise ValueError("Dataset must contain 'trip_count' column")
        
        self.feature_names = [col for col in processed_data.columns if col != 'trip_count']
        
        # Split features and target
        X = processed_data.drop('trip_count', axis=1)
        y = processed_data['trip_count']
        
        # Print statistics before scaling
        print("\nData statistics before scaling:")
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        print(f"X contains NaN: {X.isna().any().any()}")
        print(f"y contains NaN: {y.isna().any()}")
        
        # Scale data
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
            for col in X.columns:
                if X[col].isna().any():
                    print(f"Column {col} has {X[col].isna().sum()} NaN values")
            raise
        
        # Create sequences with multiple targets
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
        
        # Store for feature importance
        self.X_train = X_train
        
        # Build model
        input_shape = (X_train.shape[1], X_train.shape[2])
        self.model = self.build_model(input_shape)
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
            ModelCheckpoint('best_multistep_model.h5', monitor='val_loss', save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)
        ]
        
        # Create dictionaries for targets
        y_train_dict = {f'output_{i}': y_train[:, i].reshape(-1, 1) for i in range(self.prediction_horizon)}
        y_val_dict = {f'output_{i}': y_val[:, i].reshape(-1, 1) for i in range(self.prediction_horizon)}
        
        # Train model
        history = self.model.fit(
            X_train, y_train_dict,
            validation_data=(X_val, y_val_dict),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate the model
        print("\nEvaluating model on test data...")
        y_test_dict = {f'output_{i}': y_test[:, i].reshape(-1, 1) for i in range(self.prediction_horizon)}
        results = self.model.evaluate(X_test, y_test_dict)
        
        # Make predictions on test set
        y_pred_scaled = self.model.predict(X_test)
        
        # Convert predictions to numpy array if it's a list
        if isinstance(y_pred_scaled, list):
            y_pred_scaled = np.array([pred.flatten() for pred in y_pred_scaled]).T
        
        # Inverse transform predictions and actual values
        y_pred = np.zeros_like(y_pred_scaled)
        y_true = np.zeros_like(y_test)
        
        for i in range(self.prediction_horizon):
            if isinstance(y_pred_scaled, list):
                y_pred[:, i] = self.scaler_y.inverse_transform(
                    y_pred_scaled[i]).flatten()
            else:
                y_pred[:, i] = self.scaler_y.inverse_transform(
                    y_pred_scaled[:, i].reshape(-1, 1)).flatten()
                
            y_true[:, i] = self.scaler_y.inverse_transform(
                y_test[:, i].reshape(-1, 1)).flatten()
        
        # Calculate metrics for each horizon
        print("\nMulti-step Prediction Results:")
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        all_mae = []
        all_rmse = []
        all_r2 = []
        
        for i in range(self.prediction_horizon):
            horizon_mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
            horizon_rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
            horizon_r2 = r2_score(y_true[:, i], y_pred[:, i])
            
            all_mae.append(horizon_mae)
            all_rmse.append(horizon_rmse)
            all_r2.append(horizon_r2)
            
            hours_ahead = i + 1
            print(f"{hours_ahead} hour{'s' if hours_ahead > 1 else ''} ahead:")
            print(f"  MAE: {horizon_mae:.2f}")
            print(f"  RMSE: {horizon_rmse:.2f}")
            print(f"  R²: {horizon_r2:.4f}")
        
        # Plot multi-step prediction results
        self.plot_multistep_predictions(y_true, y_pred)
        
        # Store the results
        self.prediction_metrics = {
            'horizons': list(range(1, self.prediction_horizon + 1)),
            'mae': all_mae,
            'rmse': all_rmse,
            'r2': all_r2
        }
        
        return history
    
    def plot_multistep_predictions(self, y_true, y_pred, sample_idx=0):
        """Plot multi-step predictions for a single sample"""
        plt.figure(figsize=(12, 6))
        
        # Plot actual vs predicted for each horizon
        horizons = list(range(1, self.prediction_horizon + 1))
        plt.plot(horizons, y_true[sample_idx], 'bo-', label='Actual')
        plt.plot(horizons, y_pred[sample_idx], 'ro-', label='Predicted')
        
        plt.title('Multi-step Prediction Example')
        plt.xlabel('Hours Ahead')
        plt.ylabel('Trip Count')
        plt.xticks(horizons)
        plt.legend()
        plt.grid(True)
        plt.savefig('multistep_prediction_example.png')
        plt.show()
        
        # Plot prediction accuracy across horizons
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.plot(horizons, self.prediction_metrics['mae'], 'o-')
        plt.title('MAE by Horizon')
        plt.xlabel('Hours Ahead')
        plt.ylabel('Mean Absolute Error')
        plt.xticks(horizons)
        plt.grid(True)
        
        plt.subplot(1, 3, 2)
        plt.plot(horizons, self.prediction_metrics['rmse'], 'o-')
        plt.title('RMSE by Horizon')
        plt.xlabel('Hours Ahead')
        plt.ylabel('Root Mean Squared Error')
        plt.xticks(horizons)
        plt.grid(True)
        
        plt.subplot(1, 3, 3)
        plt.plot(horizons, self.prediction_metrics['r2'], 'o-')
        plt.title('R² by Horizon')
        plt.xlabel('Hours Ahead')
        plt.ylabel('R² Score')
        plt.xticks(horizons)
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('multistep_metrics_by_horizon.png')
        plt.show()
    
    def predict_ahead(self, new_data, steps_to_show=None):
        """Make multi-step predictions"""
        # Process the data
        processed_data = self.preprocess_data(new_data)
        X_scaled = self.scaler_X.transform(processed_data)
        
        # Create sequence for prediction
        if len(X_scaled) < self.seq_length:
            raise ValueError(f"Not enough data for prediction. Need at least {self.seq_length} rows.")
        
        sequence = X_scaled[-self.seq_length:].reshape(1, self.seq_length, X_scaled.shape[1])
        
        # Make predictions
        predictions_scaled = self.model.predict(sequence)
        
        # Convert predictions to numpy array if needed
        if isinstance(predictions_scaled, list):
            predictions = []
            for i, pred in enumerate(predictions_scaled):
                pred_value = self.scaler_y.inverse_transform(pred)[0, 0]
                predictions.append(pred_value)
        else:
            # Handle as 2D array
            predictions = np.zeros(self.prediction_horizon)
            for i in range(self.prediction_horizon):
                predictions[i] = self.scaler_y.inverse_transform(
                    predictions_scaled[0, i].reshape(-1, 1))[0, 0]
        
        # Determine how many steps to show
        if steps_to_show is None or steps_to_show > self.prediction_horizon:
            steps_to_show = self.prediction_horizon
        
        # Create result DataFrame
        if 'datetime' in new_data.columns:
            last_datetime = pd.to_datetime(new_data['datetime'].iloc[-1])
            future_datetimes = [last_datetime + pd.Timedelta(hours=i+1) for i in range(steps_to_show)]
            
            result_df = pd.DataFrame({
                'datetime': future_datetimes,
                'predicted_trips': predictions[:steps_to_show]
            })
        else:
            result_df = pd.DataFrame({
                'hours_ahead': list(range(1, steps_to_show + 1)),
                'predicted_trips': predictions[:steps_to_show]
            })
        
        return result_df


# --- Complete pipeline function ---

def run_complete_bike_sharing_pipeline():
    # Load data
    df = pd.read_csv('hourly_trips_with_weather.csv')
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    
    print("1. Running Base Model for Comparison")
    base_model = ImprovedTripPredictionModel(seq_length=8, ensemble=False)
    base_history = base_model.fit(df, epochs=20, batch_size=128)
    base_model.save('base_trip_prediction_model')
    
    print("\n2. Training Ensemble Model")
    ensemble_model = ImprovedTripPredictionModel(seq_length=8, ensemble=True)
    ensemble_history = ensemble_model.fit(df, epochs=20, batch_size=128)
    ensemble_model.save('ensemble_trip_prediction_model')
    
    print("\n3. Performing Feature Importance Analysis")
    analyze_features_in_depth(base_model, df)
    
    print("\n4. Training Multi-step Prediction Model")
    multistep_model = MultiStepTripPredictionModel(
        seq_length=8, prediction_horizon=6, ensemble=False)
    multistep_history = multistep_model.fit(df, epochs=20, batch_size=128)
    multistep_model.save('multistep_trip_prediction_model')
    
    print("\n5. Creating Weather Forecast Example (commented out - add your API key to use)")
    api_key = "your_openweathermap_api_key"
    # lat, lon = 42.3601, -71.0589  # Boston coordinates
    # predictions = predict_trips_with_forecast('ensemble_trip_prediction_model', api_key, lat, lon)
    
    print("\nComplete! All models have been trained and saved.")


# --- Main Script ---

# Load data and check its integrity first
df = pd.read_csv('hourly_trips_with_weather.csv')

# Explore data structure
print("Initial data exploration:")
print(f"Dataset shape: {df.shape}")
print(f"Column types:\n{df.dtypes}")
print(f"Missing values per column:\n{df.isna().sum()}")

# Check for suspicious values in trip_count (target variable)
if 'trip_count' in df.columns:
    print(f"Trip count statistics:")
    print(f"Min: {df['trip_count'].min()}")
    print(f"Max: {df['trip_count'].max()}")
    print(f"Mean: {df['trip_count'].mean()}")
    print(f"Zeros: {(df['trip_count'] == 0).sum()}")
    print(f"NaNs: {df['trip_count'].isna().sum()}")

# Convert datetime with explicit error handling
try:
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    # Check if conversion created NaN values
    if df['datetime'].isna().any():
        print(f"Warning: {df['datetime'].isna().sum()} datetime values could not be parsed")
except Exception as e:
    print(f"Error converting datetime: {e}")

# Initialize model with simpler architecture to avoid errors
model = ImprovedTripPredictionModel(
    seq_length=8,     # Use shorter sequence length to reduce memory/complexity
    ensemble=True,   # Set to True for separate weekday/weekend models
    lr=0.0005         # Lower initial learning rate
)

# Train the model with reduced complexity
try:
    history = model.fit(
        df, 
        epochs=100,      # Reduced epochs for faster training
        batch_size=128, # Larger batc h size for faster training
        test_size=0.2,  # 20% test data
        val_size=0.2    # 20% validation data
    )
    
    # Save model if successful
    model.save('improved_trip_prediction_model')

    # Test loading and prediction
    try:
        loaded_model = ImprovedTripPredictionModel.load('improved_trip_prediction_model')
        
        # For testing, just use a subset of the original data as "new" data
        sample_new_data = df.head(20).copy()  # Using 20 hours for testing
        predicted_trips = loaded_model.predict(sample_new_data)
        print(f"Predicted trips: {predicted_trips:.2f}")
        
        # Print actual trips for comparison
        actual_trips = df.iloc[20]['trip_count']
        print(f"Actual trips: {actual_trips}")
        print(f"Prediction error: {abs(predicted_trips - actual_trips):.2f} trips")
        
    except Exception as e:
        print(f"Error testing model: {e}")
        
except Exception as e:
    print(f"Training failed: {e}")
    # Print additional debugging information
    print("\nAdditional debugging information:")
    print(f"DataFrame contains NaN values: {df.isna().any().any()}")
    print(f"DataFrame contains infinite values: {np.isinf(df.select_dtypes(include=['float64', 'int64']).values).any()}")
    print(f"Column with most NaNs: {df.columns[df.isna().sum().argmax()]} ({df.isna().sum().max()} NaNs)")


# Uncomment the line below to run the complete pipeline
run_complete_bike_sharing_pipeline()
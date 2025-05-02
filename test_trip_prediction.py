import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os

# Import the TripPrediction2Hour class
from final_enhanced_predict2h import TripPrediction2Hour, smape_loss

class TestTripPrediction(unittest.TestCase):
    
    def setUp(self):
        """Create a small synthetic dataset for testing"""
        # Generate a small dataset with datetime and trip_count
        self.dates = pd.date_range(start='2023-01-01', periods=100, freq='h')
        self.trip_counts = np.random.randint(10, 100, size=100)
        
        # Create basic weather features
        self.temp = np.random.uniform(0, 30, size=100)  # Temperature in Celsius
        self.humidity = np.random.uniform(30, 90, size=100)  # Humidity percentage
        self.wind_speed = np.random.uniform(0, 20, size=100)  # Wind speed
        self.rain_1h = np.zeros(100)  # Mostly no rain
        self.rain_1h[::10] = np.random.uniform(1, 10, size=10)  # Random rain every 10th hour
        
        # Create weather categories
        self.weather_categories = ['Clear', 'Clouds', 'Rain']
        self.weather_main = np.random.choice(self.weather_categories, size=100)
        
        # Combine into a dataframe
        self.df = pd.DataFrame({
            'datetime': self.dates,
            'trip_count': self.trip_counts,
            'temp': self.temp,
            'humidity': self.humidity,
            'wind_speed': self.wind_speed,
            'rain_1h': self.rain_1h,
            'weather_main': self.weather_main
        })
        
        # Create a model with minimal settings for quick testing
        self.model = TripPrediction2Hour(
            seq_length=6,  # Use shorter sequences for testing
            ensemble=False,
            lr=0.01,
            batch_size=16,
            dropout_rate=0.2,
            skip_feature_importance=True  # Skip feature importance analysis for faster tests
        )
    
    def test_preprocessing(self):
        """Test the data preprocessing function"""
        processed_data = self.model.preprocess_data(self.df)
        
        # Check that basic processing was done
        self.assertIn('trip_count', processed_data.columns)
        self.assertIn('hour_of_day', processed_data.columns)
        self.assertIn('is_weekday', processed_data.columns)
        self.assertIn('is_weekend', processed_data.columns)
        
        # Check lag features were created
        self.assertIn('trip_count_lag_1', processed_data.columns)
        self.assertIn('trip_count_ma_3h', processed_data.columns)
        
        # Check weather categories
        for category in self.weather_categories:
            self.assertIn(f'weather_{category}', processed_data.columns)
        
        # Check no NaN values remain
        self.assertEqual(processed_data.isna().sum().sum(), 0)
    
    def test_create_sequences(self):
        """Test sequence creation for LSTM input"""
        # Preprocess data first
        processed_data = self.model.preprocess_data(self.df)
        
        # Split features and target
        X = processed_data.drop('trip_count', axis=1).values
        y = processed_data['trip_count'].values.reshape(-1, 1)
        
        # Create sequences
        X_seq, y_seq = self.model.create_sequences(X, y)
        
        # Check shapes
        self.assertEqual(X_seq.shape[1], self.model.seq_length)
        self.assertEqual(X_seq.shape[2], X.shape[1])
        self.assertEqual(len(y_seq), len(X_seq))
    
    def test_build_model(self):
        """Test model building"""
        # Build a model with a dummy input shape
        input_shape = (6, 10)  # (sequence_length, num_features)
        model = self.model.build_model(input_shape)
        
        # Check that model has been created with correct structure
        self.assertIsNotNone(model)
        self.assertEqual(model.input_shape, (None, 6, 10))
        self.assertEqual(model.output_shape, (None, 1))
    
    def test_save_load(self):
        """Test model saving and loading"""
        # Create a temporary directory for saved models
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, 'test_model.keras')
            
            # First, we need to process the data and build a model to save
            processed_data = self.model.preprocess_data(self.df)
            X = processed_data.drop('trip_count', axis=1)
            input_shape = (self.model.seq_length, X.shape[1])
            self.model.model = self.model.build_model(input_shape)
            
            # Save the model
            self.model.feature_names = X.columns.tolist()
            self.model.save(model_path)
            
            # Check that model files were created
            self.assertTrue(os.path.exists(model_path))
            self.assertTrue(os.path.exists(f"{model_path.replace('.keras', '')}_scalers.pkl"))
            
            # Try to load the model (this just tests the function doesn't crash)
            try:
                loaded_model = TripPrediction2Hour.load(model_path)
                self.assertIsNotNone(loaded_model.model)
            except Exception as e:
                self.fail(f"Model loading raised exception: {e}")
    
    @unittest.skip("Skip fit test as it takes too long for CI pipeline")
    def test_fit_mini(self):
        """Test model fitting with minimal dataset and epochs"""
        # This is a very minimal test just to ensure the fit method runs
        # without error. Not intended to train a good model.
        small_df = self.df.iloc[:50].copy()  # Use just 50 samples
        
        try:
            # Note: epochs parameter is passed to fit(), not the constructor
            history = self.model.fit(small_df, epochs=1, batch_size=16)
            self.assertIsNotNone(history)
        except Exception as e:
            self.fail(f"Model fitting raised exception: {e}")

    def test_smape_loss(self):
        """Test SMAPE loss function"""
        # Test with dummy values
        y_true = np.array([10, 20, 30, 40])
        y_pred = np.array([12, 18, 33, 38])
        
        # Convert to tensors to match function requirements
        import tensorflow as tf
        y_true_tf = tf.constant(y_true, dtype=tf.float32)
        y_pred_tf = tf.constant(y_pred, dtype=tf.float32)
        
        # Calculate loss
        loss = smape_loss(y_true_tf, y_pred_tf).numpy()
        
        # SMAPE should be between 0 and 2
        self.assertGreaterEqual(loss, 0)
        self.assertLessEqual(loss, 2)

if __name__ == '__main__':
    unittest.main()
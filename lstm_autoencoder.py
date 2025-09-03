"""
Ultra-Minimal LSTM Autoencoder - Clean TensorFlow Implementation
Different import strategy to avoid 'asset' conflicts
"""

import numpy as np
import pandas as pd

# Try different TensorFlow import approach
print("🔄 Importing TensorFlow with clean namespace...")
try:
    # Clear any existing TensorFlow imports
    import sys
    tf_modules = [module for module in sys.modules if 'tensorflow' in module.lower()]
    for module in tf_modules:
        if module in sys.modules:
            del sys.modules[module]
    
    # Fresh import
    import tensorflow as tf
    print(f"✅ TensorFlow {tf.__version__} imported successfully")
    
    # Import specific components
    from tensorflow.keras import layers, models, optimizers
    from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
    print("✅ Keras components imported successfully")
    
except Exception as e:
    print(f"❌ TensorFlow import failed: {e}")
    exit(1)

# Disable TensorFlow warnings
tf.get_logger().setLevel('ERROR')

class MinimalLSTMAutoencoder:
    """Ultra-minimal LSTM Autoencoder implementation"""
    
    def __init__(self, sequence_length=20, features=None):
        self.sequence_length = sequence_length
        self.features = features or 5
        self.model = None
        self.threshold = None
        self.scaler_mean = None
        self.scaler_std = None
        
        print(f"🧠 Minimal LSTM Autoencoder initialized")
        print(f"   Sequence length: {sequence_length}")
        print(f"   Expected features: {self.features}")
    
    def build_model(self):
        """Build the simplest possible LSTM autoencoder"""
        try:
            print("🔨 Building LSTM model...")
            
            # Use Sequential model (simplest approach)
            model = models.Sequential(name='lstm_autoencoder')
            
            # Encoder
            model.add(layers.LSTM(32, activation='tanh', input_shape=(self.sequence_length, self.features), 
                                return_sequences=True, name='encoder_1'))
            model.add(layers.LSTM(16, activation='tanh', return_sequences=False, name='encoder_2'))
            
            # Decoder
            model.add(layers.RepeatVector(self.sequence_length, name='repeat'))
            model.add(layers.LSTM(16, activation='tanh', return_sequences=True, name='decoder_1'))
            model.add(layers.LSTM(32, activation='tanh', return_sequences=True, name='decoder_2'))
            
            # Output
            model.add(layers.TimeDistributed(layers.Dense(self.features, name='output_dense'), name='output'))
            
            # Compile with basic settings
            model.compile(optimizer=optimizers.Adam(learning_rate=0.001), 
                         loss='mse', 
                         metrics=['mae'])
            
            self.model = model
            print("✅ Model built successfully")
            print(f"   Total parameters: {model.count_params():,}")
            
            return True
            
        except Exception as e:
            print(f"❌ Model building failed: {e}")
            return False
    
    def prepare_data(self, df):
        """Prepare data for LSTM"""
        try:
            # Select numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                return None, None
            
            # Use first few columns
            selected_cols = numeric_cols[:self.features]
            data = df[selected_cols].values.astype(np.float32)
            
            # Simple normalization
            self.scaler_mean = np.mean(data, axis=0)
            self.scaler_std = np.std(data, axis=0) + 1e-8
            normalized_data = (data - self.scaler_mean) / self.scaler_std
            
            # Create sequences
            X, y = [], []
            for i in range(len(normalized_data) - self.sequence_length + 1):
                seq = normalized_data[i:i + self.sequence_length]
                X.append(seq)
                y.append(seq)
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            print(f"❌ Data preparation failed: {e}")
            return None, None
    
    def train(self, df, epochs=20, batch_size=16):
        """Train the LSTM autoencoder"""
        try:
            print("🚀 Training LSTM Autoencoder...")
            
            # Prepare data
            X, y = self.prepare_data(df)
            if X is None:
                return {'status': 'failed', 'message': 'Data preparation failed'}
            
            print(f"📊 Training data shape: {X.shape}")
            
            # Build model
            if not self.build_model():
                return {'status': 'failed', 'message': 'Model building failed'}
            
            # Train
            print("🔄 Training...")
            history = self.model.fit(
                X, y,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                verbose=1,
                shuffle=True
            )
            
            # Calculate threshold
            predictions = self.model.predict(X, verbose=0)
            mse_values = np.mean(np.square(X - predictions), axis=(1, 2))
            self.threshold = np.percentile(mse_values, 95)
            
            result = {
                'status': 'success',
                'final_loss': float(history.history['loss'][-1]),
                'threshold': float(self.threshold),
                'epochs': len(history.history['loss'])
            }
            
            print("✅ Training completed!")
            print(f"   Final loss: {result['final_loss']:.6f}")
            print(f"   Threshold: {result['threshold']:.6f}")
            
            return result
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return {'status': 'failed', 'message': str(e)}
    
    def detect_anomalies(self, df):
        """Detect anomalies"""
        try:
            if self.model is None:
                return {'anomaly_scores': np.array([]), 'is_anomaly': np.array([])}
            
            print("🔍 Detecting anomalies...")
            
            # Prepare data
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            selected_cols = numeric_cols[:self.features]
            data = df[selected_cols].values.astype(np.float32)
            
            # Normalize
            normalized_data = (data - self.scaler_mean) / self.scaler_std
            
            # Create sequences
            X = []
            for i in range(len(normalized_data) - self.sequence_length + 1):
                seq = normalized_data[i:i + self.sequence_length]
                X.append(seq)
            
            if not X:
                return {'anomaly_scores': np.zeros(len(df)), 'is_anomaly': np.zeros(len(df), dtype=bool)}
            
            X = np.array(X)
            
            # Predict
            predictions = self.model.predict(X, verbose=0)
            mse_values = np.mean(np.square(X - predictions), axis=(1, 2))
            
            # Detect anomalies
            anomaly_flags = mse_values > self.threshold
            
            # Pad to original length
            pad_length = len(df) - len(mse_values)
            if pad_length > 0:
                median_mse = np.median(mse_values)
                full_mse = np.concatenate([np.full(pad_length, median_mse), mse_values])
                full_flags = np.concatenate([np.zeros(pad_length, dtype=bool), anomaly_flags])
            else:
                full_mse = mse_values
                full_flags = anomaly_flags
            
            anomaly_count = np.sum(full_flags)
            
            result = {
                'anomaly_scores': full_mse,
                'is_anomaly': full_flags,
                'anomaly_count': int(anomaly_count),
                'anomaly_rate': float(anomaly_count / len(full_flags) * 100)
            }
            
            print(f"✅ Found {anomaly_count} anomalies ({result['anomaly_rate']:.1f}%)")
            return result
            
        except Exception as e:
            print(f"❌ Detection failed: {e}")
            return {'anomaly_scores': np.array([]), 'is_anomaly': np.array([])}


# Test the minimal implementation
if __name__ == "__main__":
    print("🧪 Testing Minimal LSTM Autoencoder...")
    
    # Create test data
    np.random.seed(42)
    n_samples = 200
    
    # Generate time series data
    t = np.linspace(0, 4*np.pi, n_samples)
    cpu_util = 50 + 20 * np.sin(t) + np.random.normal(0, 3, n_samples)
    memory_util = 40 + 15 * np.cos(t * 1.2) + np.random.normal(0, 2, n_samples)
    error_rate = 0.05 + 0.03 * np.sin(t * 2) + np.random.normal(0, 0.01, n_samples)
    
    # Add anomalies
    anomaly_indices = [30, 80, 130, 180]
    cpu_util[anomaly_indices] += 40
    memory_util[anomaly_indices] += 30
    error_rate[anomaly_indices] += 0.2
    
    # Create DataFrame
    test_df = pd.DataFrame({
        'cpu_util': np.clip(cpu_util, 0, 100),
        'memory_util': np.clip(memory_util, 0, 100),
        'error_rate': np.clip(error_rate, 0, 1),
        'health_score': 100 - (cpu_util + memory_util) / 2,
        'pressure': (cpu_util + memory_util) / 2
    })
    
    # Test
    lstm_ae = MinimalLSTMAutoencoder(sequence_length=15, features=5)
    
    train_result = lstm_ae.train(test_df, epochs=10, batch_size=8)
    
    if train_result['status'] == 'success':
        detect_result = lstm_ae.detect_anomalies(test_df)
        print(f"\n✅ SUCCESS! Detected {detect_result['anomaly_count']} anomalies")
        print(f"   Anomaly rate: {detect_result['anomaly_rate']:.1f}%")
    else:
        print(f"\n❌ Failed: {train_result.get('message')}")

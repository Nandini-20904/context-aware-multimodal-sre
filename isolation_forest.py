"""
Isolation Forest for Multivariate Anomaly Detection
Uses scikit-learn's Isolation Forest for detecting outliers in feature space
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from typing import Dict, List, Optional, Tuple
import logging
import joblib
from pathlib import Path

class IsolationForestDetector:
    """Isolation Forest for detecting multivariate anomalies"""
    
    def __init__(self, config: Dict):
        self.config = config.get('ml_models', {}).get('anomaly_detection', {}).get('isolation_forest', {})
        self.logger = logging.getLogger(__name__)
        
        # Model parameters
        self.contamination = self.config.get('contamination', 0.1)
        self.n_estimators = self.config.get('n_estimators', 100)
        self.max_samples = self.config.get('max_samples', 'auto')
        self.random_state = self.config.get('random_state', 42)
        
        # Model components
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
        print(f"🌳 Isolation Forest initialized with contamination: {self.contamination}")
    
    def _build_model(self) -> IsolationForest:
        """Build the Isolation Forest model"""
        try:
            model = IsolationForest(
                contamination=self.contamination,
                n_estimators=self.n_estimators,
                max_samples=self.max_samples,
                random_state=self.random_state,
                n_jobs=-1,  # Use all available cores
                verbose=0
            )
            
            print("✅ Isolation Forest model built successfully")
            print(f"   Parameters: n_estimators={self.n_estimators}, contamination={self.contamination}")
            
            return model
            
        except Exception as e:
            print(f"❌ Error building Isolation Forest model: {str(e)}")
            raise
    
    def prepare_features(self, df: pd.DataFrame, feature_cols: List[str] = None) -> np.ndarray:
        """Prepare features for training/prediction"""
        try:
            if df.empty:
                return np.array([])
            
            # Select features
            if feature_cols is None:
                # Get numeric columns but exclude some that might not be useful
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                
                # Exclude timestamp-related and derived percentage columns
                exclude_patterns = ['timestamp', 'id', 'index', '_diff', '_pct_change', 
                                  'thread_id', 'time_since', 'age_hours', 'age_days']
                
                feature_cols = []
                for col in numeric_cols:
                    if not any(pattern in col.lower() for pattern in exclude_patterns):
                        feature_cols.append(col)
                
                # Limit to most important features to avoid curse of dimensionality
                if len(feature_cols) > 20:
                    # Prioritize certain feature types
                    priority_features = []
                    priority_patterns = ['util', 'rate', 'score', 'pressure', 'count', 
                                       'anomaly', 'critical', 'rolling', 'level_numeric']
                    
                    for pattern in priority_patterns:
                        for col in feature_cols:
                            if pattern in col.lower() and col not in priority_features:
                                priority_features.append(col)
                    
                    # Add remaining features up to limit
                    remaining_features = [col for col in feature_cols if col not in priority_features]
                    feature_cols = priority_features + remaining_features[:20-len(priority_features)]
            
            if not feature_cols:
                raise ValueError("No suitable features found for Isolation Forest")
            
            self.feature_names = feature_cols
            print(f"📊 Using features for Isolation Forest: {len(feature_cols)} features")
            print(f"   Features: {feature_cols[:10]}{'...' if len(feature_cols) > 10 else ''}")
            
            # Extract and handle missing values
            data = df[feature_cols].fillna(0)
            
            # Remove any infinite values
            data = data.replace([np.inf, -np.inf], 0)
            
            return data.values
            
        except Exception as e:
            print(f"❌ Error preparing features: {str(e)}")
            raise
    
    def train(self, df: pd.DataFrame, feature_cols: List[str] = None) -> Dict[str, any]:
        """Train the Isolation Forest model"""
        try:
            print("🚀 Training Isolation Forest...")
            
            # Prepare features
            data = self.prepare_features(df, feature_cols)
            if len(data) == 0:
                return {'status': 'failed', 'message': 'No data available for training'}
            
            print(f"📊 Training data shape: {data.shape}")
            
            # Scale features
            scaled_data = self.scaler.fit_transform(data)
            
            # Build and train model
            self.model = self._build_model()
            self.model.fit(scaled_data)
            
            # Get anomaly scores and predictions on training data
            anomaly_scores = self.model.decision_function(scaled_data)
            predictions = self.model.predict(scaled_data)
            
            # Calculate statistics
            n_anomalies = np.sum(predictions == -1)
            anomaly_rate = n_anomalies / len(predictions) * 100
            
            training_results = {
                'status': 'success',
                'n_features': data.shape[1],
                'training_samples': len(data),
                'n_anomalies_detected': int(n_anomalies),
                'anomaly_rate': float(anomaly_rate),
                'contamination': self.contamination,
                'feature_names': self.feature_names,
                'anomaly_score_stats': {
                    'mean': float(np.mean(anomaly_scores)),
                    'std': float(np.std(anomaly_scores)),
                    'min': float(np.min(anomaly_scores)),
                    'max': float(np.max(anomaly_scores))
                }
            }
            
            print("✅ Isolation Forest training completed!")
            print(f"   Anomalies detected: {n_anomalies} ({anomaly_rate:.1f}%)")
            print(f"   Feature count: {data.shape[1]}")
            
            return training_results
            
        except Exception as e:
            print(f"❌ Error training Isolation Forest: {str(e)}")
            return {'status': 'failed', 'message': str(e)}
    
    def detect_anomalies(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Detect anomalies in new data"""
        try:
            if self.model is None:
                print("⚠️ Model not trained yet. Training on provided data...")
                train_result = self.train(df)
                if train_result['status'] != 'success':
                    return {'anomaly_scores': np.array([]), 'is_anomaly': np.array([])}
            
            print("🔍 Detecting anomalies with Isolation Forest...")
            
            if df.empty:
                return {'anomaly_scores': np.array([]), 'is_anomaly': np.array([])}
            
            # Prepare features (use same features as training)
            if self.feature_names:
                available_features = [col for col in self.feature_names if col in df.columns]
                if not available_features:
                    print("⚠️ No matching features found for anomaly detection")
                    return {'anomaly_scores': np.array([]), 'is_anomaly': np.array([])}
                
                data = df[available_features].fillna(0)
                data = data.replace([np.inf, -np.inf], 0).values
            else:
                data = self.prepare_features(df)
            
            # Scale features
            scaled_data = self.scaler.transform(data)
            
            # Get anomaly scores and predictions
            anomaly_scores = self.model.decision_function(scaled_data)
            predictions = self.model.predict(scaled_data)
            
            # Convert predictions (-1 for anomaly, 1 for normal) to boolean
            is_anomaly = predictions == -1
            
            # Calculate statistics
            anomaly_count = np.sum(is_anomaly)
            anomaly_rate = anomaly_count / len(is_anomaly) * 100
            
            print(f"✅ Isolation Forest anomaly detection completed:")
            print(f"   {anomaly_count} anomalies detected ({anomaly_rate:.1f}%)")
            
            return {
                'anomaly_scores': anomaly_scores,
                'is_anomaly': is_anomaly,
                'predictions': predictions,
                'anomaly_count': anomaly_count,
                'anomaly_rate': anomaly_rate,
                'score_threshold': np.min(anomaly_scores[is_anomaly]) if anomaly_count > 0 else 0
            }
            
        except Exception as e:
            print(f"❌ Error detecting anomalies: {str(e)}")
            return {'anomaly_scores': np.array([]), 'is_anomaly': np.array([])}
    
    def get_feature_importance(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate feature importance based on anomaly contribution"""
        try:
            if self.model is None or self.feature_names is None:
                print("⚠️ Model not trained or features not defined")
                return {}
            
            # Get data
            data = df[self.feature_names].fillna(0)
            data = data.replace([np.inf, -np.inf], 0)
            scaled_data = self.scaler.transform(data.values)
            
            # Get base anomaly scores
            base_scores = self.model.decision_function(scaled_data)
            
            # Calculate feature importance by permutation
            feature_importance = {}
            
            for i, feature in enumerate(self.feature_names):
                # Permute this feature
                permuted_data = scaled_data.copy()
                np.random.shuffle(permuted_data[:, i])
                
                # Get new scores
                permuted_scores = self.model.decision_function(permuted_data)
                
                # Calculate importance as mean absolute difference
                importance = np.mean(np.abs(base_scores - permuted_scores))
                feature_importance[feature] = float(importance)
            
            # Normalize importance scores
            max_importance = max(feature_importance.values()) if feature_importance else 1
            if max_importance > 0:
                for feature in feature_importance:
                    feature_importance[feature] /= max_importance
            
            # Sort by importance
            sorted_importance = dict(sorted(feature_importance.items(), 
                                         key=lambda x: x[1], reverse=True))
            
            print(f"✅ Feature importance calculated for {len(sorted_importance)} features")
            
            return sorted_importance
            
        except Exception as e:
            print(f"❌ Error calculating feature importance: {str(e)}")
            return {}
    
    def save_model(self, model_path: str = "models/trained_models/isolation_forest.pkl"):
        """Save the trained model and scaler"""
        try:
            if self.model is None:
                print("⚠️ No model to save")
                return False
            
            # Create directory if it doesn't exist
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save model, scaler, and metadata together
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'contamination': self.contamination,
                'n_estimators': self.n_estimators
            }
            
            joblib.dump(model_data, model_path)
            print(f"✅ Model saved to {model_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving model: {str(e)}")
            return False
    
    def load_model(self, model_path: str = "models/trained_models/isolation_forest.pkl"):
        """Load a saved model and scaler"""
        try:
            if not Path(model_path).exists():
                print(f"⚠️ Model file not found: {model_path}")
                return False
            
            # Load model data
            model_data = joblib.load(model_path)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.contamination = model_data['contamination']
            self.n_estimators = model_data['n_estimators']
            
            print(f"✅ Model loaded from {model_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            return False


# Example usage
if __name__ == "__main__":
    # Test with sample data
    import pandas as pd
    
    # Create sample data with some anomalies
    np.random.seed(42)
    n_samples = 1000
    
    # Normal data
    cpu_util = np.random.normal(40, 10, n_samples)
    memory_util = np.random.normal(50, 8, n_samples)
    error_rate = np.random.exponential(0.01, n_samples)
    response_time = np.random.gamma(2, 50, n_samples)
    
    # Add some correlated anomalies
    anomaly_mask = np.random.random(n_samples) < 0.05  # 5% anomalies
    cpu_util[anomaly_mask] += np.random.normal(40, 10, np.sum(anomaly_mask))
    memory_util[anomaly_mask] += np.random.normal(30, 5, np.sum(anomaly_mask))
    error_rate[anomaly_mask] *= np.random.uniform(5, 15, np.sum(anomaly_mask))
    
    # Create DataFrame
    sample_df = pd.DataFrame({
        'cpu_util': np.clip(cpu_util, 0, 100),
        'memory_util': np.clip(memory_util, 0, 100),
        'error_rate': np.clip(error_rate, 0, 1),
        'response_time': response_time,
        'system_health_score': 100 - (cpu_util + memory_util) / 2,
        'resource_pressure': (cpu_util + memory_util) / 2,
        'has_error': error_rate > 0.05,
        'critical_cpu': cpu_util > 80,
        'critical_memory': memory_util > 80
    })
    
    # Test Isolation Forest
    config = {
        'ml_models': {
            'anomaly_detection': {
                'isolation_forest': {
                    'contamination': 0.05,
                    'n_estimators': 100,
                    'random_state': 42
                }
            }
        }
    }
    
    iso_forest = IsolationForestDetector(config)
    
    # Train and detect anomalies
    print("Training Isolation Forest...")
    train_result = iso_forest.train(sample_df)
    print(f"Training result: {train_result}")
    
    if train_result['status'] == 'success':
        print("\nDetecting anomalies...")
        detection_result = iso_forest.detect_anomalies(sample_df)
        print(f"Detected {detection_result['anomaly_count']} anomalies")
        
        # Get feature importance
        print("\nCalculating feature importance...")
        importance = iso_forest.get_feature_importance(sample_df)
        print("Top 5 most important features:")
        for feature, score in list(importance.items())[:5]:
            print(f"   {feature}: {score:.3f}")

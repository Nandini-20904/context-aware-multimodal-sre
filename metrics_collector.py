"""
Metrics Data Collector for SRE Incident Insight Engine
Handles loading and processing metrics data: timestamp, cpu_util, memory_util, error_rate
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from scipy import stats

class MetricsCollector:
    """Collect and process system metrics data"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def load_metrics_data(self, file_path: str = None) -> pd.DataFrame:
        """
        Load metrics data from CSV file
        Expected columns: timestamp, cpu_util, memory_util, error_rate
        """
        try:
            if file_path is None:
                file_path = "metrics.csv"
            
            print(f"📊 Loading metrics data from: {file_path}")
            
            # Load CSV data
            df = pd.read_csv(file_path)
            
            # Validate required columns
            required_cols = ['timestamp', 'cpu_util', 'memory_util', 'error_rate']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Validate numeric columns
            numeric_cols = ['cpu_util', 'memory_util', 'error_rate']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove rows with invalid data
            df = df.dropna(subset=numeric_cols)
            
            print(f"✅ Loaded {len(df)} metrics entries")
            return df
            
        except Exception as e:
            print(f"❌ Error loading metrics data: {str(e)}")
            return pd.DataFrame()
    
    def calculate_rolling_features(self, df: pd.DataFrame, windows: List[int] = [5, 15, 30]) -> pd.DataFrame:
        """Calculate rolling statistics for metrics"""
        try:
            if df.empty:
                return df
            
            processed_df = df.copy()
            metrics_cols = ['cpu_util', 'memory_util', 'error_rate']
            
            for window in windows:
                for col in metrics_cols:
                    # Rolling mean
                    processed_df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
                    
                    # Rolling std
                    processed_df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window, min_periods=1).std().fillna(0)
                    
                    # Rolling max
                    processed_df[f'{col}_rolling_max_{window}'] = df[col].rolling(window=window, min_periods=1).max()
                    
                    # Rolling min
                    processed_df[f'{col}_rolling_min_{window}'] = df[col].rolling(window=window, min_periods=1).min()
            
            print(f"✅ Added rolling features for windows: {windows}")
            return processed_df
            
        except Exception as e:
            print(f"❌ Error calculating rolling features: {str(e)}")
            return df
    
    def detect_anomalies(self, df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
        """Detect anomalies using statistical methods"""
        try:
            if df.empty:
                return df
            
            processed_df = df.copy()
            metrics_cols = ['cpu_util', 'memory_util', 'error_rate']
            
            for col in metrics_cols:
                # Z-score based anomaly detection
                z_scores = np.abs(stats.zscore(df[col]))
                processed_df[f'{col}_is_anomaly'] = z_scores > threshold
                processed_df[f'{col}_z_score'] = z_scores
                
                # IQR based anomaly detection
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                processed_df[f'{col}_is_outlier'] = (df[col] < lower_bound) | (df[col] > upper_bound)
            
            # Overall anomaly flag
            anomaly_cols = [col for col in processed_df.columns if '_is_anomaly' in col]
            processed_df['has_anomaly'] = processed_df[anomaly_cols].any(axis=1)
            
            outlier_cols = [col for col in processed_df.columns if '_is_outlier' in col]
            processed_df['has_outlier'] = processed_df[outlier_cols].any(axis=1)
            
            anomaly_count = processed_df['has_anomaly'].sum()
            outlier_count = processed_df['has_outlier'].sum()
            
            print(f"✅ Detected {anomaly_count} anomalies and {outlier_count} outliers")
            return processed_df
            
        except Exception as e:
            print(f"❌ Error detecting anomalies: {str(e)}")
            return df
    
    def calculate_derived_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived metrics and health scores"""
        try:
            if df.empty:
                return df
            
            processed_df = df.copy()
            
            # System health score (0-100, where 100 is healthy)
            cpu_health = 100 - processed_df['cpu_util']
            memory_health = 100 - processed_df['memory_util']
            error_health = 100 - (processed_df['error_rate'] * 100)  # Assuming error_rate is 0-1
            
            processed_df['system_health_score'] = (cpu_health + memory_health + error_health) / 3
            processed_df['system_health_score'] = processed_df['system_health_score'].clip(0, 100)
            
            # Resource pressure indicators
            processed_df['resource_pressure'] = (processed_df['cpu_util'] + processed_df['memory_util']) / 2
            processed_df['high_resource_pressure'] = processed_df['resource_pressure'] > 80
            
            # Critical thresholds
            processed_df['critical_cpu'] = processed_df['cpu_util'] > 90
            processed_df['critical_memory'] = processed_df['memory_util'] > 90
            processed_df['critical_errors'] = processed_df['error_rate'] > 0.1  # >10% error rate
            
            # Time-based features
            processed_df['hour'] = processed_df['timestamp'].dt.hour
            processed_df['day_of_week'] = processed_df['timestamp'].dt.dayofweek
            processed_df['is_business_hours'] = processed_df['hour'].between(9, 17)
            processed_df['is_weekend'] = processed_df['day_of_week'].isin([5, 6])
            
            # Rate of change features
            for col in ['cpu_util', 'memory_util', 'error_rate']:
                processed_df[f'{col}_diff'] = processed_df[col].diff().fillna(0)
                processed_df[f'{col}_pct_change'] = processed_df[col].pct_change().fillna(0)
            
            print("✅ Added derived metrics and health scores")
            return processed_df
            
        except Exception as e:
            print(f"❌ Error calculating derived metrics: {str(e)}")
            return df
    
    def get_metrics_summary(self, df: pd.DataFrame) -> Dict[str, any]:
        """Generate summary statistics for metrics"""
        try:
            if df.empty:
                return {}
            
            metrics_cols = ['cpu_util', 'memory_util', 'error_rate']
            summary = {}
            
            for col in metrics_cols:
                summary[col] = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'q25': float(df[col].quantile(0.25)),
                    'q50': float(df[col].quantile(0.50)),
                    'q75': float(df[col].quantile(0.75)),
                    'q95': float(df[col].quantile(0.95)),
                    'current': float(df[col].iloc[-1]) if len(df) > 0 else 0
                }
            
            # Overall health metrics
            if 'system_health_score' in df.columns:
                summary['overall'] = {
                    'avg_health_score': float(df['system_health_score'].mean()),
                    'current_health_score': float(df['system_health_score'].iloc[-1]) if len(df) > 0 else 0,
                    'anomaly_count': int(df.get('has_anomaly', pd.Series([False])).sum()),
                    'outlier_count': int(df.get('has_outlier', pd.Series([False])).sum()),
                    'high_pressure_periods': int(df.get('high_resource_pressure', pd.Series([False])).sum())
                }
            
            return summary
            
        except Exception as e:
            print(f"❌ Error generating metrics summary: {str(e)}")
            return {}
    
    def collect(self, file_path: str = None) -> Dict[str, any]:
        """Main collection method for metrics data"""
        try:
            # Load raw metrics data
            raw_data = self.load_metrics_data(file_path)
            if raw_data.empty:
                return {'metrics': pd.DataFrame(), 'summary': {}}
            
            # Calculate rolling features
            processed_data = self.calculate_rolling_features(raw_data)
            
            # Detect anomalies
            processed_data = self.detect_anomalies(processed_data)
            
            # Calculate derived metrics
            processed_data = self.calculate_derived_metrics(processed_data)
            
            # Generate summary
            summary = self.get_metrics_summary(processed_data)
            
            return {
                'metrics': processed_data,
                'summary': summary,
                'total_entries': len(processed_data),
                'anomalies_detected': processed_data.get('has_anomaly', pd.Series([False])).sum(),
                'outliers_detected': processed_data.get('has_outlier', pd.Series([False])).sum(),
                'avg_health_score': processed_data.get('system_health_score', pd.Series([0])).mean()
            }
            
        except Exception as e:
            print(f"❌ Error in metrics collection: {str(e)}")
            return {'metrics': pd.DataFrame(), 'summary': {}}


# Example usage
if __name__ == "__main__":
    sample_config = {'data_sources': {'metrics': {'enabled': True}}}
    collector = MetricsCollector(sample_config)
    result = collector.collect("metrics.csv")
    print(f"Collected {result['total_entries']} metrics entries")
    print(f"Health Score: {result['avg_health_score']:.2f}/100")

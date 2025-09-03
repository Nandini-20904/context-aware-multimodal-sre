"""
Log Data Collector for SRE Incident Insight Engine
Handles loading and processing log data with columns: timestamp, level, message
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
import re

class LogCollector:
    """Collect and process log data from various sources"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def load_log_data(self, file_path: str = None) -> pd.DataFrame:
        """
        Load log data from CSV file
        Expected columns: timestamp, level, message
        """
        try:
            if file_path is None:
                file_path = "logs.csv"
            
            print(f"📄 Loading log data from: {file_path}")
            
            # Load CSV data
            df = pd.read_csv(file_path)
            
            # Validate required columns
            required_cols = ['timestamp', 'level', 'message']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            print(f"✅ Loaded {len(df)} log entries from {file_path}")
            return df
            
        except Exception as e:
            print(f"❌ Error loading log data: {str(e)}")
            return pd.DataFrame()
    
    def preprocess_logs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess log data for ML models"""
        try:
            if df.empty:
                return df
            
            processed_df = df.copy()
            
            # Standardize log levels
            processed_df['level'] = processed_df['level'].str.upper()
            
            # Extract features from log messages
            processed_df['message_length'] = processed_df['message'].str.len()
            processed_df['has_error'] = processed_df['message'].str.contains(
                r'error|fail|exception|critical', case=False, na=False
            )
            processed_df['has_warning'] = processed_df['message'].str.contains(
                r'warning|warn|alert', case=False, na=False
            )
            
            # Extract IP addresses
            ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
            processed_df['has_ip'] = processed_df['message'].str.contains(ip_pattern, na=False)
            
            # Extract numerical values
            processed_df['numeric_count'] = processed_df['message'].str.findall(r'\d+').apply(len)
            
            # Time-based features
            processed_df['hour'] = processed_df['timestamp'].dt.hour
            processed_df['day_of_week'] = processed_df['timestamp'].dt.dayofweek
            processed_df['is_business_hours'] = processed_df['hour'].between(9, 17)
            
            # Log level encoding
            level_mapping = {'DEBUG': 0, 'INFO': 1, 'WARNING': 2, 'ERROR': 3, 'CRITICAL': 4}
            processed_df['level_numeric'] = processed_df['level'].map(level_mapping).fillna(1)
            
            print(f"✅ Preprocessed {len(processed_df)} log entries with features")
            return processed_df
            
        except Exception as e:
            print(f"❌ Error preprocessing logs: {str(e)}")
            return df
    
    def extract_error_patterns(self, df: pd.DataFrame) -> Dict[str, int]:
        """Extract common error patterns from log messages"""
        try:
            if df.empty:
                return {}
            
            error_logs = df[df['level'].isin(['ERROR', 'CRITICAL'])]
            if error_logs.empty:
                return {}
            
            patterns = {
                'database_error': r'database|db|sql|connection.*fail',
                'network_error': r'network|timeout|connection.*refused',
                'memory_error': r'memory|out of memory|oom|heap',
                'disk_error': r'disk|storage|filesystem|no space',
                'auth_error': r'auth|permission|unauthorized|forbidden',
                'server_error': r'server error|500|internal error'
            }
            
            pattern_counts = {}
            for pattern_name, pattern_regex in patterns.items():
                count = error_logs['message'].str.contains(pattern_regex, case=False, na=False).sum()
                if count > 0:
                    pattern_counts[pattern_name] = count
            
            return pattern_counts
            
        except Exception as e:
            print(f"❌ Error extracting error patterns: {str(e)}")
            return {}
    
    def collect(self, file_path: str = None) -> Dict[str, any]:
        """Main collection method"""
        try:
            raw_data = self.load_log_data(file_path)
            if raw_data.empty:
                return {'logs': pd.DataFrame(), 'error_patterns': {}}
            
            processed_data = self.preprocess_logs(raw_data)
            error_patterns = self.extract_error_patterns(processed_data)
            
            return {
                'logs': processed_data,
                'error_patterns': error_patterns,
                'total_entries': len(processed_data),
                'error_count': len(processed_data[processed_data['level'].isin(['ERROR', 'CRITICAL'])]),
                'warning_count': len(processed_data[processed_data['level'] == 'WARNING'])
            }
            
        except Exception as e:
            print(f"❌ Error in log collection: {str(e)}")
            return {'logs': pd.DataFrame(), 'error_patterns': {}}

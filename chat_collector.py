"""
Chat Data Collector for SRE Incident Insight Engine
Handles loading and processing chat data: timestamp, user, message
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
import logging
import re
from collections import Counter

class ChatCollector:
    """Collect and process chat/communication data"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def load_chat_data(self, file_path: str = None) -> pd.DataFrame:
        """
        Load chat data from CSV file
        Expected columns: timestamp, user, message
        """
        try:
            if file_path is None:
                file_path = "chat.csv"
            
            print(f"💬 Loading chat data from: {file_path}")
            
            # Load CSV data
            df = pd.read_csv(file_path)
            
            # Validate required columns
            required_cols = ['timestamp', 'user', 'message']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Clean message data
            df['message'] = df['message'].fillna('').astype(str)
            df['user'] = df['user'].fillna('unknown').astype(str)
            
            print(f"✅ Loaded {len(df)} chat messages from {len(df['user'].unique())} users")
            return df
            
        except Exception as e:
            print(f"❌ Error loading chat data: {str(e)}")
            return pd.DataFrame()
    
    def extract_incident_keywords(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract incident-related keywords and urgency indicators"""
        try:
            if df.empty:
                return df
            
            processed_df = df.copy()
            
            # Incident-related keywords
            incident_patterns = {
                'urgent': r'urgent|emergency|critical|asap|immediately|now',
                'down': r'down|outage|offline|unavailable|not working',
                'error': r'error|fail|broken|issue|problem|bug',
                'performance': r'slow|timeout|latency|performance|response time',
                'database': r'database|db|sql|connection|query',
                'network': r'network|connection|dns|routing|firewall',
                'server': r'server|host|instance|node|container',
                'monitoring': r'alert|alarm|monitoring|metric|dashboard'
            }
            
            # Extract keyword presence
            for keyword, pattern in incident_patterns.items():
                processed_df[f'mentions_{keyword}'] = processed_df['message'].str.contains(
                    pattern, case=False, na=False
                )
            
            # Overall incident mention flag
            incident_cols = [col for col in processed_df.columns if col.startswith('mentions_')]
            processed_df['mentions_incident'] = processed_df[incident_cols].any(axis=1)
            
            # Urgency scoring (0-10)
            urgency_score = 0
            urgency_score += processed_df['mentions_urgent'] * 3
            urgency_score += processed_df['mentions_down'] * 2
            urgency_score += processed_df['mentions_error'] * 1
            processed_df['urgency_score'] = urgency_score.clip(0, 10)
            
            print("✅ Extracted incident keywords and urgency indicators")
            return processed_df
            
        except Exception as e:
            print(f"❌ Error extracting incident keywords: {str(e)}")
            return df
    
    def analyze_communication_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze communication patterns and conversation flow"""
        try:
            if df.empty:
                return df
            
            processed_df = df.copy()
            
            # Message characteristics
            processed_df['message_length'] = processed_df['message'].str.len()
            processed_df['word_count'] = processed_df['message'].str.split().str.len()
            processed_df['has_question'] = processed_df['message'].str.contains(r'\?', na=False)
            processed_df['has_exclamation'] = processed_df['message'].str.contains(r'!', na=False)
            processed_df['all_caps'] = processed_df['message'].str.isupper()
            
            # Time-based features
            processed_df['hour'] = processed_df['timestamp'].dt.hour
            processed_df['day_of_week'] = processed_df['timestamp'].dt.dayofweek
            processed_df['is_business_hours'] = processed_df['hour'].between(9, 17)
            processed_df['is_weekend'] = processed_df['day_of_week'].isin([5, 6])
            
            # User activity patterns
            user_message_counts = processed_df['user'].value_counts()
            processed_df['user_activity_level'] = processed_df['user'].map(user_message_counts)
            
            # Time gaps between messages
            processed_df['time_since_last'] = processed_df['timestamp'].diff().dt.total_seconds().fillna(0)
            processed_df['rapid_response'] = processed_df['time_since_last'] < 60  # < 1 minute
            
            # Conversation threads (messages within 5 minutes grouped)
            time_diff = processed_df['timestamp'].diff().dt.total_seconds().fillna(0)
            thread_breaks = time_diff > 300  # 5 minutes
            processed_df['thread_id'] = thread_breaks.cumsum()
            
            print("✅ Analyzed communication patterns and conversation flow")
            return processed_df
            
        except Exception as e:
            print(f"❌ Error analyzing communication patterns: {str(e)}")
            return df
    
    def extract_mentioned_entities(self, df: pd.DataFrame) -> Dict[str, any]:
        """Extract mentioned services, systems, and technical entities"""
        try:
            if df.empty:
                return {}
            
            # Common technical entities to look for
            entity_patterns = {
                'services': r'\b(api|service|microservice|web|app|application)\b',
                'databases': r'\b(mysql|postgres|mongodb|redis|elasticsearch|db)\b',
                'infrastructure': r'\b(aws|azure|gcp|kubernetes|docker|server|instance)\b',
                'monitoring': r'\b(prometheus|grafana|datadog|newrelic|cloudwatch)\b',
                'urls': r'https?://[^\s]+',
                'ips': r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
                'error_codes': r'\b[4-5][0-9][0-9]\b'  # HTTP error codes
            }
            
            entities_found = {}
            all_messages = ' '.join(df['message'].str.lower())
            
            for entity_type, pattern in entity_patterns.items():
                matches = re.findall(pattern, all_messages, re.IGNORECASE)
                if matches:
                    entities_found[entity_type] = list(set(matches))  # Remove duplicates
            
            # Count mentions per entity type
            entity_counts = {}
            for entity_type in entity_patterns.keys():
                count = df['message'].str.contains(entity_patterns[entity_type], case=False, na=False).sum()
                if count > 0:
                    entity_counts[entity_type] = int(count)
            
            print(f"✅ Found entities: {list(entity_counts.keys())}")
            return {'entities': entities_found, 'entity_counts': entity_counts}
            
        except Exception as e:
            print(f"❌ Error extracting entities: {str(e)}")
            return {}
    
    def identify_incident_discussions(self, df: pd.DataFrame, time_window: int = 30) -> List[Dict]:
        """Identify periods of intensive incident-related discussions"""
        try:
            if df.empty:
                return []
            
            # Filter to incident-related messages
            incident_messages = df[df.get('mentions_incident', False) == True].copy()
            
            if incident_messages.empty:
                return []
            
            incidents = []
            current_incident = None
            
            for idx, row in incident_messages.iterrows():
                if current_incident is None:
                    # Start new incident
                    current_incident = {
                        'start_time': row['timestamp'],
                        'end_time': row['timestamp'],
                        'messages': [row.to_dict()],
                        'participants': {row['user']},
                        'urgency_scores': [row.get('urgency_score', 0)]
                    }
                else:
                    # Check if this message is within the time window
                    time_diff = (row['timestamp'] - current_incident['end_time']).total_seconds() / 60
                    
                    if time_diff <= time_window:
                        # Continue current incident
                        current_incident['end_time'] = row['timestamp']
                        current_incident['messages'].append(row.to_dict())
                        current_incident['participants'].add(row['user'])
                        current_incident['urgency_scores'].append(row.get('urgency_score', 0))
                    else:
                        # Close current incident and start new one
                        current_incident['duration_minutes'] = (
                            current_incident['end_time'] - current_incident['start_time']
                        ).total_seconds() / 60
                        current_incident['message_count'] = len(current_incident['messages'])
                        current_incident['participant_count'] = len(current_incident['participants'])
                        current_incident['avg_urgency'] = np.mean(current_incident['urgency_scores'])
                        current_incident['participants'] = list(current_incident['participants'])
                        
                        incidents.append(current_incident)
                        
                        # Start new incident
                        current_incident = {
                            'start_time': row['timestamp'],
                            'end_time': row['timestamp'],
                            'messages': [row.to_dict()],
                            'participants': {row['user']},
                            'urgency_scores': [row.get('urgency_score', 0)]
                        }
            
            # Don't forget the last incident
            if current_incident:
                current_incident['duration_minutes'] = (
                    current_incident['end_time'] - current_incident['start_time']
                ).total_seconds() / 60
                current_incident['message_count'] = len(current_incident['messages'])
                current_incident['participant_count'] = len(current_incident['participants'])
                current_incident['avg_urgency'] = np.mean(current_incident['urgency_scores'])
                current_incident['participants'] = list(current_incident['participants'])
                incidents.append(current_incident)
            
            print(f"✅ Identified {len(incidents)} incident discussion periods")
            return incidents
            
        except Exception as e:
            print(f"❌ Error identifying incident discussions: {str(e)}")
            return []
    
    def collect(self, file_path: str = None) -> Dict[str, any]:
        """Main collection method for chat data"""
        try:
            # Load raw chat data
            raw_data = self.load_chat_data(file_path)
            if raw_data.empty:
                return {'chats': pd.DataFrame(), 'entities': {}, 'incidents': []}
            
            # Extract incident keywords
            processed_data = self.extract_incident_keywords(raw_data)
            
            # Analyze communication patterns
            processed_data = self.analyze_communication_patterns(processed_data)
            
            # Extract mentioned entities
            entities = self.extract_mentioned_entities(processed_data)
            
            # Identify incident discussions
            incident_discussions = self.identify_incident_discussions(processed_data)
            
            return {
                'chats': processed_data,
                'entities': entities,
                'incident_discussions': incident_discussions,
                'total_messages': len(processed_data),
                'unique_users': processed_data['user'].nunique(),
                'incident_related_messages': processed_data.get('mentions_incident', pd.Series([False])).sum(),
                'avg_urgency_score': processed_data.get('urgency_score', pd.Series([0])).mean()
            }
            
        except Exception as e:
            print(f"❌ Error in chat collection: {str(e)}")
            return {'chats': pd.DataFrame(), 'entities': {}, 'incidents': []}


# Example usage
if __name__ == "__main__":
    sample_config = {'data_sources': {'chats': {'enabled': True}}}
    collector = ChatCollector(sample_config)
    result = collector.collect("chat.csv")
    print(f"Processed {result['total_messages']} messages from {result['unique_users']} users")

"""
Ticket Data Collector for SRE Incident Insight Engine
Handles loading and processing ticket data: ticket_id, created_at, status, summary
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
import re
from collections import Counter

class TicketCollector:
    """Collect and process ticket/issue data"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def load_ticket_data(self, file_path: str = None) -> pd.DataFrame:
        """
        Load ticket data from CSV file
        Expected columns: ticket_id, created_at, status, summary
        """
        try:
            if file_path is None:
                file_path = "tickets.csv"
            
            print(f"🎫 Loading ticket data from: {file_path}")
            
            # Load CSV data
            df = pd.read_csv(file_path)
            
            # Validate required columns
            required_cols = ['ticket_id', 'created_at', 'status', 'summary']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Convert created_at to datetime
            df['created_at'] = pd.to_datetime(df['created_at'])
            
            # Sort by created_at
            df = df.sort_values('created_at').reset_index(drop=True)
            
            # Clean data
            df['ticket_id'] = df['ticket_id'].astype(str)
            df['status'] = df['status'].fillna('unknown').astype(str).str.lower()
            df['summary'] = df['summary'].fillna('').astype(str)
            
            print(f"✅ Loaded {len(df)} tickets")
            return df
            
        except Exception as e:
            print(f"❌ Error loading ticket data: {str(e)}")
            return pd.DataFrame()
    
    def categorize_tickets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Categorize tickets by type and severity based on summary"""
        try:
            if df.empty:
                return df
            
            processed_df = df.copy()
            
            # Priority/Severity indicators
            priority_patterns = {
                'critical': r'critical|urgent|emergency|p0|sev1|down|outage',
                'high': r'high|important|p1|sev2|major|significant',
                'medium': r'medium|normal|p2|sev3|moderate',
                'low': r'low|minor|p3|p4|sev4|sev5|cosmetic'
            }
            
            # Initialize priority as 'medium' by default
            processed_df['priority'] = 'medium'
            
            # Assign priority based on summary content
            for priority, pattern in priority_patterns.items():
                mask = processed_df['summary'].str.contains(pattern, case=False, na=False)
                processed_df.loc[mask, 'priority'] = priority
            
            # Ticket type categorization
            type_patterns = {
                'incident': r'incident|outage|down|failed|not working|broken',
                'bug': r'bug|defect|error|issue|problem|incorrect',
                'performance': r'slow|performance|timeout|latency|response|speed',
                'security': r'security|vulnerability|breach|unauthorized|auth',
                'maintenance': r'maintenance|update|upgrade|patch|deployment',
                'request': r'request|feature|enhancement|improvement|change'
            }
            
            # Initialize type as 'request' by default
            processed_df['ticket_type'] = 'request'
            
            # Assign type based on summary content (priority order matters)
            type_priority_order = ['incident', 'security', 'bug', 'performance', 'maintenance', 'request']
            for ticket_type in type_priority_order:
                pattern = type_patterns[ticket_type]
                mask = processed_df['summary'].str.contains(pattern, case=False, na=False)
                processed_df.loc[mask, 'ticket_type'] = ticket_type
            
            # Status standardization
            status_mapping = {
                'open': ['open', 'new', 'created', 'submitted'],
                'in_progress': ['in progress', 'in-progress', 'assigned', 'working', 'active'],
                'resolved': ['resolved', 'fixed', 'completed', 'done'],
                'closed': ['closed', 'cancelled', 'rejected'],
                'blocked': ['blocked', 'waiting', 'hold', 'pending']
            }
            
            processed_df['status_standardized'] = processed_df['status']
            for standard_status, variations in status_mapping.items():
                for variation in variations:
                    mask = processed_df['status'].str.contains(variation, case=False, na=False)
                    processed_df.loc[mask, 'status_standardized'] = standard_status
            
            print("✅ Categorized tickets by priority, type, and standardized status")
            return processed_df
            
        except Exception as e:
            print(f"❌ Error categorizing tickets: {str(e)}")
            return df
    
    def calculate_ticket_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate ticket metrics and timing information"""
        try:
            if df.empty:
                return df
            
            processed_df = df.copy()
            
            # Time-based features
            processed_df['created_hour'] = processed_df['created_at'].dt.hour
            processed_df['created_day_of_week'] = processed_df['created_at'].dt.dayofweek
            processed_df['created_in_business_hours'] = processed_df['created_hour'].between(9, 17)
            processed_df['created_on_weekend'] = processed_df['created_day_of_week'].isin([5, 6])
            
            # Summary text analysis
            processed_df['summary_length'] = processed_df['summary'].str.len()
            processed_df['summary_word_count'] = processed_df['summary'].str.split().str.len()
            processed_df['has_technical_keywords'] = processed_df['summary'].str.contains(
                r'error|exception|fail|timeout|connection|database|server|api', case=False, na=False
            )
            
            # Age of tickets (assuming current time is the latest created_at if not resolved)
            current_time = processed_df['created_at'].max()
            processed_df['age_hours'] = (current_time - processed_df['created_at']).dt.total_seconds() / 3600
            processed_df['age_days'] = processed_df['age_hours'] / 24
            
            # Priority scoring (for ML models)
            priority_scores = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
            processed_df['priority_score'] = processed_df['priority'].map(priority_scores).fillna(2)
            
            # Type scoring
            type_scores = {'request': 1, 'maintenance': 2, 'performance': 3, 'bug': 4, 'security': 5, 'incident': 6}
            processed_df['type_score'] = processed_df['ticket_type'].map(type_scores).fillna(1)
            
            print("✅ Calculated ticket metrics and timing information")
            return processed_df
            
        except Exception as e:
            print(f"❌ Error calculating ticket metrics: {str(e)}")
            return df
    
    def analyze_ticket_patterns(self, df: pd.DataFrame) -> Dict[str, any]:
        """Analyze patterns in ticket creation and resolution"""
        try:
            if df.empty:
                return {}
            
            patterns = {}
            
            # Status distribution
            patterns['status_distribution'] = df['status_standardized'].value_counts().to_dict()
            
            # Priority distribution
            patterns['priority_distribution'] = df['priority'].value_counts().to_dict()
            
            # Type distribution
            patterns['type_distribution'] = df['ticket_type'].value_counts().to_dict()
            
            # Time patterns
            patterns['creation_by_hour'] = df.groupby('created_hour').size().to_dict()
            patterns['creation_by_day'] = df.groupby('created_day_of_week').size().to_dict()
            
            # Recent activity (last 24 hours)
            recent_cutoff = df['created_at'].max() - timedelta(days=1)
            recent_tickets = df[df['created_at'] >= recent_cutoff]
            patterns['recent_activity'] = {
                'total_tickets_24h': len(recent_tickets),
                'critical_tickets_24h': len(recent_tickets[recent_tickets['priority'] == 'critical']),
                'incident_tickets_24h': len(recent_tickets[recent_tickets['ticket_type'] == 'incident'])
            }
            
            # Average metrics
            patterns['averages'] = {
                'avg_summary_length': float(df['summary_length'].mean()),
                'avg_age_hours': float(df['age_hours'].mean()),
                'avg_priority_score': float(df['priority_score'].mean())
            }
            
            return patterns
            
        except Exception as e:
            print(f"❌ Error analyzing ticket patterns: {str(e)}")
            return {}
    
    def identify_critical_tickets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identify tickets that require immediate attention"""
        try:
            if df.empty:
                return df
            
            # Critical tickets are those that are:
            # 1. High/Critical priority AND open/in_progress
            # 2. Incident type AND not resolved
            # 3. Old tickets (>72 hours) that are still open
            
            critical_mask = (
                (df['priority'].isin(['high', 'critical']) & 
                 df['status_standardized'].isin(['open', 'in_progress'])) |
                (df['ticket_type'] == 'incident' & 
                 ~df['status_standardized'].isin(['resolved', 'closed'])) |
                (df['age_hours'] > 72 & 
                 df['status_standardized'].isin(['open', 'in_progress']))
            )
            
            critical_tickets = df[critical_mask].copy()
            
            if not critical_tickets.empty:
                # Add urgency score
                urgency_score = 0
                urgency_score += (critical_tickets['priority'] == 'critical') * 4
                urgency_score += (critical_tickets['priority'] == 'high') * 3
                urgency_score += (critical_tickets['ticket_type'] == 'incident') * 3
                urgency_score += (critical_tickets['ticket_type'] == 'security') * 2
                urgency_score += (critical_tickets['age_hours'] > 72) * 2
                urgency_score += (critical_tickets['age_hours'] > 168) * 1  # >1 week
                
                critical_tickets['urgency_score'] = urgency_score.clip(0, 10)
                critical_tickets = critical_tickets.sort_values('urgency_score', ascending=False)
            
            print(f"✅ Identified {len(critical_tickets)} critical tickets requiring attention")
            return critical_tickets
            
        except Exception as e:
            print(f"❌ Error identifying critical tickets: {str(e)}")
            return pd.DataFrame()
    
    def collect(self, file_path: str = None) -> Dict[str, any]:
        """Main collection method for ticket data"""
        try:
            # Load raw ticket data
            raw_data = self.load_ticket_data(file_path)
            if raw_data.empty:
                return {'tickets': pd.DataFrame(), 'patterns': {}, 'critical_tickets': pd.DataFrame()}
            
            # Categorize tickets
            processed_data = self.categorize_tickets(raw_data)
            
            # Calculate metrics
            processed_data = self.calculate_ticket_metrics(processed_data)
            
            # Analyze patterns
            patterns = self.analyze_ticket_patterns(processed_data)
            
            # Identify critical tickets
            critical_tickets = self.identify_critical_tickets(processed_data)
            
            return {
                'tickets': processed_data,
                'patterns': patterns,
                'critical_tickets': critical_tickets,
                'total_tickets': len(processed_data),
                'open_tickets': len(processed_data[processed_data['status_standardized'].isin(['open', 'in_progress'])]),
                'critical_count': len(critical_tickets),
                'incident_count': len(processed_data[processed_data['ticket_type'] == 'incident'])
            }
            
        except Exception as e:
            print(f"❌ Error in ticket collection: {str(e)}")
            return {'tickets': pd.DataFrame(), 'patterns': {}, 'critical_tickets': pd.DataFrame()}


# Example usage
if __name__ == "__main__":
    sample_config = {'data_sources': {'tickets': {'enabled': True}}}
    collector = TicketCollector(sample_config)
    result = collector.collect("tickets.csv")
    print(f"Processed {result['total_tickets']} tickets, {result['critical_count']} critical")

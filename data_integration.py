"""
Data Integrator for SRE Incident Insight Engine
Orchestrates collection and integration of all data sources:
- Logs (timestamp, level, message)
- Metrics (timestamp, cpu_util, memory_util, error_rate)  
- Chats (timestamp, user, message)
- Tickets (ticket_id, created_at, status, summary)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import asyncio
from pathlib import Path

from logs_collector import LogCollector
from metrics_collector import MetricsCollector
from chat_collector import ChatCollector
from tickets_collector import TicketCollector


class DataIntegrator:
    """Main class to integrate all data sources for the SRE Incident Insight Engine"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize collectors
        self.log_collector = LogCollector(config)
        self.metrics_collector = MetricsCollector(config)
        self.chat_collector = ChatCollector(config)
        self.ticket_collector = TicketCollector(config)
        
        print("🔗 Data Integrator initialized with all collectors")
    
    async def collect_all_data(self, data_paths: Dict[str, str] = None) -> Dict[str, any]:
        """
        Collect data from all sources concurrently
        
        Args:
            data_paths: Optional dictionary with custom file paths
                       {'logs': 'path/to/logs.csv', 'metrics': 'path/to/metrics.csv', ...}
        """
        try:
            print("📊 Starting parallel data collection from all sources...")
            
            # Default paths if not provided
            if data_paths is None:
                data_paths = {
                    'logs': 'logs.csv',
                    'metrics': 'metrics.csv',
                    'chats': 'chat.csv',
                    'tickets': 'tickets.csv'
                }
            
            # Collect data from all sources concurrently
            tasks = []
            
            # Note: Since the collectors are not async, we'll run them in executor
            loop = asyncio.get_event_loop()
            
            # Create tasks for each collector
            tasks.append(loop.run_in_executor(None, self.log_collector.collect, data_paths.get('logs')))
            tasks.append(loop.run_in_executor(None, self.metrics_collector.collect, data_paths.get('metrics')))
            tasks.append(loop.run_in_executor(None, self.chat_collector.collect, data_paths.get('chats')))
            tasks.append(loop.run_in_executor(None, self.ticket_collector.collect, data_paths.get('tickets')))
            
            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            collected_data = {}
            source_names = ['logs', 'metrics', 'chats', 'tickets']
            
            for i, (source, result) in enumerate(zip(source_names, results)):
                if isinstance(result, Exception):
                    print(f"❌ Error collecting {source} data: {result}")
                    collected_data[source] = self._get_empty_result(source)
                else:
                    collected_data[source] = result
                    print(f"✅ {source.capitalize()} data collected successfully")
            
            print("🎉 All data sources collected successfully!")
            return collected_data
            
        except Exception as e:
            print(f"❌ Error in parallel data collection: {str(e)}")
            return self._get_empty_results()
    
    def synchronous_collect_all_data(self, data_paths: Dict[str, str] = None) -> Dict[str, any]:
        """
        Collect data from all sources synchronously (for testing/debugging)
        """
        try:
            print("📊 Starting sequential data collection from all sources...")
            
            # Default paths if not provided
            if data_paths is None:
                data_paths = {
                    'logs': 'logs.csv',
                    'metrics': 'metrics.csv',
                    'chats': 'chat.csv',
                    'tickets': 'tickets.csv'
                }
            
            collected_data = {}
            
            # Collect logs
            print("\n1️⃣ Collecting log data...")
            collected_data['logs'] = self.log_collector.collect(data_paths.get('logs'))
            
            # Collect metrics
            print("\n2️⃣ Collecting metrics data...")
            collected_data['metrics'] = self.metrics_collector.collect(data_paths.get('metrics'))
            
            # Collect chats
            print("\n3️⃣ Collecting chat data...")
            collected_data['chats'] = self.chat_collector.collect(data_paths.get('chats'))
            
            # Collect tickets
            print("\n4️⃣ Collecting ticket data...")
            collected_data['tickets'] = self.ticket_collector.collect(data_paths.get('tickets'))
            
            print("\n🎉 All data sources collected successfully!")
            return collected_data
            
        except Exception as e:
            print(f"❌ Error in data collection: {str(e)}")
            return self._get_empty_results()
    
    def correlate_data_by_time(self, collected_data: Dict[str, any], time_window: int = 30) -> pd.DataFrame:
        """
        Correlate events across different data sources based on timestamps
        
        Args:
            collected_data: Dictionary containing all collected data
            time_window: Time window in minutes for correlation
        """
        try:
            print(f"🔗 Correlating data across sources with {time_window}-minute time window...")
            
            correlation_events = []
            
            # Get all dataframes with timestamps
            dfs_with_timestamps = {}
            
            if not collected_data['logs']['logs'].empty:
                logs_df = collected_data['logs']['logs'].copy()
                logs_df['source'] = 'logs'
                logs_df['event_type'] = logs_df['level']
                logs_df['event_description'] = logs_df['message']
                dfs_with_timestamps['logs'] = logs_df[['timestamp', 'source', 'event_type', 'event_description']]
            
            if not collected_data['metrics']['metrics'].empty:
                metrics_df = collected_data['metrics']['metrics'].copy()
                # Create events for anomalies and critical thresholds
                anomaly_mask = metrics_df.get('has_anomaly', pd.Series([False]))
                critical_cpu_mask = metrics_df.get('critical_cpu', pd.Series([False]))
                critical_memory_mask = metrics_df.get('critical_memory', pd.Series([False]))
                critical_errors_mask = metrics_df.get('critical_errors', pd.Series([False]))
                
                events_mask = anomaly_mask | critical_cpu_mask | critical_memory_mask | critical_errors_mask
                
                if events_mask.any():
                    metrics_events = metrics_df[events_mask].copy()
                    metrics_events['source'] = 'metrics'
                    metrics_events['event_type'] = 'metric_anomaly'
                    metrics_events['event_description'] = (
                        'CPU: ' + metrics_events['cpu_util'].astype(str) + '%, ' +
                        'Memory: ' + metrics_events['memory_util'].astype(str) + '%, ' +
                        'Errors: ' + metrics_events['error_rate'].astype(str)
                    )
                    dfs_with_timestamps['metrics'] = metrics_events[['timestamp', 'source', 'event_type', 'event_description']]
            
            if not collected_data['chats']['chats'].empty:
                chats_df = collected_data['chats']['chats'].copy()
                # Focus on incident-related messages
                incident_mask = chats_df.get('mentions_incident', pd.Series([False]))
                if incident_mask.any():
                    chat_events = chats_df[incident_mask].copy()
                    chat_events['source'] = 'chats'
                    chat_events['event_type'] = 'incident_discussion'
                    chat_events['event_description'] = chat_events['user'] + ': ' + chat_events['message']
                    dfs_with_timestamps['chats'] = chat_events[['timestamp', 'source', 'event_type', 'event_description']]
            
            if not collected_data['tickets']['tickets'].empty:
                tickets_df = collected_data['tickets']['tickets'].copy()
                tickets_df['source'] = 'tickets'
                tickets_df['event_type'] = tickets_df['ticket_type']
                tickets_df['event_description'] = tickets_df['ticket_id'] + ': ' + tickets_df['summary']
                tickets_df['timestamp'] = tickets_df['created_at']
                dfs_with_timestamps['tickets'] = tickets_df[['timestamp', 'source', 'event_type', 'event_description']]
            
            # Combine all events
            if dfs_with_timestamps:
                all_events = pd.concat(list(dfs_with_timestamps.values()), ignore_index=True)
                all_events = all_events.sort_values('timestamp').reset_index(drop=True)
                
                # Group events within time windows
                all_events['time_group'] = pd.cut(
                    all_events['timestamp'].astype(np.int64) // 10**9,  # Convert to seconds
                    bins=pd.interval_range(
                        start=all_events['timestamp'].min().value // 10**9,
                        end=all_events['timestamp'].max().value // 10**9,
                        freq=time_window * 60  # Convert minutes to seconds
                    ),
                    include_lowest=True
                )
                
                print(f"✅ Correlated {len(all_events)} events across {len(dfs_with_timestamps)} data sources")
                return all_events
            else:
                print("⚠️ No events found for correlation")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"❌ Error correlating data: {str(e)}")
            return pd.DataFrame()
    
    def generate_integration_summary(self, collected_data: Dict[str, any]) -> Dict[str, any]:
        """Generate a comprehensive summary of all collected data"""
        try:
            summary = {
                'collection_timestamp': datetime.now().isoformat(),
                'data_sources': {},
                'overall_health': {},
                'alerts': []
            }
            
            # Logs summary
            logs_data = collected_data.get('logs', {})
            summary['data_sources']['logs'] = {
                'total_entries': logs_data.get('total_entries', 0),
                'error_count': logs_data.get('error_count', 0),
                'warning_count': logs_data.get('warning_count', 0),
                'error_patterns': logs_data.get('error_patterns', {}),
                'status': 'success' if logs_data.get('total_entries', 0) > 0 else 'no_data'
            }
            
            # Metrics summary
            metrics_data = collected_data.get('metrics', {})
            summary['data_sources']['metrics'] = {
                'total_entries': metrics_data.get('total_entries', 0),
                'anomalies_detected': metrics_data.get('anomalies_detected', 0),
                'avg_health_score': round(metrics_data.get('avg_health_score', 0), 2),
                'outliers_detected': metrics_data.get('outliers_detected', 0),
                'status': 'success' if metrics_data.get('total_entries', 0) > 0 else 'no_data'
            }
            
            # Chats summary
            chats_data = collected_data.get('chats', {})
            summary['data_sources']['chats'] = {
                'total_messages': chats_data.get('total_messages', 0),
                'unique_users': chats_data.get('unique_users', 0),
                'incident_related_messages': chats_data.get('incident_related_messages', 0),
                'avg_urgency_score': round(chats_data.get('avg_urgency_score', 0), 2),
                'status': 'success' if chats_data.get('total_messages', 0) > 0 else 'no_data'
            }
            
            # Tickets summary
            tickets_data = collected_data.get('tickets', {})
            summary['data_sources']['tickets'] = {
                'total_tickets': tickets_data.get('total_tickets', 0),
                'open_tickets': tickets_data.get('open_tickets', 0),
                'critical_count': tickets_data.get('critical_count', 0),
                'incident_count': tickets_data.get('incident_count', 0),
                'status': 'success' if tickets_data.get('total_tickets', 0) > 0 else 'no_data'
            }
            
            # Overall health assessment
            health_score = metrics_data.get('avg_health_score', 50)
            error_rate = logs_data.get('error_count', 0) / max(logs_data.get('total_entries', 1), 1)
            critical_tickets = tickets_data.get('critical_count', 0)
            incident_chatter = chats_data.get('incident_related_messages', 0) / max(chats_data.get('total_messages', 1), 1)
            
            summary['overall_health'] = {
                'health_score': round(health_score, 2),
                'error_rate': round(error_rate * 100, 2),  # As percentage
                'critical_tickets': critical_tickets,
                'incident_chatter_rate': round(incident_chatter * 100, 2),  # As percentage
                'status': 'healthy' if health_score > 70 and critical_tickets == 0 else 'attention_needed'
            }
            
            # Generate alerts
            if critical_tickets > 0:
                summary['alerts'].append(f"🚨 {critical_tickets} critical tickets require immediate attention")
            
            if health_score < 50:
                summary['alerts'].append(f"⚠️ System health score is low: {health_score:.1f}/100")
            
            if error_rate > 0.1:  # >10% error rate
                summary['alerts'].append(f"🔥 High error rate detected: {error_rate*100:.1f}%")
            
            if incident_chatter > 0.2:  # >20% of messages are incident-related
                summary['alerts'].append(f"💬 High incident discussion activity: {incident_chatter*100:.1f}%")
            
            return summary
            
        except Exception as e:
            print(f"❌ Error generating integration summary: {str(e)}")
            return {'error': str(e)}
    
    def _get_empty_result(self, source: str) -> Dict[str, any]:
        """Get empty result structure for a specific source"""
        empty_results = {
            'logs': {'logs': pd.DataFrame(), 'error_patterns': {}, 'total_entries': 0, 'error_count': 0, 'warning_count': 0},
            'metrics': {'metrics': pd.DataFrame(), 'summary': {}, 'total_entries': 0, 'anomalies_detected': 0, 'avg_health_score': 0},
            'chats': {'chats': pd.DataFrame(), 'entities': {}, 'incident_discussions': [], 'total_messages': 0, 'unique_users': 0, 'incident_related_messages': 0, 'avg_urgency_score': 0},
            'tickets': {'tickets': pd.DataFrame(), 'patterns': {}, 'critical_tickets': pd.DataFrame(), 'total_tickets': 0, 'critical_count': 0, 'incident_count': 0}
        }
        return empty_results.get(source, {})
    
    def _get_empty_results(self) -> Dict[str, any]:
        """Get empty results for all sources"""
        return {
            'logs': self._get_empty_result('logs'),
            'metrics': self._get_empty_result('metrics'),
            'chats': self._get_empty_result('chats'),
            'tickets': self._get_empty_result('tickets')
        }


# Example usage and testing
if __name__ == "__main__":
    # Test the data integrator
    sample_config = {
        'data_sources': {
            'logs': {'enabled': True},
            'metrics': {'enabled': True},
            'chats': {'enabled': True},
            'tickets': {'enabled': True}
        }
    }
    
    integrator = DataIntegrator(sample_config)
    
    # Test synchronous collection (easier for testing)
    collected_data = integrator.synchronous_collect_all_data()
    
    # Generate correlation and summary
    correlation_df = integrator.correlate_data_by_time(collected_data)
    summary = integrator.generate_integration_summary(collected_data)
    
    print("\n📋 INTEGRATION SUMMARY:")
    print("=" * 50)
    for source, data in summary['data_sources'].items():
        print(f"{source.upper()}: {data}")
    
    print(f"\nOVERALL HEALTH: {summary['overall_health']}")
    
    if summary['alerts']:
        print("\n🚨 ALERTS:")
        for alert in summary['alerts']:
            print(f"  {alert}")

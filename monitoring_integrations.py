# 🔌 Real System Integrations
# Connect your website dashboard to real monitoring systems

import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class PrometheusIntegration:
    """Integration with Prometheus monitoring system"""
    
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
    
    def query(self, query: str, time: Optional[datetime] = None) -> Dict[str, Any]:
        """Execute Prometheus query"""
        try:
            params = {'query': query}
            if time:
                params['time'] = time.timestamp()
            
            response = self.session.get(
                f"{self.base_url}/api/v1/query",
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        
        except Exception as e:
            logger.error(f"Prometheus query failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def query_range(self, query: str, start: datetime, end: datetime, step: str = '1m') -> Dict[str, Any]:
        """Execute Prometheus range query"""
        try:
            params = {
                'query': query,
                'start': start.timestamp(),
                'end': end.timestamp(),
                'step': step
            }
            
            response = self.session.get(
                f"{self.base_url}/api/v1/query_range",
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        
        except Exception as e:
            logger.error(f"Prometheus range query failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        query = '100 - (avg(rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)'
        result = self.query(query)
        
        if result.get('status') == 'success' and result.get('data', {}).get('result'):
            return float(result['data']['result'][0]['value'][1])
        return 0.0
    
    def get_memory_usage(self) -> float:
        """Get current memory usage percentage"""
        query = '(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100'
        result = self.query(query)
        
        if result.get('status') == 'success' and result.get('data', {}).get('result'):
            return float(result['data']['result'][0]['value'][1])
        return 0.0
    
    def get_disk_usage(self) -> Dict[str, float]:
        """Get disk usage by filesystem"""
        query = '(1 - (node_filesystem_avail_bytes / node_filesystem_size_bytes)) * 100'
        result = self.query(query)
        
        disk_usage = {}
        if result.get('status') == 'success':
            for item in result.get('data', {}).get('result', []):
                filesystem = item['metric'].get('mountpoint', 'unknown')
                usage = float(item['value'][1])
                disk_usage[filesystem] = usage
        
        return disk_usage
    
    def get_network_io(self) -> Dict[str, float]:
        """Get network I/O rates"""
        queries = {
            'bytes_in': 'rate(node_network_receive_bytes_total[5m])',
            'bytes_out': 'rate(node_network_transmit_bytes_total[5m])'
        }
        
        network_io = {}
        for metric, query in queries.items():
            result = self.query(query)
            if result.get('status') == 'success' and result.get('data', {}).get('result'):
                total = sum(float(item['value'][1]) for item in result['data']['result'])
                network_io[metric] = total
        
        return network_io

class GrafanaIntegration:
    """Integration with Grafana for dashboards and alerts"""
    
    def __init__(self, base_url: str, api_key: str, timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def get_dashboards(self) -> List[Dict[str, Any]]:
        """Get list of Grafana dashboards"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/search",
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        
        except Exception as e:
            logger.error(f"Failed to get Grafana dashboards: {e}")
            return []
    
    def get_alerts(self) -> List[Dict[str, Any]]:
        """Get current Grafana alerts"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/alerts",
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        
        except Exception as e:
            logger.error(f"Failed to get Grafana alerts: {e}")
            return []
    
    def get_alert_notifications(self) -> List[Dict[str, Any]]:
        """Get alert notification channels"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/alert-notifications",
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        
        except Exception as e:
            logger.error(f"Failed to get alert notifications: {e}")
            return []

class DatadogIntegration:
    """Integration with Datadog monitoring service"""
    
    def __init__(self, api_key: str, app_key: str, site: str = "datadoghq.com"):
        self.api_key = api_key
        self.app_key = app_key
        self.base_url = f"https://api.{site}"
        self.headers = {
            'DD-API-KEY': api_key,
            'DD-APPLICATION-KEY': app_key,
            'Content-Type': 'application/json'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def get_metrics(self, query: str, from_time: datetime, to_time: datetime) -> Dict[str, Any]:
        """Query Datadog metrics"""
        try:
            params = {
                'query': query,
                'from': int(from_time.timestamp()),
                'to': int(to_time.timestamp())
            }
            
            response = self.session.get(
                f"{self.base_url}/api/v1/query",
                params=params
            )
            response.raise_for_status()
            return response.json()
        
        except Exception as e:
            logger.error(f"Datadog metrics query failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def get_monitors(self) -> List[Dict[str, Any]]:
        """Get Datadog monitors"""
        try:
            response = self.session.get(f"{self.base_url}/api/v1/monitor")
            response.raise_for_status()
            return response.json()
        
        except Exception as e:
            logger.error(f"Failed to get Datadog monitors: {e}")
            return []
    
    def get_events(self, start: datetime, end: datetime, priority: str = "normal") -> Dict[str, Any]:
        """Get Datadog events"""
        try:
            params = {
                'start': int(start.timestamp()),
                'end': int(end.timestamp()),
                'priority': priority
            }
            
            response = self.session.get(
                f"{self.base_url}/api/v1/events",
                params=params
            )
            response.raise_for_status()
            return response.json()
        
        except Exception as e:
            logger.error(f"Failed to get Datadog events: {e}")
            return {'status': 'error', 'error': str(e)}

class NewRelicIntegration:
    """Integration with New Relic monitoring service"""
    
    def __init__(self, api_key: str, account_id: str):
        self.api_key = api_key
        self.account_id = account_id
        self.base_url = "https://api.newrelic.com"
        self.headers = {
            'X-Api-Key': api_key,
            'Content-Type': 'application/json'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def nrql_query(self, query: str) -> Dict[str, Any]:
        """Execute NRQL query"""
        try:
            url = f"{self.base_url}/graphql"
            graphql_query = {
                'query': f'''
                {{
                    actor {{
                        account(id: {self.account_id}) {{
                            nrql(query: "{query}") {{
                                results
                            }}
                        }}
                    }}
                }}
                '''
            }
            
            response = self.session.post(url, json=graphql_query)
            response.raise_for_status()
            return response.json()
        
        except Exception as e:
            logger.error(f"New Relic NRQL query failed: {e}")
            return {'errors': [{'message': str(e)}]}
    
    def get_applications(self) -> List[Dict[str, Any]]:
        """Get New Relic applications"""
        try:
            response = self.session.get(f"{self.base_url}/v2/applications.json")
            response.raise_for_status()
            return response.json().get('applications', [])
        
        except Exception as e:
            logger.error(f"Failed to get New Relic applications: {e}")
            return []

class KubernetesIntegration:
    """Integration with Kubernetes cluster"""
    
    def __init__(self, api_server: str, token: str, verify_ssl: bool = True):
        self.api_server = api_server.rstrip('/')
        self.headers = {'Authorization': f'Bearer {token}'}
        self.verify_ssl = verify_ssl
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def get_pods(self, namespace: str = "default") -> Dict[str, Any]:
        """Get pods in namespace"""
        try:
            response = self.session.get(
                f"{self.api_server}/api/v1/namespaces/{namespace}/pods",
                verify=self.verify_ssl
            )
            response.raise_for_status()
            return response.json()
        
        except Exception as e:
            logger.error(f"Failed to get Kubernetes pods: {e}")
            return {'items': []}
    
    def get_nodes(self) -> Dict[str, Any]:
        """Get cluster nodes"""
        try:
            response = self.session.get(
                f"{self.api_server}/api/v1/nodes",
                verify=self.verify_ssl
            )
            response.raise_for_status()
            return response.json()
        
        except Exception as e:
            logger.error(f"Failed to get Kubernetes nodes: {e}")
            return {'items': []}
    
    def get_services(self, namespace: str = "default") -> Dict[str, Any]:
        """Get services in namespace"""
        try:
            response = self.session.get(
                f"{self.api_server}/api/v1/namespaces/{namespace}/services",
                verify=self.verify_ssl
            )
            response.raise_for_status()
            return response.json()
        
        except Exception as e:
            logger.error(f"Failed to get Kubernetes services: {e}")
            return {'items': []}

class ElasticSearchIntegration:
    """Integration with ElasticSearch for log analysis"""
    
    def __init__(self, host: str, port: int = 9200, username: str = None, password: str = None):
        self.base_url = f"http://{host}:{port}"
        self.auth = (username, password) if username and password else None
        self.session = requests.Session()
    
    def search(self, index: str, query: Dict[str, Any]) -> Dict[str, Any]:
        """Search ElasticSearch index"""
        try:
            response = self.session.post(
                f"{self.base_url}/{index}/_search",
                json=query,
                auth=self.auth
            )
            response.raise_for_status()
            return response.json()
        
        except Exception as e:
            logger.error(f"ElasticSearch query failed: {e}")
            return {'error': str(e)}
    
    def get_log_errors(self, index: str, hours: int = 1) -> List[Dict[str, Any]]:
        """Get error logs from last N hours"""
        query = {
            "query": {
                "bool": {
                    "must": [
                        {"match": {"level": "ERROR"}},
                        {"range": {"@timestamp": {"gte": f"now-{hours}h"}}}
                    ]
                }
            },
            "sort": [{"@timestamp": {"order": "desc"}}],
            "size": 100
        }
        
        result = self.search(index, query)
        return result.get('hits', {}).get('hits', [])

# Integration factory
class MonitoringIntegrations:
    """Factory class for managing multiple monitoring integrations"""
    
    def __init__(self, config: Dict[str, Dict[str, Any]]):
        self.integrations = {}
        self.config = config
        self._initialize_integrations()
    
    def _initialize_integrations(self):
        """Initialize configured integrations"""
        
        # Prometheus
        if 'prometheus' in self.config:
            try:
                self.integrations['prometheus'] = PrometheusIntegration(
                    base_url=self.config['prometheus']['url']
                )
            except Exception as e:
                logger.error(f"Failed to initialize Prometheus: {e}")
        
        # Grafana
        if 'grafana' in self.config:
            try:
                self.integrations['grafana'] = GrafanaIntegration(
                    base_url=self.config['grafana']['url'],
                    api_key=self.config['grafana']['api_key']
                )
            except Exception as e:
                logger.error(f"Failed to initialize Grafana: {e}")
        
        # Datadog
        if 'datadog' in self.config:
            try:
                self.integrations['datadog'] = DatadogIntegration(
                    api_key=self.config['datadog']['api_key'],
                    app_key=self.config['datadog']['app_key']
                )
            except Exception as e:
                logger.error(f"Failed to initialize Datadog: {e}")
        
        # New Relic
        if 'newrelic' in self.config:
            try:
                self.integrations['newrelic'] = NewRelicIntegration(
                    api_key=self.config['newrelic']['api_key'],
                    account_id=self.config['newrelic']['account_id']
                )
            except Exception as e:
                logger.error(f"Failed to initialize New Relic: {e}")
        
        # Kubernetes
        if 'kubernetes' in self.config:
            try:
                self.integrations['kubernetes'] = KubernetesIntegration(
                    api_server=self.config['kubernetes']['api_server'],
                    token=self.config['kubernetes']['token']
                )
            except Exception as e:
                logger.error(f"Failed to initialize Kubernetes: {e}")
        
        # ElasticSearch
        if 'elasticsearch' in self.config:
            try:
                self.integrations['elasticsearch'] = ElasticSearchIntegration(
                    host=self.config['elasticsearch']['host'],
                    port=self.config['elasticsearch'].get('port', 9200),
                    username=self.config['elasticsearch'].get('username'),
                    password=self.config['elasticsearch'].get('password')
                )
            except Exception as e:
                logger.error(f"Failed to initialize ElasticSearch: {e}")
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Collect metrics from all available integrations"""
        metrics = {
            'cpu_usage': 0,
            'memory_usage': 0,
            'disk_usage': {},
            'network_io': {},
            'alerts': [],
            'errors': [],
            'timestamp': datetime.now()
        }
        
        # Prometheus metrics
        if 'prometheus' in self.integrations:
            prom = self.integrations['prometheus']
            metrics['cpu_usage'] = prom.get_cpu_usage()
            metrics['memory_usage'] = prom.get_memory_usage()
            metrics['disk_usage'] = prom.get_disk_usage()
            metrics['network_io'] = prom.get_network_io()
        
        # Grafana alerts
        if 'grafana' in self.integrations:
            grafana = self.integrations['grafana']
            metrics['alerts'].extend(grafana.get_alerts())
        
        # Datadog monitors
        if 'datadog' in self.integrations:
            datadog = self.integrations['datadog']
            monitors = datadog.get_monitors()
            # Convert monitors to alert format
            for monitor in monitors:
                if monitor.get('overall_state') in ['Alert', 'Warn']:
                    alert = {
                        'id': monitor.get('id'),
                        'title': monitor.get('name'),
                        'message': monitor.get('message', ''),
                        'level': 'critical' if monitor['overall_state'] == 'Alert' else 'warning',
                        'source': 'Datadog',
                        'timestamp': datetime.now()
                    }
                    metrics['alerts'].append(alert)
        
        # ElasticSearch errors
        if 'elasticsearch' in self.integrations:
            es = self.integrations['elasticsearch']
            errors = es.get_log_errors('logs-*', hours=1)
            for error in errors[:10]:  # Limit to 10 recent errors
                error_data = error.get('_source', {})
                metrics['errors'].append({
                    'timestamp': error_data.get('@timestamp'),
                    'level': error_data.get('level', 'ERROR'),
                    'message': error_data.get('message', ''),
                    'service': error_data.get('service', 'unknown')
                })
        
        return metrics
    
    def health_check(self) -> Dict[str, bool]:
        """Check health of all integrations"""
        health = {}
        
        for name, integration in self.integrations.items():
            try:
                if name == 'prometheus':
                    result = integration.query('up')
                    health[name] = result.get('status') == 'success'
                elif name == 'grafana':
                    dashboards = integration.get_dashboards()
                    health[name] = isinstance(dashboards, list)
                elif name == 'datadog':
                    monitors = integration.get_monitors()
                    health[name] = isinstance(monitors, list)
                elif name == 'kubernetes':
                    nodes = integration.get_nodes()
                    health[name] = 'items' in nodes
                else:
                    health[name] = True  # Assume healthy if no specific check
            except Exception as e:
                logger.error(f"Health check failed for {name}: {e}")
                health[name] = False
        
        return health

# Sample configuration
SAMPLE_CONFIG = {
    'prometheus': {
        'url': 'http://prometheus:9090'
    },
    'grafana': {
        'url': 'http://grafana:3000',
        'api_key': 'your-grafana-api-key'
    },
    'datadog': {
        'api_key': 'your-datadog-api-key',
        'app_key': 'your-datadog-app-key'
    },
    'newrelic': {
        'api_key': 'your-newrelic-api-key',
        'account_id': 'your-account-id'
    },
    'kubernetes': {
        'api_server': 'https://kubernetes.default.svc',
        'token': 'your-service-account-token'
    },
    'elasticsearch': {
        'host': 'elasticsearch',
        'port': 9200,
        'username': 'elastic',
        'password': 'changeme'
    }
}

# Usage example for integration with your dashboard
def integrate_with_dashboard():
    """Example of how to integrate with your dashboard"""
    
    # Load configuration (from environment, config file, etc.)
    config = SAMPLE_CONFIG
    
    # Initialize integrations
    integrations = MonitoringIntegrations(config)
    
    # Get real metrics
    metrics = integrations.get_system_metrics()
    
    # Check integration health
    health = integrations.health_check()
    
    return {
        'metrics': metrics,
        'health': health,
        'integrations_available': list(integrations.integrations.keys())
    }

# Add this to your dashboard to use real data instead of mock data
def get_real_system_data():
    """Get real system data from integrations"""
    try:
        return integrate_with_dashboard()
    except Exception as e:
        logger.error(f"Failed to get real system data: {e}")
        return None
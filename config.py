"""
Configuration Management for SRE Incident Insight Engine
Integrates with your production ML system and handles Gemini API
"""

import os
from typing import Dict, Any, Optional
from datetime import datetime

class SREEngineConfig:
    """Complete configuration for SRE Incident Insight Engine"""
    
    def __init__(self):
        """Initialize all configurations"""
        
        # 🔑 API KEYS - Your Gemini API key is set here
        self.GEMINI_API_KEY = "AIzaSyCrWnYxNjlgfNf-ST0En7AvwPup1XUDJBs"
        
        # Override with environment variable if available
        env_key = os.getenv('GEMINI_API_KEY')
        if env_key:
            self.GEMINI_API_KEY = env_key
            print("✅ Using Gemini API key from environment variable")
        else:
            print("✅ Using Gemini API key from configuration")
        
        # 📧 NOTIFICATION SETTINGS (Optional)
        self.SLACK_TOKEN = os.getenv('SLACK_TOKEN', '')
        self.EMAIL_USERNAME = os.getenv('EMAIL_USERNAME', '')
        self.EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD', '')
        
        # 🤖 ML MODEL SETTINGS (Aligned with your ProductionMetaSystem)
        self.ml_config = {
            'anomaly_threshold': 0.7,
            'failure_threshold': 0.6, 
            'zero_day_threshold': 0.8,
            'meta_threshold': 0.5,  # For your production meta-model
            'correlation_time_window_minutes': 30,
            'confidence_threshold': 0.75,
            'model_dir': 'production_models'  # Same as your system
        }
        
        # 📝 NLP SETTINGS
        self.nlp_config = {
            'spacy_model': 'en_core_web_sm',
            'max_text_length': 5000,
            'similarity_threshold': 0.7,
            'entity_confidence_threshold': 0.8,
            'context_window_minutes': 30,
            'incident_keywords': [
                'error', 'fail', 'failure', 'down', 'outage', 'incident', 
                'alert', 'critical', 'urgent', 'problem', 'issue', 'crash', 
                'timeout', 'exception', 'bug', 'broken', 'unavailable',
                'degraded', 'slow', 'latency', 'spike', 'anomaly'
            ],
            'severity_keywords': {
                'critical': ['critical', 'p0', 'emergency', 'outage', 'down', 'severe'],
                'high': ['high', 'p1', 'urgent', 'major', 'important'],
                'medium': ['medium', 'p2', 'moderate', 'warning', 'concern'],
                'low': ['low', 'p3', 'minor', 'info', 'notice']
            },
            'system_components': {
                'servers': r'\b(?:server|host|node|instance|vm|machine)-?\w*\d*\w*\b',
                'services': r'\b(?:api|service|app|web|db|database|cache|queue|worker|nginx|apache)-?\w*\b',
                'databases': r'\b(?:mysql|postgres|postgresql|redis|mongo|mongodb|cassandra|elasticsearch|sql)\b',
                'networks': r'\b(?:load.?balancer|lb|cdn|proxy|gateway|router|switch|firewall)\b'
            }
        }
        
        # 🤖 GEMINI AI SETTINGS
        self.genai_config = {
            'model_name': 'gemini-1.5-flash',
            'temperature': 0.3,
            'max_output_tokens': 2000,
            'top_p': 0.8,
            'top_k': 40,
            'safety_settings_enabled': True
        }
        
        # 📊 SYSTEM SETTINGS
        self.system_config = {
            'debug_mode': os.getenv('DEBUG_MODE', 'False').lower() == 'true',
            'log_level': os.getenv('LOG_LEVEL', 'INFO'),
            'data_retention_days': int(os.getenv('DATA_RETENTION_DAYS', '30')),
            'max_concurrent_incidents': int(os.getenv('MAX_CONCURRENT_INCIDENTS', '10')),
            'notification_channels': ['console'],  # Add 'slack', 'email' when configured
            'production_mode': True  # Set to False for development
        }
        
        # Integration settings with your ProductionMetaSystem
        self.integration_config = {
            'use_production_models': True,
            'model_directory': 'production_models',
            'enable_meta_stacking': True,
            'classification_threshold': 0.5
        }
        
        print("⚙️ SRE Engine Configuration loaded successfully")
        print(f"   🔑 Gemini API: {'Configured' if self.GEMINI_API_KEY else 'Missing'}")
        print(f"   🤖 ML Models: {self.ml_config['model_dir']}")
        print(f"   📝 NLP Model: {self.nlp_config['spacy_model']}")
        print(f"   🎯 Production Mode: {self.system_config['production_mode']}")
    
    def validate_api_key(self) -> bool:
        """Validate Gemini API key is available"""
        return bool(self.GEMINI_API_KEY and len(self.GEMINI_API_KEY) > 10)
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary for debugging"""
        return {
            'api_keys_configured': {
                'gemini': bool(self.GEMINI_API_KEY),
                'slack': bool(self.SLACK_TOKEN),
                'email': bool(self.EMAIL_USERNAME and self.EMAIL_PASSWORD)
            },
            'ml_settings': self.ml_config,
            'nlp_settings': {k: v for k, v in self.nlp_config.items() if k != 'incident_keywords'},
            'genai_settings': self.genai_config,
            'system_settings': self.system_config,
            'config_loaded_at': datetime.now().isoformat()
        }

# Global configuration instance
config = SREEngineConfig()

# Export for easy imports
__all__ = ['config', 'SREEngineConfig']
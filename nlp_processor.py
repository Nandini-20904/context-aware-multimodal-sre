"""
Advanced NLP Context Processor for SRE Incident Insight Engine
Integrates with your ProductionMetaSystem and processes all text data
Uses spaCy, NLTK, and Sentence Transformers for comprehensive analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import logging
import re
import json

# NLP Libraries
try:
    import spacy
    import nltk
    from sentence_transformers import SentenceTransformer
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.stem import WordNetLemmatizer
    
    # Download required NLTK data silently
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print("📥 Downloading required NLTK data...")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    
    NLP_AVAILABLE = True
    print("✅ NLP libraries loaded successfully")
except ImportError as e:
    print(f"⚠️ NLP libraries not available: {e}")
    print("💡 Install with: pip install spacy nltk sentence-transformers")
    print("💡 Then run: python -m spacy download en_core_web_sm")
    NLP_AVAILABLE = False

from config import config

class ProductionNLPProcessor:
    """Production-grade NLP processor for SRE incident analysis"""
    
    def __init__(self):
        """Initialize NLP processor with production settings"""
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        print("🧠 Initializing Production NLP Processor...")
        
        if not NLP_AVAILABLE:
            print("❌ NLP libraries not available - using fallback processing")
            self._init_fallback_mode()
            return
        
        try:
            # Initialize spaCy model
            model_name = self.config.nlp_config['spacy_model']
            try:
                self.nlp_model = spacy.load(model_name)
                print(f"   ✅ spaCy model loaded: {model_name}")
            except OSError:
                print(f"   📥 Installing spaCy model: {model_name}")
                import subprocess
                subprocess.run([
                    "python", "-m", "spacy", "download", model_name
                ], capture_output=True)
                self.nlp_model = spacy.load(model_name)
            
            # Initialize sentence transformer for semantic similarity
            try:
                self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
                print("   ✅ Sentence Transformer loaded: all-MiniLM-L6-v2")
            except Exception as e:
                print(f"   ⚠️ Sentence Transformer failed: {e}")
                self.sentence_transformer = None
            
            # Initialize NLTK components
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
            
            # Initialize SRE-specific patterns
            self._init_sre_patterns()
            
            print("✅ Production NLP Processor initialized successfully")
            
        except Exception as e:
            print(f"❌ Error initializing NLP processor: {e}")
            self._init_fallback_mode()
    
    def _init_fallback_mode(self):
        """Initialize fallback mode when NLP libraries aren't available"""
        self.nlp_model = None
        self.sentence_transformer = None
        self.lemmatizer = None
        self.stop_words = set()
        print("⚠️ Running in fallback mode - limited NLP functionality")
    
    def _init_sre_patterns(self):
        """Initialize SRE-specific patterns and keywords"""
        
        # Get patterns from config
        self.severity_keywords = self.config.nlp_config['severity_keywords']
        self.incident_keywords = self.config.nlp_config['incident_keywords']
        self.system_components = self.config.nlp_config['system_components']
        
        # Error patterns for log analysis
        self.error_patterns = [
            r'(?i)(?:error|exception|failure|fault|crash|timeout|rejected|denied)',
            r'(?i)(?:50[0-9]|40[0-9]|timeout|connection.?refused|out.?of.?memory)',
            r'(?i)(?:disk.?full|no.?space|quota.?exceeded|permission.?denied)',
            r'(?i)(?:database.?connection|sql.?error|deadlock|constraint.?violation)',
            r'(?i)(?:null.?pointer|segmentation.?fault|stack.?overflow|heap.?overflow)'
        ]
        
        # Action patterns for extracting recommended actions
        self.action_patterns = [
            r'(?i)(?:restart|reboot|kill|stop|start|deploy|rollback|scale|investigate)',
            r'(?i)(?:check|monitor|alert|escalate|contact|notify|page|call)',
            r'(?i)(?:fix|patch|update|upgrade|downgrade|configure|tune)',
            r'(?i)(?:backup|restore|failover|switch|redirect|reroute)'
        ]
        
        # User mention patterns
        self.user_mention_patterns = [
            r'@(\w+)',  # @username
            r'(?i)(?:contact|call|notify|page|ask)\s+(\w+)',  # contact john
            r'(?i)(\w+)\s+(?:can|should|will|needs?\s+to)',  # john can handle
            r'(?i)(?:assign|assigned?\s+to)\s+(\w+)'  # assigned to jane
        ]
    
    def process_production_data(self, collected_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process production SRE data for context extraction
        Integrates with your ProductionMetaSystem data structure
        """
        
        print("🧠 Processing production data with advanced NLP...")
        
        try:
            nlp_results = {
                'processing_metadata': {
                    'processed_at': datetime.now().isoformat(),
                    'nlp_engine': 'production_nlp_processor',
                    'data_sources_processed': [],
                    'processing_mode': 'production' if self.nlp_model else 'fallback'
                },
                'incident_insights': {},
                'context_summary': {},
                'temporal_analysis': {},
                'entity_analysis': {},
                'severity_analysis': {},
                'action_extraction': {},
                'user_analysis': {}
            }
            
            # Process each data source
            processed_logs = self._process_logs_production(collected_data.get('logs', {}))
            processed_chats = self._process_chats_production(collected_data.get('chats', {}))
            processed_tickets = self._process_tickets_production(collected_data.get('tickets', {}))
            processed_metrics = self._process_metrics_context_production(collected_data.get('metrics', {}))
            
            # Record processed sources
            if processed_logs:
                nlp_results['processing_metadata']['data_sources_processed'].append('logs')
            if processed_chats:
                nlp_results['processing_metadata']['data_sources_processed'].append('chats')
            if processed_tickets:
                nlp_results['processing_metadata']['data_sources_processed'].append('tickets')
            if processed_metrics:
                nlp_results['processing_metadata']['data_sources_processed'].append('metrics')
            
            # Extract comprehensive incident insights
            nlp_results['incident_insights'] = self._extract_production_incident_insights(
                processed_logs, processed_chats, processed_tickets, processed_metrics
            )
            
            # Generate contextual summary
            nlp_results['context_summary'] = self._generate_production_context_summary(
                processed_logs, processed_chats, processed_tickets, processed_metrics
            )
            
            # Temporal analysis for timeline correlation
            nlp_results['temporal_analysis'] = self._perform_production_temporal_analysis(
                processed_logs, processed_chats, processed_tickets
            )
            
            # Entity analysis for system component identification
            nlp_results['entity_analysis'] = self._perform_production_entity_analysis(
                processed_logs, processed_chats, processed_tickets
            )
            
            # Severity analysis
            nlp_results['severity_analysis'] = self._perform_severity_analysis(
                processed_logs, processed_chats, processed_tickets
            )
            
            # Action extraction for recommendations
            nlp_results['action_extraction'] = self._extract_actions_and_recommendations(
                processed_logs, processed_chats, processed_tickets
            )
            
            # User and team analysis
            nlp_results['user_analysis'] = self._analyze_user_involvement(
                processed_chats, processed_tickets
            )
            
            print("✅ Production NLP processing completed successfully")
            print(f"   📊 Data sources processed: {len(nlp_results['processing_metadata']['data_sources_processed'])}")
            print(f"   🎯 Incident severity: {nlp_results['incident_insights'].get('incident_severity', 'unknown')}")
            print(f"   🔧 Components affected: {len(nlp_results['incident_insights'].get('affected_components', []))}")
            
            return nlp_results
            
        except Exception as e:
            print(f"❌ Error in production NLP processing: {e}")
            return self._generate_fallback_results(collected_data)
    
    def _process_logs_production(self, logs_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process logs with production-grade analysis"""
        
        if not logs_data or 'logs' not in logs_data:
            return {}
        
        logs_df = logs_data.get('logs', pd.DataFrame())
        if logs_df.empty:
            return {}
        
        try:
            print("   📝 Processing log data...")
            
            processed_logs = []
            error_patterns_found = Counter()
            severity_distribution = Counter()
            
            for _, log_row in logs_df.iterrows():
                message = str(log_row.get('message', ''))
                level = str(log_row.get('level', 'INFO'))
                timestamp = log_row.get('timestamp')
                
                if not message or message == 'nan':
                    continue
                
                # Clean and process message
                cleaned_message = self._clean_text(message)
                
                # Extract entities and keywords
                entities = self._extract_entities_from_text(message)
                keywords = self._extract_keywords(message)
                
                # Detect error patterns
                error_patterns = self._detect_error_patterns(message)
                for pattern in error_patterns:
                    error_patterns_found[pattern] += 1
                
                # Detect severity indicators
                severity_indicators = self._detect_severity_indicators(message)
                for severity in severity_indicators:
                    severity_distribution[severity] += 1
                
                # Calculate importance score
                importance_score = self._calculate_log_importance(message, level)
                
                processed_log = {
                    'original_message': message,
                    'cleaned_message': cleaned_message,
                    'level': level,
                    'timestamp': timestamp,
                    'entities': entities,
                    'keywords': keywords[:10],  # Top 10 keywords
                    'error_patterns': error_patterns,
                    'severity_indicators': severity_indicators,
                    'importance_score': importance_score,
                    'is_incident_related': self._is_incident_related(message),
                    'component_mentions': self._extract_component_mentions(message)
                }
                
                processed_logs.append(processed_log)
            
            # Generate log insights
            log_insights = {
                'total_logs': len(processed_logs),
                'error_logs': len([log for log in processed_logs if log['level'].upper() in ['ERROR', 'CRITICAL']]),
                'incident_related_logs': len([log for log in processed_logs if log['is_incident_related']]),
                'error_patterns_found': dict(error_patterns_found.most_common(10)),
                'severity_distribution': dict(severity_distribution),
                'avg_importance_score': np.mean([log['importance_score'] for log in processed_logs]) if processed_logs else 0,
                'top_components_mentioned': self._get_top_components(processed_logs)
            }
            
            return {
                'processed_logs': processed_logs,
                'log_insights': log_insights,
                'processing_status': 'success'
            }
            
        except Exception as e:
            print(f"   ❌ Error processing logs: {e}")
            return {'processing_status': 'failed', 'error': str(e)}
    
    def _process_chats_production(self, chats_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process chat messages with advanced context analysis"""
        
        if not chats_data or 'chats' not in chats_data:
            return {}
        
        chats_df = chats_data.get('chats', pd.DataFrame())
        if chats_df.empty:
            return {}
        
        try:
            print("   💬 Processing chat data...")
            
            processed_chats = []
            user_activity = Counter()
            urgency_scores = []
            action_mentions = Counter()
            
            for _, chat_row in chats_df.iterrows():
                message = str(chat_row.get('message', ''))
                user = str(chat_row.get('user', 'unknown'))
                timestamp = chat_row.get('timestamp')
                
                if not message or message == 'nan':
                    continue
                
                # Clean message
                cleaned_message = self._clean_text(message)
                
                # Extract entities and keywords
                entities = self._extract_entities_from_text(message)
                keywords = self._extract_keywords(message)
                
                # Calculate urgency and sentiment
                urgency_score = self._calculate_urgency_score(message)
                urgency_scores.append(urgency_score)
                
                sentiment_score = self._calculate_sentiment_score(message)
                
                # Extract user mentions and actions
                user_mentions = self._extract_user_mentions(message)
                actions_mentioned = self._detect_action_indicators(message)
                
                for action in actions_mentioned:
                    action_mentions[action] += 1
                
                # Check for incident discussion
                mentions_incident = self._is_incident_related(message)
                
                # Component and system mentions
                component_mentions = self._extract_component_mentions(message)
                
                user_activity[user] += 1
                
                processed_chat = {
                    'original_message': message,
                    'cleaned_message': cleaned_message,
                    'user': user,
                    'timestamp': timestamp,
                    'entities': entities,
                    'keywords': keywords[:10],
                    'urgency_score': urgency_score,
                    'sentiment_score': sentiment_score,
                    'mentions_incident': mentions_incident,
                    'user_mentions': user_mentions,
                    'actions_mentioned': actions_mentioned,
                    'component_mentions': component_mentions,
                    'is_escalation': self._is_escalation_message(message),
                    'is_resolution': self._is_resolution_message(message)
                }
                
                processed_chats.append(processed_chat)
            
            # Generate chat insights
            chat_insights = {
                'total_messages': len(processed_chats),
                'unique_users': len(user_activity),
                'incident_related_messages': len([chat for chat in processed_chats if chat['mentions_incident']]),
                'avg_urgency_score': np.mean(urgency_scores) if urgency_scores else 0,
                'most_active_users': dict(user_activity.most_common(5)),
                'top_actions_mentioned': dict(action_mentions.most_common(10)),
                'escalation_count': len([chat for chat in processed_chats if chat['is_escalation']]),
                'resolution_count': len([chat for chat in processed_chats if chat['is_resolution']]),
                'communication_timeline': self._build_communication_timeline(processed_chats)
            }
            
            return {
                'processed_chats': processed_chats,
                'chat_insights': chat_insights,
                'processing_status': 'success'
            }
            
        except Exception as e:
            print(f"   ❌ Error processing chats: {e}")
            return {'processing_status': 'failed', 'error': str(e)}
    
    def _process_tickets_production(self, tickets_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process tickets with comprehensive analysis"""
        
        if not tickets_data or 'tickets' not in tickets_data:
            return {}
        
        tickets_df = tickets_data.get('tickets', pd.DataFrame())
        if tickets_df.empty:
            return {}
        
        try:
            print("   🎫 Processing ticket data...")
            
            processed_tickets = []
            priority_distribution = Counter()
            status_distribution = Counter()
            
            for _, ticket_row in tickets_df.iterrows():
                summary = str(ticket_row.get('summary', ''))
                ticket_id = str(ticket_row.get('ticket_id', 'unknown'))
                status = str(ticket_row.get('status', 'unknown'))
                created_at = ticket_row.get('created_at')
                
                if not summary or summary == 'nan':
                    continue
                
                # Clean summary
                cleaned_summary = self._clean_text(summary)
                
                # Extract entities and keywords
                entities = self._extract_entities_from_text(summary)
                keywords = self._extract_keywords(summary)
                
                # Detect severity and priority
                severity_indicators = self._detect_severity_indicators(summary)
                priority_score = self._calculate_ticket_priority(summary, status)
                
                # Component analysis
                component_mentions = self._extract_component_mentions(summary)
                
                # Classification
                is_incident = self._is_incident_related(summary)
                ticket_category = self._classify_ticket_category(summary)
                
                status_distribution[status] += 1
                
                if severity_indicators:
                    for severity in severity_indicators:
                        priority_distribution[severity] += 1
                else:
                    priority_distribution['normal'] += 1
                
                processed_ticket = {
                    'ticket_id': ticket_id,
                    'summary': summary,
                    'cleaned_summary': cleaned_summary,
                    'status': status,
                    'created_at': created_at,
                    'entities': entities,
                    'keywords': keywords[:10],
                    'severity_indicators': severity_indicators,
                    'priority_score': priority_score,
                    'component_mentions': component_mentions,
                    'is_incident': is_incident,
                    'ticket_category': ticket_category,
                    'estimated_impact': self._estimate_ticket_impact(summary)
                }
                
                processed_tickets.append(processed_ticket)
            
            # Generate ticket insights
            ticket_insights = {
                'total_tickets': len(processed_tickets),
                'incident_tickets': len([ticket for ticket in processed_tickets if ticket['is_incident']]),
                'status_distribution': dict(status_distribution),
                'priority_distribution': dict(priority_distribution),
                'avg_priority_score': np.mean([ticket['priority_score'] for ticket in processed_tickets]) if processed_tickets else 0,
                'ticket_categories': Counter([ticket['ticket_category'] for ticket in processed_tickets]),
                'high_impact_tickets': len([ticket for ticket in processed_tickets if ticket['estimated_impact'] == 'high'])
            }
            
            return {
                'processed_tickets': processed_tickets,
                'ticket_insights': ticket_insights,
                'processing_status': 'success'
            }
            
        except Exception as e:
            print(f"   ❌ Error processing tickets: {e}")
            return {'processing_status': 'failed', 'error': str(e)}
    
    def _process_metrics_context_production(self, metrics_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process metrics for contextual information"""
        
        if not metrics_data or 'metrics' not in metrics_data:
            return {}
        
        metrics_df = metrics_data.get('metrics', pd.DataFrame())
        if metrics_df.empty:
            return {}
        
        try:
            print("   📊 Processing metrics context...")
            
            # Extract time range
            time_range = {
                'start': str(metrics_df['timestamp'].min()) if 'timestamp' in metrics_df.columns else 'unknown',
                'end': str(metrics_df['timestamp'].max()) if 'timestamp' in metrics_df.columns else 'unknown',
                'duration_minutes': 0
            }
            
            if 'timestamp' in metrics_df.columns:
                duration = (metrics_df['timestamp'].max() - metrics_df['timestamp'].min())
                time_range['duration_minutes'] = duration.total_seconds() / 60
            
            # Analyze each metric
            metrics_context = {
                'time_range': time_range,
                'cpu_analysis': self._analyze_metric(metrics_df, 'cpu_util', 80),
                'memory_analysis': self._analyze_metric(metrics_df, 'memory_util', 85),
                'error_analysis': self._analyze_metric(metrics_df, 'error_rate', 0.05),
                'overall_health': self._calculate_overall_health(metrics_df),
                'anomaly_periods': self._identify_anomaly_periods(metrics_df),
                'trend_analysis': self._analyze_trends(metrics_df)
            }
            
            return {
                'metrics_context': metrics_context,
                'processing_status': 'success'
            }
            
        except Exception as e:
            print(f"   ❌ Error processing metrics context: {e}")
            return {'processing_status': 'failed', 'error': str(e)}
    
    def _extract_production_incident_insights(self, processed_logs: Dict, processed_chats: Dict, 
                                            processed_tickets: Dict, processed_metrics: Dict) -> Dict[str, Any]:
        """Extract comprehensive incident insights for production system"""
        
        try:
            insights = {
                'incident_severity': 'unknown',
                'confidence_score': 0.0,
                'affected_components': [],
                'probable_causes': [],
                'recommended_actions': [],
                'people_involved': [],
                'timeline_summary': [],
                'business_impact': 'unknown',
                'escalation_needed': False
            }
            
            # Determine incident severity
            severity_scores = Counter()
            
            # From logs
            if processed_logs and 'log_insights' in processed_logs:
                log_insights = processed_logs['log_insights']
                for severity, count in log_insights.get('severity_distribution', {}).items():
                    severity_scores[severity] += count * 2  # Weight logs higher
            
            # From chats
            if processed_chats and 'processed_chats' in processed_chats:
                for chat in processed_chats['processed_chats']:
                    if chat.get('urgency_score', 0) > 0.7:
                        severity_scores['high'] += 1
                    elif chat.get('urgency_score', 0) > 0.5:
                        severity_scores['medium'] += 1
            
            # From tickets
            if processed_tickets and 'ticket_insights' in processed_tickets:
                ticket_insights = processed_tickets['ticket_insights']
                for severity, count in ticket_insights.get('priority_distribution', {}).items():
                    severity_scores[severity] += count
            
            # Determine final severity
            if severity_scores:
                top_severity = severity_scores.most_common(1)[0][0]
                insights['incident_severity'] = top_severity
                insights['confidence_score'] = min(1.0, severity_scores[top_severity] / sum(severity_scores.values()))
            
            # Extract affected components
            all_components = []
            
            for data_source in [processed_logs, processed_chats, processed_tickets]:
                if data_source and 'processed_logs' in data_source:
                    for item in data_source['processed_logs']:
                        all_components.extend(item.get('component_mentions', []))
                elif data_source and 'processed_chats' in data_source:
                    for item in data_source['processed_chats']:
                        all_components.extend(item.get('component_mentions', []))
                elif data_source and 'processed_tickets' in data_source:
                    for item in data_source['processed_tickets']:
                        all_components.extend(item.get('component_mentions', []))
            
            component_counts = Counter(all_components)
            insights['affected_components'] = [comp for comp, count in component_counts.most_common(10)]
            
            # Extract probable causes from logs
            if processed_logs and 'log_insights' in processed_logs:
                error_patterns = processed_logs['log_insights'].get('error_patterns_found', {})
                insights['probable_causes'] = list(error_patterns.keys())[:5]
            
            # Extract recommended actions
            all_actions = []
            
            if processed_chats and 'chat_insights' in processed_chats:
                chat_actions = processed_chats['chat_insights'].get('top_actions_mentioned', {})
                all_actions.extend(list(chat_actions.keys()))
            
            insights['recommended_actions'] = all_actions[:10]
            
            # Extract people involved
            if processed_chats and 'chat_insights' in processed_chats:
                active_users = processed_chats['chat_insights'].get('most_active_users', {})
                insights['people_involved'] = list(active_users.keys())[:10]
            
            # Business impact assessment
            insights['business_impact'] = self._assess_business_impact(
                insights['incident_severity'], insights['affected_components']
            )
            
            # Escalation decision
            insights['escalation_needed'] = (
                insights['incident_severity'] in ['critical', 'high'] or
                insights['business_impact'] == 'high' or
                len(insights['affected_components']) > 3
            )
            
            return insights
            
        except Exception as e:
            print(f"⚠️ Error extracting incident insights: {e}")
            return {
                'incident_severity': 'unknown',
                'confidence_score': 0.0,
                'affected_components': [],
                'probable_causes': [],
                'recommended_actions': ['Manual analysis required'],
                'people_involved': [],
                'business_impact': 'unknown',
                'escalation_needed': True
            }
    
    # Utility methods
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text or not isinstance(text, str):
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove timestamps and common log prefixes
        text = re.sub(r'^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}', '', text)
        text = re.sub(r'^\[?\w+\]?\s*', '', text)
        
        # Replace URLs and IPs with placeholders
        text = re.sub(r'http[s]?://\S+', '[URL]', text)
        text = re.sub(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', '[IP]', text)
        
        return text.strip()
    
    def _extract_entities_from_text(self, text: str) -> Dict[str, List[str]]:
        """Extract entities using spaCy NER"""
        entities = defaultdict(list)
        
        if not self.nlp_model or not text:
            return dict(entities)
        
        try:
            # Limit text length for performance
            text = text[:self.config.nlp_config['max_text_length']]
            doc = self.nlp_model(text)
            
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT', 'EVENT']:
                    entities[ent.label_].append(ent.text)
            
            # Pattern-based entity extraction for systems
            for comp_type, pattern in self.system_components.items():
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    entities[comp_type.upper()].extend(matches)
            
            # Clean and deduplicate
            for key in entities:
                entities[key] = list(set(entities[key]))
            
            return dict(entities)
            
        except Exception as e:
            print(f"⚠️ Error extracting entities: {e}")
            return {}
    
    def _extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """Extract keywords from text"""
        if not text or not self.nlp_model:
            return []
        
        try:
            doc = self.nlp_model(text[:self.config.nlp_config['max_text_length']])
            
            keywords = []
            for token in doc:
                if (not token.is_stop and not token.is_punct and not token.is_space 
                    and len(token.text) > 2 and token.pos_ in ['NOUN', 'VERB', 'ADJ']):
                    keywords.append(token.lemma_.lower())
            
            keyword_counts = Counter(keywords)
            return [word for word, count in keyword_counts.most_common(top_k)]
            
        except Exception as e:
            print(f"⚠️ Error extracting keywords: {e}")
            return []
    
    def _is_incident_related(self, text: str) -> bool:
        """Check if text is incident-related"""
        if not text:
            return False
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.incident_keywords)
    
    def _detect_severity_indicators(self, text: str) -> List[str]:
        """Detect severity indicators"""
        indicators = []
        text_lower = text.lower()
        
        for severity, keywords in self.severity_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                indicators.append(severity)
        
        return indicators
    
    def _detect_error_patterns(self, text: str) -> List[str]:
        """Detect error patterns"""
        patterns_found = []
        
        for pattern in self.error_patterns:
            if re.search(pattern, text):
                patterns_found.append(pattern)
        
        return patterns_found
    
    def _detect_action_indicators(self, text: str) -> List[str]:
        """Detect action indicators"""
        actions_found = []
        
        for pattern in self.action_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            actions_found.extend([match.lower() for match in matches])
        
        return list(set(actions_found))
    
    def _calculate_log_importance(self, message: str, level: str) -> float:
        """Calculate importance score for log message"""
        importance = 0.0
        
        # Base score from log level
        level_scores = {'ERROR': 0.8, 'CRITICAL': 1.0, 'WARNING': 0.5, 'INFO': 0.2, 'DEBUG': 0.1}
        importance += level_scores.get(level.upper(), 0.2)
        
        # Boost for incident-related content
        if self._is_incident_related(message):
            importance += 0.3
        
        # Boost for error patterns
        if self._detect_error_patterns(message):
            importance += 0.2
        
        # Boost for component mentions
        if self._extract_component_mentions(message):
            importance += 0.1
        
        return min(1.0, importance)
    
    def _calculate_urgency_score(self, message: str) -> float:
        """Calculate urgency score"""
        urgency = 0.0
        message_lower = message.lower()
        
        urgency_indicators = {
            'critical': 1.0, 'urgent': 0.9, 'emergency': 1.0, 'asap': 0.8,
            'immediately': 0.9, 'now': 0.7, 'quickly': 0.6, 'fast': 0.5,
            'help': 0.4, 'issue': 0.3, 'problem': 0.4, 'down': 0.8
        }
        
        for indicator, score in urgency_indicators.items():
            if indicator in message_lower:
                urgency = max(urgency, score)
        
        # Boost for severity indicators
        if self._detect_severity_indicators(message):
            urgency += 0.2
        
        return min(1.0, urgency)
    
    def _calculate_sentiment_score(self, message: str) -> float:
        """Calculate basic sentiment score"""
        positive_words = ['good', 'great', 'excellent', 'working', 'resolved', 'fixed', 'stable', 'success']
        negative_words = ['bad', 'terrible', 'broken', 'failing', 'down', 'error', 'problem', 'issue', 'crash']
        
        message_lower = message.lower()
        
        positive_count = sum(1 for word in positive_words if word in message_lower)
        negative_count = sum(1 for word in negative_words if word in message_lower)
        
        if positive_count + negative_count == 0:
            return 0.0
        
        return (positive_count - negative_count) / (positive_count + negative_count)
    
    def _extract_component_mentions(self, text: str) -> List[str]:
        """Extract system component mentions"""
        components = []
        
        for comp_type, pattern in self.system_components.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            components.extend([match.lower() for match in matches])
        
        return list(set(components))
    
    def _extract_user_mentions(self, text: str) -> List[str]:
        """Extract user mentions"""
        mentions = []
        
        for pattern in self.user_mention_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            mentions.extend([match.lower() for match in matches if isinstance(match, str)])
        
        return list(set(mentions))
    
    def _calculate_ticket_priority(self, summary: str, status: str) -> float:
        """Calculate ticket priority score"""
        priority = 0.3  # Base priority
        
        # Boost for incident-related content
        if self._is_incident_related(summary):
            priority += 0.4
        
        # Boost for severity indicators
        severity_indicators = self._detect_severity_indicators(summary)
        if 'critical' in severity_indicators:
            priority += 0.3
        elif 'high' in severity_indicators:
            priority += 0.2
        
        # Adjust for status
        if status.lower() in ['open', 'new', 'in-progress']:
            priority += 0.1
        
        return min(1.0, priority)
    
    # Additional utility methods
    def _get_top_components(self, processed_items: List[Dict]) -> List[str]:
        """Get top mentioned components"""
        all_components = []
        for item in processed_items:
            all_components.extend(item.get('component_mentions', []))
        
        component_counts = Counter(all_components)
        return [comp for comp, count in component_counts.most_common(10)]
    
    def _is_escalation_message(self, message: str) -> bool:
        """Check if message indicates escalation"""
        escalation_patterns = [
            r'(?i)escalat', r'(?i)manager', r'(?i)director', r'(?i)urgent',
            r'(?i)page\s+\w+', r'(?i)call\s+\w+', r'(?i)wake\s+up'
        ]
        
        return any(re.search(pattern, message) for pattern in escalation_patterns)
    
    def _is_resolution_message(self, message: str) -> bool:
        """Check if message indicates resolution"""
        resolution_patterns = [
            r'(?i)resolv', r'(?i)fix', r'(?i)work', r'(?i)back\s+up',
            r'(?i)normal', r'(?i)stable', r'(?i)clear', r'(?i)done'
        ]
        
        return any(re.search(pattern, message) for pattern in resolution_patterns)
    
    def _classify_ticket_category(self, summary: str) -> str:
        """Classify ticket category"""
        summary_lower = summary.lower()
        
        if any(word in summary_lower for word in ['outage', 'down', 'critical', 'emergency']):
            return 'incident'
        elif any(word in summary_lower for word in ['request', 'change', 'update', 'install']):
            return 'request'
        elif any(word in summary_lower for word in ['bug', 'error', 'issue', 'problem']):
            return 'bug'
        else:
            return 'general'
    
    def _estimate_ticket_impact(self, summary: str) -> str:
        """Estimate ticket impact"""
        summary_lower = summary.lower()
        
        high_impact_indicators = ['critical', 'outage', 'down', 'all users', 'production']
        medium_impact_indicators = ['slow', 'error', 'some users', 'degraded']
        
        if any(indicator in summary_lower for indicator in high_impact_indicators):
            return 'high'
        elif any(indicator in summary_lower for indicator in medium_impact_indicators):
            return 'medium'
        else:
            return 'low'
    
    def _build_communication_timeline(self, processed_chats: List[Dict]) -> List[Dict]:
        """Build communication timeline"""
        timeline = []
        
        for chat in processed_chats:
            if chat.get('mentions_incident') or chat.get('urgency_score', 0) > 0.5:
                timeline.append({
                    'timestamp': chat.get('timestamp'),
                    'user': chat.get('user'),
                    'message_summary': chat.get('cleaned_message', '')[:100] + '...',
                    'urgency': chat.get('urgency_score', 0),
                    'type': 'escalation' if chat.get('is_escalation') else 'resolution' if chat.get('is_resolution') else 'discussion'
                })
        
        return sorted(timeline, key=lambda x: x.get('timestamp', datetime.min))[:20]
    
    def _analyze_metric(self, df: pd.DataFrame, metric_col: str, threshold: float) -> Dict[str, Any]:
        """Analyze a specific metric"""
        if metric_col not in df.columns:
            return {'status': 'not_available'}
        
        try:
            values = df[metric_col].dropna()
            if values.empty:
                return {'status': 'no_data'}
            
            analysis = {
                'status': 'available',
                'current': float(values.iloc[-1]),
                'max': float(values.max()),
                'min': float(values.min()),
                'mean': float(values.mean()),
                'std': float(values.std()),
                'threshold': threshold,
                'threshold_breaches': len(values[values > threshold]),
                'trend': 'increasing' if len(values) > 1 and values.iloc[-1] > values.iloc[0] else 'stable'
            }
            
            return analysis
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _calculate_overall_health(self, df: pd.DataFrame) -> float:
        """Calculate overall system health score"""
        try:
            health_score = 100.0
            
            # Check CPU
            if 'cpu_util' in df.columns:
                cpu_max = df['cpu_util'].max()
                if cpu_max > 80:
                    health_score -= (cpu_max - 80) * 2
            
            # Check Memory
            if 'memory_util' in df.columns:
                mem_max = df['memory_util'].max()
                if mem_max > 85:
                    health_score -= (mem_max - 85) * 2
            
            # Check Errors
            if 'error_rate' in df.columns:
                error_max = df['error_rate'].max()
                if error_max > 0.05:
                    health_score -= error_max * 1000
            
            return max(0.0, min(100.0, health_score))
            
        except Exception:
            return 50.0  # Default health score
    
    def _identify_anomaly_periods(self, df: pd.DataFrame) -> List[Dict]:
        """Identify periods with anomalous behavior"""
        anomaly_periods = []
        
        try:
            # Simple anomaly detection based on thresholds
            if 'timestamp' in df.columns:
                anomalous_rows = df[
                    (df.get('cpu_util', 0) > 80) |
                    (df.get('memory_util', 0) > 85) |
                    (df.get('error_rate', 0) > 0.05)
                ]
                
                if not anomalous_rows.empty:
                    anomaly_periods.append({
                        'start': str(anomalous_rows['timestamp'].min()),
                        'end': str(anomalous_rows['timestamp'].max()),
                        'duration_minutes': (
                            anomalous_rows['timestamp'].max() - 
                            anomalous_rows['timestamp'].min()
                        ).total_seconds() / 60,
                        'affected_metrics': []
                    })
            
            return anomaly_periods
            
        except Exception:
            return []
    
    def _analyze_trends(self, df: pd.DataFrame) -> Dict[str, str]:
        """Analyze metric trends"""
        trends = {}
        
        try:
            for col in ['cpu_util', 'memory_util', 'error_rate']:
                if col in df.columns and len(df) > 1:
                    values = df[col].dropna()
                    if len(values) > 1:
                        if values.iloc[-1] > values.iloc[0] * 1.1:
                            trends[col] = 'increasing'
                        elif values.iloc[-1] < values.iloc[0] * 0.9:
                            trends[col] = 'decreasing'
                        else:
                            trends[col] = 'stable'
            
            return trends
            
        except Exception:
            return {}
    
    def _assess_business_impact(self, severity: str, components: List[str]) -> str:
        """Assess business impact"""
        if severity in ['critical'] or len(components) > 5:
            return 'high'
        elif severity in ['high'] or len(components) > 2:
            return 'medium'
        else:
            return 'low'
    
    # Main processing methods for generating summaries
    def _generate_production_context_summary(self, processed_logs: Dict, processed_chats: Dict, 
                                           processed_tickets: Dict, processed_metrics: Dict) -> Dict[str, Any]:
        """Generate production context summary"""
        return {
            'summary_generated_at': datetime.now().isoformat(),
            'data_quality': self._assess_data_quality(processed_logs, processed_chats, processed_tickets),
            'key_events': self._extract_key_events(processed_logs, processed_chats, processed_tickets),
            'communication_summary': self._summarize_communications(processed_chats),
            'system_status': self._summarize_system_status(processed_metrics),
            'incident_indicators': self._identify_incident_indicators(processed_logs, processed_chats, processed_tickets)
        }
    
    def _perform_production_temporal_analysis(self, processed_logs: Dict, processed_chats: Dict, processed_tickets: Dict) -> Dict[str, Any]:
        """Perform temporal analysis"""
        all_events = []
        
        # Extract timestamped events
        for source, data in [('logs', processed_logs), ('chats', processed_chats), ('tickets', processed_tickets)]:
            if data and f'processed_{source}' in data:
                for item in data[f'processed_{source}']:
                    timestamp = item.get('timestamp')
                    if timestamp:
                        all_events.append({
                            'timestamp': timestamp,
                            'source': source,
                            'content': item.get('cleaned_message' if source == 'chats' else 'cleaned_summary' if source == 'tickets' else 'cleaned_message', ''),
                            'importance': item.get('urgency_score' if source == 'chats' else 'priority_score' if source == 'tickets' else 'importance_score', 0)
                        })
        
        # Sort by timestamp
        all_events.sort(key=lambda x: x.get('timestamp', datetime.min))
        
        return {
            'total_events': len(all_events),
            'timeline': all_events[:50],  # Top 50 events
            'event_distribution': Counter([event['source'] for event in all_events]),
            'peak_activity_times': self._identify_peak_times(all_events)
        }
    
    def _perform_production_entity_analysis(self, processed_logs: Dict, processed_chats: Dict, processed_tickets: Dict) -> Dict[str, Any]:
        """Perform entity analysis"""
        all_entities = defaultdict(Counter)
        
        for source, data in [('logs', processed_logs), ('chats', processed_chats), ('tickets', processed_tickets)]:
            if data and f'processed_{source}' in data:
                for item in data[f'processed_{source}']:
                    entities = item.get('entities', {})
                    for entity_type, entity_list in entities.items():
                        for entity in entity_list:
                            all_entities[entity_type][entity] += 1
        
        return {
            'entity_types': list(all_entities.keys()),
            'top_entities_by_type': {
                entity_type: dict(counter.most_common(10))
                for entity_type, counter in all_entities.items()
            },
            'total_unique_entities': sum(len(counter) for counter in all_entities.values())
        }
    
    def _perform_severity_analysis(self, processed_logs: Dict, processed_chats: Dict, processed_tickets: Dict) -> Dict[str, Any]:
        """Perform severity analysis"""
        severity_indicators = Counter()
        
        for source, data in [('logs', processed_logs), ('chats', processed_chats), ('tickets', processed_tickets)]:
            if data and f'processed_{source}' in data:
                for item in data[f'processed_{source}']:
                    indicators = item.get('severity_indicators', [])
                    for indicator in indicators:
                        severity_indicators[indicator] += 1
        
        return {
            'severity_distribution': dict(severity_indicators),
            'dominant_severity': severity_indicators.most_common(1)[0][0] if severity_indicators else 'unknown',
            'severity_confidence': severity_indicators.most_common(1)[0][1] / sum(severity_indicators.values()) if severity_indicators else 0.0
        }
    
    def _extract_actions_and_recommendations(self, processed_logs: Dict, processed_chats: Dict, processed_tickets: Dict) -> Dict[str, Any]:
        """Extract actions and recommendations"""
        all_actions = []
        action_sources = Counter()
        
        for source, data in [('logs', processed_logs), ('chats', processed_chats), ('tickets', processed_tickets)]:
            if data and f'processed_{source}' in data:
                for item in data[f'processed_{source}']:
                    actions = item.get('actions_mentioned' if source == 'chats' else 'recommended_actions', [])
                    for action in actions:
                        all_actions.append({
                            'action': action,
                            'source': source,
                            'confidence': item.get('importance_score' if source == 'logs' else 'urgency_score' if source == 'chats' else 'priority_score', 0.5)
                        })
                        action_sources[source] += 1
        
        # Sort by confidence
        all_actions.sort(key=lambda x: x['confidence'], reverse=True)
        
        return {
            'recommended_actions': all_actions[:20],
            'action_sources': dict(action_sources),
            'top_actions': Counter([action['action'] for action in all_actions]).most_common(10)
        }
    
    def _analyze_user_involvement(self, processed_chats: Dict, processed_tickets: Dict) -> Dict[str, Any]:
        """Analyze user involvement"""
        user_activity = Counter()
        user_roles = {}
        
        if processed_chats and 'processed_chats' in processed_chats:
            for chat in processed_chats['processed_chats']:
                user = chat.get('user')
                if user and user != 'unknown':
                    user_activity[user] += 1
                    
                    # Infer role from activity
                    if chat.get('is_escalation'):
                        user_roles[user] = 'escalator'
                    elif chat.get('is_resolution'):
                        user_roles[user] = 'resolver'
                    elif chat.get('urgency_score', 0) > 0.7:
                        user_roles[user] = 'reporter'
                    else:
                        user_roles[user] = user_roles.get(user, 'participant')
        
        return {
            'active_users': dict(user_activity.most_common(10)),
            'user_roles': user_roles,
            'total_participants': len(user_activity),
            'engagement_level': 'high' if len(user_activity) > 5 else 'medium' if len(user_activity) > 2 else 'low'
        }
    
    # Utility methods for context generation
    def _assess_data_quality(self, *data_sources) -> Dict[str, Any]:
        """Assess data quality across sources"""
        quality = {
            'sources_available': 0,
            'total_records': 0,
            'quality_score': 0.0
        }
        
        for source in data_sources:
            if source and 'processing_status' in source and source['processing_status'] == 'success':
                quality['sources_available'] += 1
                
                # Count records
                for key in source.keys():
                    if key.startswith('processed_') and isinstance(source[key], list):
                        quality['total_records'] += len(source[key])
        
        quality['quality_score'] = min(1.0, quality['sources_available'] / 4.0)  # 4 sources max
        
        return quality
    
    def _extract_key_events(self, *data_sources) -> List[Dict]:
        """Extract key events across sources"""
        key_events = []
        
        for source in data_sources:
            if source and 'processing_status' in source and source['processing_status'] == 'success':
                for key in source.keys():
                    if key.startswith('processed_') and isinstance(source[key], list):
                        for item in source[key]:
                            importance = item.get('importance_score', 0) or item.get('urgency_score', 0) or item.get('priority_score', 0)
                            if importance > 0.6:  # High importance threshold
                                key_events.append({
                                    'timestamp': item.get('timestamp'),
                                    'content': item.get('cleaned_message', item.get('cleaned_summary', '')),
                                    'importance': importance,
                                    'source': key.replace('processed_', '')
                                })
        
        # Sort by importance and timestamp
        key_events.sort(key=lambda x: (x['importance'], x.get('timestamp', datetime.min)), reverse=True)
        
        return key_events[:20]
    
    def _summarize_communications(self, processed_chats: Dict) -> Dict[str, Any]:
        """Summarize communications"""
        if not processed_chats or 'chat_insights' in processed_chats:
            return {'status': 'no_communications'}
        
        insights = processed_chats.get('chat_insights', {})
        
        return {
            'total_messages': insights.get('total_messages', 0),
            'incident_related': insights.get('incident_related_messages', 0),
            'avg_urgency': insights.get('avg_urgency_score', 0),
            'escalations': insights.get('escalation_count', 0),
            'resolutions': insights.get('resolution_count', 0),
            'key_participants': list(insights.get('most_active_users', {}).keys())[:5]
        }
    
    def _summarize_system_status(self, processed_metrics: Dict) -> Dict[str, Any]:
        """Summarize system status"""
        if not processed_metrics or 'metrics_context' not in processed_metrics:
            return {'status': 'no_metrics'}
        
        context = processed_metrics['metrics_context']
        
        return {
            'overall_health': context.get('overall_health', 50),
            'cpu_status': context.get('cpu_analysis', {}).get('status', 'unknown'),
            'memory_status': context.get('memory_analysis', {}).get('status', 'unknown'),
            'error_status': context.get('error_analysis', {}).get('status', 'unknown'),
            'anomaly_periods': len(context.get('anomaly_periods', [])),
            'trend_analysis': context.get('trend_analysis', {})
        }
    
    def _identify_incident_indicators(self, *data_sources) -> List[str]:
        """Identify incident indicators"""
        indicators = []
        
        for source in data_sources:
            if source and 'processing_status' in source and source['processing_status'] == 'success':
                for key in source.keys():
                    if key.startswith('processed_') and isinstance(source[key], list):
                        for item in source[key]:
                            if item.get('is_incident_related') or item.get('mentions_incident'):
                                indicators.append(f"Incident activity detected in {key.replace('processed_', '')}")
                            
                            severity_indicators = item.get('severity_indicators', [])
                            if 'critical' in severity_indicators:
                                indicators.append(f"Critical severity indicator in {key.replace('processed_', '')}")
        
        return list(set(indicators))
    
    def _identify_peak_times(self, events: List[Dict]) -> List[Dict]:
        """Identify peak activity times"""
        if not events:
            return []
        
        # Group by hour
        hourly_activity = defaultdict(int)
        
        for event in events:
            timestamp = event.get('timestamp')
            if timestamp:
                hour = timestamp.replace(minute=0, second=0, microsecond=0)
                hourly_activity[hour] += 1
        
        # Find peak hours
        if not hourly_activity:
            return []
        
        avg_activity = sum(hourly_activity.values()) / len(hourly_activity)
        peak_times = []
        
        for hour, activity in hourly_activity.items():
            if activity > avg_activity * 1.5:  # 50% above average
                peak_times.append({
                    'time': hour.isoformat(),
                    'activity_count': activity,
                    'intensity': activity / avg_activity
                })
        
        return sorted(peak_times, key=lambda x: x['intensity'], reverse=True)[:5]
    
    def _generate_fallback_results(self, collected_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fallback results when NLP processing fails"""
        
        print("📝 Generating fallback NLP results...")
        
        return {
            'processing_metadata': {
                'processed_at': datetime.now().isoformat(),
                'method': 'fallback',
                'limitation': 'NLP libraries not available or processing failed'
            },
            'incident_insights': {
                'incident_severity': 'unknown',
                'confidence_score': 0.3,
                'affected_components': [],
                'probable_causes': ["Manual analysis required - NLP processing unavailable"],
                'recommended_actions': ["Review system logs manually", "Check system metrics", "Contact on-call engineer"],
                'people_involved': [],
                'business_impact': 'unknown',
                'escalation_needed': True
            },
            'context_summary': {
                'summary_generated_at': datetime.now().isoformat(),
                'limitation': 'Advanced context analysis unavailable'
            },
            'temporal_analysis': {'total_events': 0},
            'entity_analysis': {'entity_types': []},
            'severity_analysis': {'dominant_severity': 'unknown'},
            'action_extraction': {'recommended_actions': []},
            'user_analysis': {'total_participants': 0}
        }

# Example usage
if __name__ == "__main__":
    print("🧠 Testing Production NLP Processor...")
    
    processor = ProductionNLPProcessor()
    
    # Mock data for testing
    mock_collected_data = {
        'logs': {
            'logs': pd.DataFrame([
                {'timestamp': datetime.now(), 'level': 'ERROR', 'message': 'Database connection failed on server-01'},
                {'timestamp': datetime.now(), 'level': 'CRITICAL', 'message': 'API service unresponsive, high CPU usage detected'}
            ])
        },
        'chats': {
            'chats': pd.DataFrame([
                {'timestamp': datetime.now(), 'user': 'alice', 'message': 'Seeing critical database issues, need immediate help'},
                {'timestamp': datetime.now(), 'user': 'bob', 'message': 'API is down, customers complaining. @alice can you restart the service?'}
            ])
        },
        'tickets': {
            'tickets': pd.DataFrame([
                {'ticket_id': 'INC-001', 'created_at': datetime.now(), 'status': 'Open', 'summary': 'Critical: Database outage affecting all services'}
            ])
        },
        'metrics': {
            'metrics': pd.DataFrame([
                {'timestamp': datetime.now(), 'cpu_util': 95, 'memory_util': 87, 'error_rate': 0.15}
            ])
        }
    }
    
    # Process the data
    results = processor.process_production_data(mock_collected_data)
    
    print("✅ Production NLP processing completed!")
    print(f"📊 Incident severity: {results['incident_insights']['incident_severity']}")
    print(f"🎯 Confidence score: {results['incident_insights']['confidence_score']:.3f}")
    print(f"🔧 Affected components: {results['incident_insights']['affected_components']}")
    print(f"💡 Recommended actions: {len(results['incident_insights']['recommended_actions'])}")
    print(f"👥 People involved: {results['incident_insights']['people_involved']}")
    
    print(f"\n🔄 Ready to integrate with your ProductionMetaSystem!")
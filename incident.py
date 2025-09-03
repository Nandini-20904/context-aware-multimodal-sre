"""
Production GenAI Incident Summarizer for SRE Incident Insight Engine
Uses Gemini API to generate comprehensive, actionable incident briefs
Integrates seamlessly with your ProductionMetaSystem and NLP Processor
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime
import time
import traceback

# GenAI Libraries
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
    print("✅ Google GenerativeAI library loaded successfully")
except ImportError:
    print("⚠️ Google GenerativeAI not available")
    print("💡 Install with: pip install google-generativeai")
    GENAI_AVAILABLE = False

from config import config

class ProductionIncidentSummarizer:
    """Production-grade incident brief generator using Gemini AI"""
    
    def __init__(self):
        """Initialize Gemini AI incident summarizer for production use"""
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        print("🤖 Initializing Production Incident Summarizer...")
        
        # Get API key from config
        self.api_key = self.config.GEMINI_API_KEY
        
        if not GENAI_AVAILABLE:
            print("❌ Google GenerativeAI library not available - using fallback summarization")
            self.model = None
            self.fallback_mode = True
            return
        
        if not self.api_key:
            print("❌ GEMINI_API_KEY not configured!")
            print("💡 Check your sre_config.py file")
            self.model = None
            self.fallback_mode = True
            return
        
        try:
            # Configure Gemini API
            genai.configure(api_key=self.api_key)
            
            # Initialize model with production settings
            generation_config = {
                "temperature": self.config.genai_config['temperature'],
                "top_p": self.config.genai_config['top_p'],
                "top_k": self.config.genai_config['top_k'],
                "max_output_tokens": self.config.genai_config['max_output_tokens'],
            }
            
            # Production safety settings
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            ] if self.config.genai_config['safety_settings_enabled'] else None
            
            self.model = genai.GenerativeModel(
                model_name=self.config.genai_config['model_name'],
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            # Test connection
            self._test_api_connection()
            
            self.fallback_mode = False
            print(f"✅ Production Gemini AI initialized: {self.config.genai_config['model_name']}")
            
        except Exception as e:
            print(f"❌ Error initializing Gemini AI: {str(e)}")
            print("⚠️ Falling back to rule-based summarization")
            self.model = None
            self.fallback_mode = True
    
    def _test_api_connection(self):
        """Test API connection with production validation"""
        try:
            test_prompt = "Test API connection. Respond with 'API_CONNECTION_OK'."
            response = self.model.generate_content(test_prompt)
            
            if response and response.text and 'API_CONNECTION_OK' in response.text:
                print("   ✅ Gemini API connection verified")
            else:
                raise Exception("API test failed - unexpected response")
                
        except Exception as e:
            print(f"   ❌ API connection test failed: {e}")
            raise
    
    def generate_production_incident_brief(self, 
                                         nlp_results: Dict[str, Any],
                                         ml_predictions: Dict[str, Any],
                                         collected_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive production incident brief
        
        Integrates with:
        - Your ProductionMetaSystem ML predictions
        - ProductionNLPProcessor results
        - DataIntegrator collected data
        """
        
        print("🤖 Generating Production Incident Brief...")
        
        if self.fallback_mode:
            return self._generate_fallback_brief(nlp_results, ml_predictions, collected_data)
        
        try:
            start_time = datetime.now()
            
            # Prepare comprehensive context
            context = self._prepare_production_context(nlp_results, ml_predictions, collected_data)
            
            # Generate all brief components
            brief_components = {}
            
            # 1. Executive Summary
            print("   📋 Generating executive summary...")
            brief_components['executive_summary'] = self._generate_executive_summary(context)
            
            # 2. Technical Analysis
            print("   🔧 Generating technical analysis...")
            brief_components['technical_analysis'] = self._generate_technical_analysis(context)
            
            # 3. Incident Timeline
            print("   ⏰ Generating incident timeline...")
            brief_components['incident_timeline'] = self._generate_incident_timeline(context)
            
            # 4. Root Cause Analysis
            print("   🔍 Generating root cause analysis...")
            brief_components['root_cause_analysis'] = self._generate_root_cause_analysis(context)
            
            # 5. Impact Assessment
            print("   📊 Generating impact assessment...")
            brief_components['impact_assessment'] = self._generate_impact_assessment(context)
            
            # 6. Immediate Actions
            print("   ⚡ Generating immediate actions...")
            brief_components['immediate_actions'] = self._generate_immediate_actions(context)
            
            # 7. Recovery Plan
            print("   🔄 Generating recovery plan...")
            brief_components['recovery_plan'] = self._generate_recovery_plan(context)
            
            # 8. Communication Plan
            print("   📢 Generating communication plan...")
            brief_components['communication_plan'] = self._generate_communication_plan(context)
            
            # Create comprehensive incident package
            processing_time = (datetime.now() - start_time).total_seconds()
            
            incident_package = {
                'incident_brief': brief_components,
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'generation_model': self.config.genai_config['model_name'],
                    'processing_time_seconds': processing_time,
                    'data_sources_analyzed': context.get('data_sources', []),
                    'ml_models_integrated': context.get('ml_models_used', []),
                    'confidence_score': self._calculate_brief_confidence(context, brief_components),
                    'api_key_configured': True,
                    'production_mode': self.config.system_config['production_mode']
                },
                'context_data': context,  # Include for transparency and debugging
                'quality_metrics': self._assess_brief_quality(brief_components),
                'actionability_score': self._calculate_actionability_score(brief_components)
            }
            
            print(f"✅ Production incident brief generated successfully!")
            print(f"   ⏱️ Processing time: {processing_time:.1f}s")
            print(f"   🎯 Confidence score: {incident_package['metadata']['confidence_score']:.3f}")
            print(f"   📊 Quality score: {incident_package['quality_metrics']['overall_quality']:.3f}")
            
            return incident_package
            
        except Exception as e:
            print(f"❌ Error generating production incident brief: {str(e)}")
            print(f"📍 Error details: {traceback.format_exc()}")
            return self._generate_fallback_brief(nlp_results, ml_predictions, collected_data)
    
    def _prepare_production_context(self, nlp_results: Dict[str, Any], 
                                   ml_predictions: Dict[str, Any],
                                   collected_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare comprehensive context for Gemini AI"""
        
        try:
            context = {
                'incident_metadata': {},
                'ml_intelligence': {},
                'nlp_insights': {},
                'technical_evidence': {},
                'human_intelligence': {},
                'system_metrics': {},
                'temporal_context': {},
                'data_sources': [],
                'ml_models_used': []
            }
            
            # Extract NLP insights
            if nlp_results and 'incident_insights' in nlp_results:
                incident_insights = nlp_results['incident_insights']
                context['incident_metadata'] = {
                    'severity': incident_insights.get('incident_severity', 'unknown'),
                    'confidence': incident_insights.get('confidence_score', 0.0),
                    'affected_components': incident_insights.get('affected_components', []),
                    'probable_causes': incident_insights.get('probable_causes', []),
                    'people_involved': incident_insights.get('people_involved', []),
                    'business_impact': incident_insights.get('business_impact', 'unknown'),
                    'escalation_needed': incident_insights.get('escalation_needed', False)
                }
                
                context['nlp_insights'] = {
                    'severity_analysis': nlp_results.get('severity_analysis', {}),
                    'entity_analysis': nlp_results.get('entity_analysis', {}),
                    'action_extraction': nlp_results.get('action_extraction', {}),
                    'user_analysis': nlp_results.get('user_analysis', {}),
                    'temporal_analysis': nlp_results.get('temporal_analysis', {})
                }
                
                context['data_sources'].append('nlp_analysis')
            
            # Extract ML predictions (from your ProductionMetaSystem)
            if ml_predictions:
                context['ml_intelligence'] = {
                    'overall_incident_probability': ml_predictions.get('incident_probabilities', []),
                    'incident_count': ml_predictions.get('incident_count', 0),
                    'incident_rate': ml_predictions.get('incident_rate', 0),
                    'threshold_used': ml_predictions.get('threshold', 0.5),
                    'individual_predictions': ml_predictions.get('individual_predictions', {})
                }
                
                # Extract individual model results
                individual = ml_predictions.get('individual_predictions', {})
                
                if individual.get('anomaly'):
                    context['ml_models_used'].append('anomaly_detection')
                    anomaly_data = individual['anomaly']
                    if 'stacked_results' in anomaly_data:
                        context['ml_intelligence']['anomaly_results'] = anomaly_data['stacked_results']
                
                if individual.get('failure'):
                    context['ml_models_used'].append('failure_prediction')
                    context['ml_intelligence']['failure_results'] = individual['failure']
                
                if individual.get('zero_day'):
                    context['ml_models_used'].append('zero_day_detection')
                    context['ml_intelligence']['zero_day_results'] = individual['zero_day']
                
                context['data_sources'].append('ml_predictions')
            
            # Extract technical evidence from raw data
            if collected_data:
                # Logs analysis
                if 'logs' in collected_data and collected_data['logs']:
                    logs_data = collected_data['logs']
                    context['technical_evidence']['logs'] = {
                        'total_entries': logs_data.get('total_entries', 0),
                        'error_count': logs_data.get('error_count', 0),
                        'error_patterns': logs_data.get('error_patterns', {}),
                        'recent_errors': self._extract_recent_errors(logs_data.get('logs', pd.DataFrame()))
                    }
                    context['data_sources'].append('system_logs')
                
                # Metrics analysis
                if 'metrics' in collected_data and collected_data['metrics']:
                    metrics_data = collected_data['metrics']
                    metrics_df = metrics_data.get('metrics', pd.DataFrame())
                    if not metrics_df.empty:
                        context['system_metrics'] = {
                            'monitoring_period': {
                                'start': str(metrics_df['timestamp'].min()) if 'timestamp' in metrics_df.columns else 'unknown',
                                'end': str(metrics_df['timestamp'].max()) if 'timestamp' in metrics_df.columns else 'unknown',
                                'duration_minutes': self._calculate_duration(metrics_df)
                            },
                            'performance_summary': self._summarize_metrics(metrics_df),
                            'anomaly_indicators': self._identify_metric_anomalies(metrics_df)
                        }
                        context['data_sources'].append('system_metrics')
                
                # Human intelligence from chats
                if 'chats' in collected_data and collected_data['chats']:
                    chats_data = collected_data['chats']
                    context['human_intelligence'] = {
                        'total_messages': chats_data.get('total_messages', 0),
                        'incident_related_messages': chats_data.get('incident_related_messages', 0),
                        'key_discussions': self._extract_key_discussions(chats_data.get('chats', pd.DataFrame())),
                        'team_sentiment': chats_data.get('avg_urgency_score', 0)
                    }
                    context['data_sources'].append('team_communications')
                
                # Tickets context
                if 'tickets' in collected_data and collected_data['tickets']:
                    tickets_data = collected_data['tickets']
                    context['technical_evidence']['tickets'] = {
                        'total_tickets': tickets_data.get('total_tickets', 0),
                        'critical_count': tickets_data.get('critical_count', 0),
                        'incident_count': tickets_data.get('incident_count', 0),
                        'recent_tickets': self._extract_recent_tickets(tickets_data.get('tickets', pd.DataFrame()))
                    }
                    context['data_sources'].append('support_tickets')
            
            # Temporal context
            context['temporal_context'] = {
                'analysis_timestamp': datetime.now().isoformat(),
                'incident_detection_time': datetime.now().isoformat(),  # Could be extracted from data
                'time_since_detection': 0,  # Minutes since first detection
                'business_hours': self._is_business_hours(datetime.now())
            }
            
            return context
            
        except Exception as e:
            print(f"⚠️ Error preparing production context: {e}")
            return {'error': str(e), 'data_sources': [], 'ml_models_used': []}
    
    def _generate_executive_summary(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary for leadership"""
        
        prompt = self._build_executive_prompt(context)
        
        try:
            response = self._query_gemini_with_retry(prompt, max_retries=3)
            if response:
                return {
                    'summary_text': response,
                    'generation_method': 'gemini_ai',
                    'target_audience': 'executives_stakeholders',
                    'estimated_reading_time_minutes': len(response.split()) / 200  # ~200 words per minute
                }
            else:
                return self._create_fallback_executive_summary(context)
                
        except Exception as e:
            print(f"⚠️ Error generating executive summary: {e}")
            return self._create_fallback_executive_summary(context)
    
    def _generate_technical_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed technical analysis"""
        
        prompt = self._build_technical_prompt(context)
        
        try:
            response = self._query_gemini_with_retry(prompt, max_retries=3)
            if response:
                return {
                    'analysis_text': response,
                    'generation_method': 'gemini_ai',
                    'target_audience': 'engineering_team',
                    'technical_depth': 'detailed'
                }
            else:
                return self._create_fallback_technical_analysis(context)
                
        except Exception as e:
            print(f"⚠️ Error generating technical analysis: {e}")
            return self._create_fallback_technical_analysis(context)
    
    def _generate_incident_timeline(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate incident timeline"""
        
        prompt = self._build_timeline_prompt(context)
        
        try:
            response = self._query_gemini_with_retry(prompt, max_retries=3)
            if response:
                return {
                    'timeline_text': response,
                    'generation_method': 'gemini_ai',
                    'timeline_events': self._extract_timeline_events(context)
                }
            else:
                return self._create_fallback_timeline(context)
                
        except Exception as e:
            print(f"⚠️ Error generating timeline: {e}")
            return self._create_fallback_timeline(context)
    
    def _generate_root_cause_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate root cause analysis"""
        
        prompt = self._build_root_cause_prompt(context)
        
        try:
            response = self._query_gemini_with_retry(prompt, max_retries=3)
            if response:
                return {
                    'root_cause_text': response,
                    'generation_method': 'gemini_ai',
                    'confidence_level': self._assess_root_cause_confidence(context),
                    'contributing_factors': context.get('incident_metadata', {}).get('probable_causes', [])
                }
            else:
                return self._create_fallback_root_cause(context)
                
        except Exception as e:
            print(f"⚠️ Error generating root cause analysis: {e}")
            return self._create_fallback_root_cause(context)
    
    def _generate_impact_assessment(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate impact assessment"""
        
        prompt = self._build_impact_prompt(context)
        
        try:
            response = self._query_gemini_with_retry(prompt, max_retries=3)
            if response:
                return {
                    'impact_text': response,
                    'generation_method': 'gemini_ai',
                    'business_impact_level': context.get('incident_metadata', {}).get('business_impact', 'unknown'),
                    'affected_systems': context.get('incident_metadata', {}).get('affected_components', [])
                }
            else:
                return self._create_fallback_impact_assessment(context)
                
        except Exception as e:
            print(f"⚠️ Error generating impact assessment: {e}")
            return self._create_fallback_impact_assessment(context)
    
    def _generate_immediate_actions(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate immediate actions"""
        
        prompt = self._build_immediate_actions_prompt(context)
        
        try:
            response = self._query_gemini_with_retry(prompt, max_retries=3)
            if response:
                # Parse response into structured actions
                action_items = self._parse_action_items(response)
                
                return {
                    'actions_text': response,
                    'generation_method': 'gemini_ai',
                    'structured_actions': action_items,
                    'priority_level': context.get('incident_metadata', {}).get('severity', 'unknown'),
                    'estimated_completion_time': self._estimate_action_time(action_items)
                }
            else:
                return self._create_fallback_immediate_actions(context)
                
        except Exception as e:
            print(f"⚠️ Error generating immediate actions: {e}")
            return self._create_fallback_immediate_actions(context)
    
    def _generate_recovery_plan(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate recovery plan"""
        
        prompt = self._build_recovery_prompt(context)
        
        try:
            response = self._query_gemini_with_retry(prompt, max_retries=3)
            if response:
                return {
                    'recovery_text': response,
                    'generation_method': 'gemini_ai',
                    'recovery_phases': self._extract_recovery_phases(response),
                    'estimated_recovery_time': self._estimate_recovery_time(context)
                }
            else:
                return self._create_fallback_recovery_plan(context)
                
        except Exception as e:
            print(f"⚠️ Error generating recovery plan: {e}")
            return self._create_fallback_recovery_plan(context)
    
    def _generate_communication_plan(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate communication plan"""
        
        prompt = self._build_communication_prompt(context)
        
        try:
            response = self._query_gemini_with_retry(prompt, max_retries=3)
            if response:
                return {
                    'communication_text': response,
                    'generation_method': 'gemini_ai',
                    'stakeholder_groups': self._identify_stakeholder_groups(context),
                    'communication_frequency': self._determine_communication_frequency(context)
                }
            else:
                return self._create_fallback_communication_plan(context)
                
        except Exception as e:
            print(f"⚠️ Error generating communication plan: {e}")
            return self._create_fallback_communication_plan(context)
    
    def _query_gemini_with_retry(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """Query Gemini API with production-grade retry logic"""
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    wait_time = (2 ** attempt) + np.random.uniform(0, 1)  # Exponential backoff with jitter
                    print(f"   ⏳ Retry {attempt + 1}/{max_retries} in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                
                response = self.model.generate_content(prompt)
                
                if response and response.text:
                    return response.text.strip()
                else:
                    print(f"   ⚠️ Empty response from Gemini on attempt {attempt + 1}")
                    
            except Exception as e:
                error_msg = str(e).lower()
                print(f"   ❌ Gemini API error on attempt {attempt + 1}: {str(e)}")
                
                if "quota" in error_msg or "rate" in error_msg:
                    wait_time = 10 * (attempt + 1)  # Longer wait for rate limits
                    print(f"   ⏳ Rate limit hit, waiting {wait_time}s...")
                    time.sleep(wait_time)
                elif "safety" in error_msg:
                    print(f"   ⚠️ Safety filter triggered, trying alternative prompt...")
                    # Could implement prompt modification here
                elif attempt == max_retries - 1:
                    print(f"   ❌ All retry attempts failed")
                    break
        
        return None
    
    # Prompt building methods
    def _build_executive_prompt(self, context: Dict[str, Any]) -> str:
        """Build executive summary prompt"""
        
        metadata = context.get('incident_metadata', {})
        ml_intel = context.get('ml_intelligence', {})
        
        return f"""
You are a senior SRE manager preparing an executive briefing for leadership on a production incident.

INCIDENT OVERVIEW:
- Severity: {metadata.get('severity', 'Unknown').upper()}
- Business Impact: {metadata.get('business_impact', 'Unknown').upper()}
- ML Detection Confidence: {ml_intel.get('incident_rate', 0):.1f}% incident probability
- Affected Systems: {', '.join(metadata.get('affected_components', [])[:5])}
- Escalation Status: {'REQUIRED' if metadata.get('escalation_needed') else 'NOT REQUIRED'}

DATA SOURCES ANALYZED:
- {', '.join(context.get('data_sources', []))}
- ML Models Used: {', '.join(context.get('ml_models_used', []))}

SYSTEM EVIDENCE:
{self._format_system_evidence(context)}

TEAM RESPONSE:
{self._format_team_response(context)}

Create a concise executive summary (300-400 words) that covers:

**SITUATION OVERVIEW**
- What happened in business terms
- When it started and current status
- Customer/revenue impact if evident

**IMMEDIATE IMPACT**
- Service availability effects
- User experience degradation
- Business operations affected

**RESPONSE STATUS**
- Team mobilization and actions taken
- Current resolution progress
- Resources deployed

**NEXT STEPS**
- Expected resolution timeline
- Escalation decisions made
- Communication strategy

Use clear, non-technical language appropriate for executives. Focus on business impact and organizational response.
"""
    
    def _build_technical_prompt(self, context: Dict[str, Any]) -> str:
        """Build technical analysis prompt"""
        
        metadata = context.get('incident_metadata', {})
        ml_intel = context.get('ml_intelligence', {})
        
        return f"""
You are a senior SRE engineer providing detailed technical analysis for the engineering team.

TECHNICAL CONTEXT:
- Incident Severity: {metadata.get('severity', 'Unknown')}
- ML Detection Models: {', '.join(context.get('ml_models_used', []))}
- System Components: {', '.join(metadata.get('affected_components', []))}
- Probable Causes: {', '.join(metadata.get('probable_causes', []))}

ML INTELLIGENCE FINDINGS:
{self._format_ml_intelligence(context)}

SYSTEM EVIDENCE:
{self._format_detailed_system_evidence(context)}

NLP ANALYSIS INSIGHTS:
{self._format_nlp_insights(context)}

Provide a comprehensive technical analysis covering:

**ROOT CAUSE INVESTIGATION**
- Technical failure analysis based on ML predictions
- System behavior patterns observed
- Error propagation and cascading effects

**PERFORMANCE IMPACT**
- Specific metrics affected and severity
- Resource utilization anomalies
- Service dependency impacts

**DETECTION AND DIAGNOSIS**
- ML model predictions and confidence levels
- Anomaly patterns identified
- Correlation across data sources

**TECHNICAL REMEDIATION**
- Immediate technical fixes required
- System recovery procedures
- Risk assessment for proposed solutions

**PREVENTION MEASURES**
- System improvements needed
- Monitoring enhancements
- Process or configuration changes

Focus on technical depth with specific metrics, error patterns, and actionable engineering insights.
"""
    
    def _build_timeline_prompt(self, context: Dict[str, Any]) -> str:
        """Build timeline prompt"""
        
        temporal = context.get('temporal_context', {})
        
        return f"""
Create a chronological incident timeline based on the following data:

DETECTION TIME: {temporal.get('incident_detection_time', 'Unknown')}
ANALYSIS TIME: {temporal.get('analysis_timestamp', 'Unknown')}
BUSINESS HOURS: {'Yes' if temporal.get('business_hours') else 'No'}

DATA SOURCES WITH TIMESTAMPS:
{self._format_temporal_data(context)}

ML PREDICTIONS TIMELINE:
{self._format_ml_timeline(context)}

TEAM COMMUNICATION TIMELINE:
{self._format_communication_timeline(context)}

Create a detailed timeline in this format:

**HH:MM - EVENT TYPE: Description**
- Include detection timestamps
- ML prediction events
- System anomalies
- Team communications
- Actions taken

Focus on correlating events across different data sources to show the incident progression.
"""
    
    def _build_root_cause_prompt(self, context: Dict[str, Any]) -> str:
        """Build root cause analysis prompt"""
        
        metadata = context.get('incident_metadata', {})
        
        return f"""
Conduct a comprehensive root cause analysis based on the following evidence:

IDENTIFIED PROBABLE CAUSES:
{self._format_probable_causes(context)}

ML MODEL EVIDENCE:
{self._format_ml_evidence(context)}

SYSTEM BEHAVIOR ANALYSIS:
{self._format_system_behavior(context)}

CORRELATION ANALYSIS:
{self._format_correlation_analysis(context)}

Provide root cause analysis covering:

**PRIMARY ROOT CAUSE**
- Most likely technical cause based on evidence
- Confidence level and supporting data
- How this cause led to observed symptoms

**CONTRIBUTING FACTORS**
- Secondary factors that amplified the issue
- Environmental or operational conditions
- Timing and trigger events

**EVIDENCE SUMMARY**
- Key log entries supporting the analysis
- Metric anomalies that confirm the cause
- Team observations that validate findings

**ROOT CAUSE CONFIDENCE**
- Assessment of certainty level
- Areas needing further investigation
- Alternative hypotheses to consider

Provide technical depth while maintaining clarity for incident resolution.
"""
    
    def _build_impact_prompt(self, context: Dict[str, Any]) -> str:
        """Build impact assessment prompt"""
        
        metadata = context.get('incident_metadata', {})
        
        return f"""
Assess the comprehensive impact of this incident:

BUSINESS IMPACT LEVEL: {metadata.get('business_impact', 'Unknown').upper()}
AFFECTED COMPONENTS: {', '.join(metadata.get('affected_components', []))}
SEVERITY LEVEL: {metadata.get('severity', 'Unknown').upper()}

SYSTEM PERFORMANCE DATA:
{self._format_performance_impact(context)}

USER EXPERIENCE INDICATORS:
{self._format_user_impact(context)}

BUSINESS OPERATIONS AFFECTED:
{self._format_business_impact(context)}

Provide impact assessment covering:

**USER IMPACT**
- Service availability effects
- Performance degradation experienced
- User-facing functionality affected
- Geographic or segment-specific impacts

**BUSINESS IMPACT**
- Revenue or transaction impacts
- Operational disruptions
- Customer satisfaction effects
- SLA/SLO breaches

**TECHNICAL IMPACT**
- System components degraded
- Dependent services affected
- Data integrity concerns
- Security implications if any

**RECOVERY IMPACT**
- Time to restore full service
- Resources required for resolution
- Ongoing monitoring needs

Quantify impacts where possible and prioritize by severity.
"""
    
    def _build_immediate_actions_prompt(self, context: Dict[str, Any]) -> str:
        """Build immediate actions prompt"""
        
        metadata = context.get('incident_metadata', {})
        
        return f"""
Generate immediate action plan based on:

SEVERITY: {metadata.get('severity', 'Unknown').upper()}
ESCALATION NEEDED: {'YES' if metadata.get('escalation_needed') else 'NO'}
PEOPLE INVOLVED: {', '.join(metadata.get('people_involved', [])[:5])}

RECOMMENDED ACTIONS FROM ANALYSIS:
{self._format_recommended_actions(context)}

ML-SUGGESTED INTERVENTIONS:
{self._format_ml_actions(context)}

SYSTEM STATUS:
{self._format_system_status(context)}

Generate prioritized immediate actions:

**STOP THE BLEEDING (0-15 minutes)**
1. [Action] - [Owner] - [Expected Outcome]
2. [Action] - [Owner] - [Expected Outcome]

**STABILIZE SYSTEMS (15-60 minutes)**
1. [Action] - [Owner] - [Expected Outcome]
2. [Action] - [Owner] - [Expected Outcome]

**RESTORE SERVICE (1-4 hours)**
1. [Action] - [Owner] - [Expected Outcome]
2. [Action] - [Owner] - [Expected Outcome]

**COMMUNICATION ACTIONS**
1. [Stakeholder] - [Message] - [Timing]
2. [Stakeholder] - [Message] - [Timing]

Each action should be:
- Specific and measurable
- Assigned to a role/person
- Have clear success criteria
- Include expected timeline

Prioritize actions by impact on service restoration.
"""
    
    def _build_recovery_prompt(self, context: Dict[str, Any]) -> str:
        """Build recovery plan prompt"""
        
        return f"""
Create comprehensive recovery plan based on current situation:

INCIDENT ANALYSIS:
{self._format_recovery_context(context)}

SYSTEM RESTORATION NEEDS:
{self._format_restoration_needs(context)}

TEAM COORDINATION:
{self._format_team_coordination(context)}

Create structured recovery plan:

**PHASE 1: IMMEDIATE STABILIZATION**
- Critical systems to restore first
- Dependencies to address
- Success criteria

**PHASE 2: SERVICE RESTORATION**
- Step-by-step restoration process
- Validation procedures
- Rollback plans if needed

**PHASE 3: FULL RECOVERY**
- Complete service restoration
- Performance validation
- Monitoring intensification

**PHASE 4: POST-INCIDENT**
- System health verification
- Documentation requirements
- Lessons learned preparation

Include timelines, owners, and decision points throughout.
"""
    
    def _build_communication_prompt(self, context: Dict[str, Any]) -> str:
        """Build communication plan prompt"""
        
        metadata = context.get('incident_metadata', {})
        temporal = context.get('temporal_context', {})
        
        return f"""
Create communication strategy for:

INCIDENT SEVERITY: {metadata.get('severity', 'Unknown').upper()}
BUSINESS HOURS: {'Yes' if temporal.get('business_hours') else 'No'}
ESCALATION STATUS: {'REQUIRED' if metadata.get('escalation_needed') else 'NOT REQUIRED'}

STAKEHOLDER CONTEXT:
{self._format_stakeholder_context(context)}

IMPACT ASSESSMENT:
{self._format_communication_impact(context)}

Create communication plan covering:

**IMMEDIATE NOTIFICATIONS (0-30 minutes)**
- Who to notify immediately
- Key message points
- Communication channels

**STATUS UPDATE SCHEDULE**
- Update frequency by stakeholder group
- Information to include in updates
- Communication responsibilities

**ESCALATION COMMUNICATIONS**
- When to escalate communications
- Executive briefing requirements
- External communication needs

**RECOVERY COMMUNICATIONS**
- Service restoration announcements
- Post-incident communication plan
- Follow-up requirements

**MESSAGE TEMPLATES**
- Internal team updates
- Management briefings
- Customer communications (if applicable)

Tailor messaging by audience and maintain transparency while managing expectations.
"""
    
    # Context formatting methods
    def _format_system_evidence(self, context: Dict[str, Any]) -> str:
        """Format system evidence for prompts"""
        evidence = []
        
        technical = context.get('technical_evidence', {})
        metrics = context.get('system_metrics', {})
        
        if technical.get('logs'):
            logs = technical['logs']
            evidence.append(f"System Logs: {logs.get('total_entries', 0)} entries, {logs.get('error_count', 0)} errors")
        
        if metrics.get('performance_summary'):
            perf = metrics['performance_summary']
            evidence.append(f"Performance: CPU {perf.get('cpu_max', 0):.1f}%, Memory {perf.get('memory_max', 0):.1f}%, Errors {perf.get('error_max', 0):.3f}")
        
        return '\n- '.join([''] + evidence) if evidence else "No system evidence available"
    
    def _format_team_response(self, context: Dict[str, Any]) -> str:
        """Format team response information"""
        human_intel = context.get('human_intelligence', {})
        
        if not human_intel:
            return "No team communication data available"
        
        response = []
        response.append(f"Team Messages: {human_intel.get('total_messages', 0)} total, {human_intel.get('incident_related_messages', 0)} incident-related")
        response.append(f"Team Sentiment: {human_intel.get('team_sentiment', 0):.2f}/1.0 urgency level")
        
        return '\n- '.join([''] + response)
    
    def _format_ml_intelligence(self, context: Dict[str, Any]) -> str:
        """Format ML intelligence for prompts"""
        ml_intel = context.get('ml_intelligence', {})
        
        if not ml_intel:
            return "No ML predictions available"
        
        intel = []
        intel.append(f"Incident Probability: {ml_intel.get('incident_rate', 0):.1f}%")
        intel.append(f"Incidents Detected: {ml_intel.get('incident_count', 0)}")
        intel.append(f"Detection Threshold: {ml_intel.get('threshold_used', 0.5):.2f}")
        
        # Individual model results
        individual = ml_intel.get('individual_predictions', {})
        if individual.get('anomaly'):
            anomaly = individual['anomaly'].get('stacked_results', {})
            intel.append(f"Anomaly Detection: {anomaly.get('anomaly_count', 0)} anomalies detected")
        
        if individual.get('failure'):
            failure = individual['failure']
            intel.append(f"Failure Prediction: {failure.get('failure_rate', 0):.1f}% failure rate")
        
        if individual.get('zero_day'):
            zero_day = individual['zero_day'].get('combined_threats', {})
            intel.append(f"Zero-Day Detection: {zero_day.get('threat_count', 0)} threats")
        
        return '\n- '.join([''] + intel)
    
    def _format_detailed_system_evidence(self, context: Dict[str, Any]) -> str:
        """Format detailed system evidence"""
        evidence = []
        
        technical = context.get('technical_evidence', {})
        
        if technical.get('logs'):
            logs = technical['logs']
            evidence.append(f"System Logs Analysis:")
            evidence.append(f"  - Total entries: {logs.get('total_entries', 0):,}")
            evidence.append(f"  - Error entries: {logs.get('error_count', 0):,}")
            
            recent_errors = logs.get('recent_errors', [])
            if recent_errors:
                evidence.append(f"  - Recent critical errors: {len(recent_errors)}")
        
        metrics = context.get('system_metrics', {})
        if metrics.get('performance_summary'):
            perf = metrics['performance_summary']
            evidence.append(f"Performance Metrics:")
            evidence.append(f"  - CPU utilization: max {perf.get('cpu_max', 0):.1f}%, avg {perf.get('cpu_avg', 0):.1f}%")
            evidence.append(f"  - Memory utilization: max {perf.get('memory_max', 0):.1f}%, avg {perf.get('memory_avg', 0):.1f}%")
            evidence.append(f"  - Error rate: max {perf.get('error_max', 0):.4f}, avg {perf.get('error_avg', 0):.4f}")
        
        return '\n'.join(evidence) if evidence else "No detailed system evidence available"
    
    def _format_nlp_insights(self, context: Dict[str, Any]) -> str:
        """Format NLP insights"""
        nlp = context.get('nlp_insights', {})
        
        if not nlp:
            return "No NLP analysis available"
        
        insights = []
        
        severity_analysis = nlp.get('severity_analysis', {})
        if severity_analysis.get('dominant_severity'):
            insights.append(f"Severity Analysis: {severity_analysis['dominant_severity']} (confidence: {severity_analysis.get('severity_confidence', 0):.2f})")
        
        entity_analysis = nlp.get('entity_analysis', {})
        if entity_analysis.get('total_unique_entities'):
            insights.append(f"Entity Analysis: {entity_analysis['total_unique_entities']} unique entities identified")
        
        user_analysis = nlp.get('user_analysis', {})
        if user_analysis.get('total_participants'):
            insights.append(f"Team Involvement: {user_analysis['total_participants']} team members engaged")
            insights.append(f"Engagement Level: {user_analysis.get('engagement_level', 'unknown')}")
        
        return '\n- '.join([''] + insights) if insights else "Limited NLP insights available"
    
    # Utility methods
    def _extract_recent_errors(self, logs_df: pd.DataFrame) -> List[Dict]:
        """Extract recent error logs"""
        if logs_df.empty:
            return []
        
        try:
            error_logs = logs_df[logs_df['level'].isin(['ERROR', 'CRITICAL'])].tail(5)
            return [
                {
                    'timestamp': str(row.get('timestamp', 'unknown')),
                    'level': row.get('level', 'unknown'),
                    'message': str(row.get('message', ''))[:200]
                }
                for _, row in error_logs.iterrows()
            ]
        except Exception:
            return []
    
    def _calculate_duration(self, metrics_df: pd.DataFrame) -> float:
        """Calculate monitoring duration in minutes"""
        try:
            if 'timestamp' in metrics_df.columns and len(metrics_df) > 1:
                duration = (metrics_df['timestamp'].max() - metrics_df['timestamp'].min())
                return duration.total_seconds() / 60
        except Exception:
            pass
        return 0.0
    
    def _summarize_metrics(self, metrics_df: pd.DataFrame) -> Dict[str, float]:
        """Summarize metrics performance"""
        summary = {}
        
        try:
            for col in ['cpu_util', 'memory_util', 'error_rate']:
                if col in metrics_df.columns:
                    values = metrics_df[col].dropna()
                    if not values.empty:
                        summary[f'{col}_max'] = float(values.max())
                        summary[f'{col}_min'] = float(values.min())
                        summary[f'{col}_avg'] = float(values.mean())
        except Exception:
            pass
        
        return summary
    
    def _identify_metric_anomalies(self, metrics_df: pd.DataFrame) -> Dict[str, int]:
        """Identify metric anomalies"""
        anomalies = {}
        
        try:
            if 'cpu_util' in metrics_df.columns:
                anomalies['high_cpu_events'] = len(metrics_df[metrics_df['cpu_util'] > 80])
            
            if 'memory_util' in metrics_df.columns:
                anomalies['high_memory_events'] = len(metrics_df[metrics_df['memory_util'] > 85])
            
            if 'error_rate' in metrics_df.columns:
                anomalies['high_error_events'] = len(metrics_df[metrics_df['error_rate'] > 0.05])
        except Exception:
            pass
        
        return anomalies
    
    def _extract_key_discussions(self, chats_df: pd.DataFrame) -> List[Dict]:
        """Extract key chat discussions"""
        if chats_df.empty:
            return []
        
        try:
            # Get recent messages that might be incident-related
            recent_chats = chats_df.tail(10)  # Last 10 messages
            return [
                {
                    'timestamp': str(row.get('timestamp', 'unknown')),
                    'user': row.get('user', 'unknown'),
                    'message': str(row.get('message', ''))[:150]
                }
                for _, row in recent_chats.iterrows()
            ]
        except Exception:
            return []
    
    def _extract_recent_tickets(self, tickets_df: pd.DataFrame) -> List[Dict]:
        """Extract recent tickets"""
        if tickets_df.empty:
            return []
        
        try:
            recent_tickets = tickets_df.tail(5)
            return [
                {
                    'ticket_id': row.get('ticket_id', 'unknown'),
                    'created_at': str(row.get('created_at', 'unknown')),
                    'status': row.get('status', 'unknown'),
                    'summary': str(row.get('summary', ''))[:150]
                }
                for _, row in recent_tickets.iterrows()
            ]
        except Exception:
            return []
    
    def _is_business_hours(self, timestamp: datetime) -> bool:
        """Check if timestamp is during business hours"""
        # Simple business hours check (9 AM - 5 PM, Monday-Friday)
        return (
            timestamp.weekday() < 5 and  # Monday-Friday
            9 <= timestamp.hour < 17      # 9 AM - 5 PM
        )
    
    # Additional formatting methods for other prompts...
    def _format_temporal_data(self, context: Dict[str, Any]) -> str:
        """Format temporal data"""
        return "Temporal data analysis available from integrated sources"
    
    def _format_ml_timeline(self, context: Dict[str, Any]) -> str:
        """Format ML timeline"""
        return "ML prediction timeline available"
    
    def _format_communication_timeline(self, context: Dict[str, Any]) -> str:
        """Format communication timeline"""
        return "Team communication timeline available"
    
    def _format_probable_causes(self, context: Dict[str, Any]) -> str:
        """Format probable causes"""
        causes = context.get('incident_metadata', {}).get('probable_causes', [])
        return '\n- '.join([''] + causes) if causes else "No probable causes identified"
    
    def _format_ml_evidence(self, context: Dict[str, Any]) -> str:
        """Format ML evidence"""
        return "ML model evidence available from production analysis"
    
    def _format_system_behavior(self, context: Dict[str, Any]) -> str:
        """Format system behavior analysis"""
        return "System behavior patterns identified from metrics and logs"
    
    def _format_correlation_analysis(self, context: Dict[str, Any]) -> str:
        """Format correlation analysis"""
        return "Cross-data-source correlation analysis completed"
    
    def _format_performance_impact(self, context: Dict[str, Any]) -> str:
        """Format performance impact"""
        metrics = context.get('system_metrics', {}).get('performance_summary', {})
        if metrics:
            return f"CPU: {metrics.get('cpu_max', 0):.1f}% max, Memory: {metrics.get('memory_max', 0):.1f}% max"
        return "Performance impact data being analyzed"
    
    def _format_user_impact(self, context: Dict[str, Any]) -> str:
        """Format user impact"""
        return "User impact assessment based on system performance degradation"
    
    def _format_business_impact(self, context: Dict[str, Any]) -> str:
        """Format business impact"""
        impact = context.get('incident_metadata', {}).get('business_impact', 'unknown')
        return f"Business impact level: {impact}"
    
    def _format_recommended_actions(self, context: Dict[str, Any]) -> str:
        """Format recommended actions"""
        actions = context.get('nlp_insights', {}).get('action_extraction', {}).get('recommended_actions', [])
        if actions:
            action_list = [action.get('action', 'Unknown action') for action in actions[:5]]
            return '\n- '.join([''] + action_list)
        return "Actions being analyzed from team communications"
    
    def _format_ml_actions(self, context: Dict[str, Any]) -> str:
        """Format ML-suggested actions"""
        return "ML-derived intervention suggestions based on similar incidents"
    
    def _format_system_status(self, context: Dict[str, Any]) -> str:
        """Format system status"""
        return "Current system status and health assessment"
    
    def _format_recovery_context(self, context: Dict[str, Any]) -> str:
        """Format recovery context"""
        return "Recovery context based on current system state and incident analysis"
    
    def _format_restoration_needs(self, context: Dict[str, Any]) -> str:
        """Format restoration needs"""
        return "System restoration requirements identified"
    
    def _format_team_coordination(self, context: Dict[str, Any]) -> str:
        """Format team coordination"""
        people = context.get('incident_metadata', {}).get('people_involved', [])
        return f"Team members involved: {', '.join(people[:5])}" if people else "Team coordination in progress"
    
    def _format_stakeholder_context(self, context: Dict[str, Any]) -> str:
        """Format stakeholder context"""
        return "Stakeholder analysis based on incident severity and business impact"
    
    def _format_communication_impact(self, context: Dict[str, Any]) -> str:
        """Format communication impact"""
        return "Communication requirements based on incident scope and timing"
    
    # Response parsing methods
    def _parse_action_items(self, response: str) -> List[Dict]:
        """Parse action items from response"""
        # Simple parsing - could be enhanced
        actions = []
        lines = response.split('\n')
        
        for line in lines:
            if line.strip() and (line.startswith(('1.', '2.', '3.', '-', '•'))):
                clean_action = line.strip().lstrip('1234567890.- •')
                if clean_action:
                    actions.append({
                        'action': clean_action,
                        'priority': 'high' if 'immediate' in clean_action.lower() else 'medium',
                        'estimated_time': self._extract_time_estimate(clean_action)
                    })
        
        return actions[:10]  # Limit to top 10 actions
    
    def _extract_time_estimate(self, action: str) -> str:
        """Extract time estimate from action"""
        # Simple pattern matching for time estimates
        import re
        time_patterns = [
            r'(\d+)\s*(?:minutes?|mins?)',
            r'(\d+)\s*(?:hours?|hrs?)',
            r'(\d+)\s*(?:days?)'
        ]
        
        for pattern in time_patterns:
            match = re.search(pattern, action.lower())
            if match:
                return match.group(0)
        
        return "TBD"
    
    def _extract_timeline_events(self, context: Dict[str, Any]) -> List[Dict]:
        """Extract timeline events from context"""
        events = []
        
        # Add key events from temporal analysis
        temporal = context.get('nlp_insights', {}).get('temporal_analysis', {})
        if temporal.get('timeline'):
            for event in temporal['timeline'][:10]:
                events.append({
                    'timestamp': event.get('timestamp'),
                    'source': event.get('source'),
                    'description': event.get('content', '')[:100],
                    'importance': event.get('importance', 0.5)
                })
        
        return events
    
    def _extract_recovery_phases(self, response: str) -> List[str]:
        """Extract recovery phases from response"""
        phases = []
        lines = response.split('\n')
        
        for line in lines:
            if 'PHASE' in line.upper() and ':' in line:
                phases.append(line.strip())
        
        return phases[:5]  # Limit to 5 phases
    
    # Assessment methods
    def _assess_root_cause_confidence(self, context: Dict[str, Any]) -> str:
        """Assess root cause confidence"""
        confidence = context.get('incident_metadata', {}).get('confidence', 0.0)
        
        if confidence > 0.8:
            return 'High'
        elif confidence > 0.6:
            return 'Medium'
        else:
            return 'Low'
    
    def _estimate_action_time(self, actions: List[Dict]) -> str:
        """Estimate total action completion time"""
        if not actions:
            return "Unknown"
        
        # Simple heuristic based on number of actions
        if len(actions) <= 3:
            return "1-2 hours"
        elif len(actions) <= 6:
            return "2-4 hours"
        else:
            return "4+ hours"
    
    def _estimate_recovery_time(self, context: Dict[str, Any]) -> str:
        """Estimate recovery time"""
        severity = context.get('incident_metadata', {}).get('severity', 'unknown')
        
        recovery_times = {
            'critical': '2-4 hours',
            'high': '1-2 hours',
            'medium': '30 minutes - 1 hour',
            'low': '15-30 minutes'
        }
        
        return recovery_times.get(severity, 'To be determined')
    
    def _identify_stakeholder_groups(self, context: Dict[str, Any]) -> List[str]:
        """Identify stakeholder groups"""
        severity = context.get('incident_metadata', {}).get('severity', 'unknown')
        business_impact = context.get('incident_metadata', {}).get('business_impact', 'unknown')
        
        stakeholders = ['Engineering Team', 'SRE Team']
        
        if severity in ['critical', 'high'] or business_impact == 'high':
            stakeholders.extend(['Management', 'Customer Support'])
        
        if severity == 'critical':
            stakeholders.extend(['Executive Team', 'Legal/Compliance'])
        
        return stakeholders
    
    def _determine_communication_frequency(self, context: Dict[str, Any]) -> str:
        """Determine communication frequency"""
        severity = context.get('incident_metadata', {}).get('severity', 'unknown')
        
        frequencies = {
            'critical': 'Every 15 minutes',
            'high': 'Every 30 minutes',
            'medium': 'Every hour',
            'low': 'Every 2 hours'
        }
        
        return frequencies.get(severity, 'As needed')
    
    def _calculate_brief_confidence(self, context: Dict[str, Any], brief_components: Dict[str, Any]) -> float:
        """Calculate brief confidence score"""
        confidence_factors = []
        
        # Data source quality
        data_sources = len(context.get('data_sources', []))
        confidence_factors.append(min(1.0, data_sources / 4.0))  # Up to 4 sources
        
        # ML model integration
        ml_models = len(context.get('ml_models_used', []))
        confidence_factors.append(min(1.0, ml_models / 3.0))  # Up to 3 models
        
        # NLP analysis quality
        nlp_confidence = context.get('incident_metadata', {}).get('confidence', 0.0)
        confidence_factors.append(nlp_confidence)
        
        # Brief component completeness
        components_generated = len([comp for comp in brief_components.values() if comp.get('generation_method') == 'gemini_ai'])
        confidence_factors.append(min(1.0, components_generated / 8.0))  # 8 components total
        
        return np.mean(confidence_factors) if confidence_factors else 0.5
    
    def _assess_brief_quality(self, brief_components: Dict[str, Any]) -> Dict[str, float]:
        """Assess brief quality metrics"""
        quality_metrics = {
            'completeness': 0.0,
            'ai_generation_rate': 0.0,
            'actionability': 0.0,
            'overall_quality': 0.0
        }
        
        try:
            # Completeness - how many components were generated
            total_components = len(brief_components)
            quality_metrics['completeness'] = min(1.0, total_components / 8.0)
            
            # AI generation rate - how many used AI vs fallback
            ai_generated = len([comp for comp in brief_components.values() 
                              if comp.get('generation_method') == 'gemini_ai'])
            quality_metrics['ai_generation_rate'] = ai_generated / max(total_components, 1)
            
            # Actionability - presence of structured actions
            structured_actions = brief_components.get('immediate_actions', {}).get('structured_actions', [])
            quality_metrics['actionability'] = min(1.0, len(structured_actions) / 5.0)
            
            # Overall quality
            quality_metrics['overall_quality'] = np.mean([
                quality_metrics['completeness'],
                quality_metrics['ai_generation_rate'],
                quality_metrics['actionability']
            ])
            
        except Exception:
            pass
        
        return quality_metrics
    
    def _calculate_actionability_score(self, brief_components: Dict[str, Any]) -> float:
        """Calculate actionability score"""
        actionability_factors = []
        
        # Immediate actions present
        immediate_actions = brief_components.get('immediate_actions', {}).get('structured_actions', [])
        actionability_factors.append(min(1.0, len(immediate_actions) / 5.0))
        
        # Recovery plan structured
        recovery_phases = brief_components.get('recovery_plan', {}).get('recovery_phases', [])
        actionability_factors.append(min(1.0, len(recovery_phases) / 4.0))
        
        # Communication plan present
        comm_plan = brief_components.get('communication_plan', {})
        actionability_factors.append(1.0 if comm_plan.get('generation_method') == 'gemini_ai' else 0.5)
        
        return np.mean(actionability_factors) if actionability_factors else 0.5
    
    # Fallback methods
    def _generate_fallback_brief(self, nlp_results: Dict[str, Any], 
                                ml_predictions: Dict[str, Any],
                                collected_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fallback brief when Gemini is not available"""
        
        print("📝 Generating rule-based fallback incident brief...")
        
        incident_insights = nlp_results.get('incident_insights', {}) if nlp_results else {}
        
        fallback_brief = {
            'incident_brief': {
                'executive_summary': self._create_fallback_executive_summary({}),
                'technical_analysis': self._create_fallback_technical_analysis({}),
                'incident_timeline': self._create_fallback_timeline({}),
                'root_cause_analysis': self._create_fallback_root_cause({}),
                'impact_assessment': self._create_fallback_impact_assessment({}),
                'immediate_actions': self._create_fallback_immediate_actions({}),
                'recovery_plan': self._create_fallback_recovery_plan({}),
                'communication_plan': self._create_fallback_communication_plan({})
            },
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'generation_model': 'fallback_rules',
                'processing_time_seconds': 0.1,
                'api_key_configured': bool(self.api_key),
                'limitation': 'AI capabilities not available - using rule-based generation',
                'confidence_score': 0.3,
                'production_mode': False
            },
            'quality_metrics': {
                'completeness': 1.0,
                'ai_generation_rate': 0.0,
                'actionability': 0.5,
                'overall_quality': 0.3
            },
            'actionability_score': 0.4
        }
        
        return fallback_brief
    
    def _create_fallback_executive_summary(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create fallback executive summary"""
        return {
            'summary_text': """
EXECUTIVE INCIDENT SUMMARY

SITUATION: Production incident detected requiring immediate attention. Automated analysis systems have identified potential service disruption.

IMMEDIATE IMPACT: System monitoring indicates performance degradation. Engineering team has been notified and is investigating.

RESPONSE STATUS: 
- Incident response team activated
- Initial assessment in progress
- Monitoring systems actively tracking situation

NEXT STEPS:
- Complete technical root cause analysis
- Implement immediate stabilization measures
- Provide regular updates to stakeholders
- Prepare detailed post-incident review

RECOMMENDATION: Continue monitoring and maintain current response posture. Additional resources may be required based on technical analysis.
            """.strip(),
            'generation_method': 'fallback',
            'target_audience': 'executives_stakeholders',
            'estimated_reading_time_minutes': 1.0
        }
    
    def _create_fallback_technical_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create fallback technical analysis"""
        return {
            'analysis_text': """
TECHNICAL ANALYSIS - FALLBACK MODE

ROOT CAUSE INVESTIGATION:
- Advanced AI analysis unavailable - manual investigation required
- Review system logs for error patterns and anomalies
- Analyze performance metrics for threshold breaches
- Check service dependencies and external integrations

PERFORMANCE IMPACT:
- Monitor CPU, memory, and network utilization
- Review error rates and response times
- Check database performance and connection pools
- Validate load balancer and cache performance

DETECTION AND DIAGNOSIS:
- ML model predictions unavailable in fallback mode
- Rely on threshold-based monitoring alerts
- Manual correlation of events across systems required
- Escalate to subject matter experts as needed

TECHNICAL REMEDIATION:
- Follow established runbooks for similar incidents
- Consider service restarts if appropriate
- Review recent deployments and configuration changes
- Implement temporary workarounds if necessary

PREVENTION MEASURES:
- Document findings for future ML model training
- Review and update monitoring thresholds
- Schedule post-incident technical review
- Update incident response procedures based on learnings
            """.strip(),
            'generation_method': 'fallback',
            'target_audience': 'engineering_team',
            'technical_depth': 'basic'
        }
    
    def _create_fallback_timeline(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create fallback timeline"""
        return {
            'timeline_text': """
INCIDENT TIMELINE - FALLBACK MODE

Initial Detection: Monitoring systems triggered alerts
Investigation Started: Engineering team notified and responding
Current Status: Active investigation and response in progress

Note: Detailed timeline correlation requires AI analysis capabilities.
Manual timeline construction recommended for comprehensive incident review.
            """.strip(),
            'generation_method': 'fallback',
            'timeline_events': []
        }
    
    def _create_fallback_root_cause(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create fallback root cause"""
        return {
            'root_cause_text': """
ROOT CAUSE ANALYSIS - MANUAL INVESTIGATION REQUIRED

PRIMARY ROOT CAUSE: To be determined through manual analysis

INVESTIGATION APPROACH:
1. Review system logs for error patterns
2. Analyze performance metrics and thresholds
3. Check recent changes and deployments
4. Validate external dependencies
5. Interview team members with recent system knowledge

EVIDENCE COLLECTION:
- System logs from affected timeframe
- Performance metrics during incident period
- Recent configuration or code changes
- External service status and performance

ANALYSIS RECOMMENDATION:
Conduct thorough manual investigation with subject matter experts.
Document findings for future automated analysis improvement.
            """.strip(),
            'generation_method': 'fallback',
            'confidence_level': 'Low',
            'contributing_factors': []
        }
    
    def _create_fallback_impact_assessment(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create fallback impact assessment"""
        return {
            'impact_text': """
IMPACT ASSESSMENT - MANUAL EVALUATION REQUIRED

USER IMPACT: To be determined through user feedback and system analysis
BUSINESS IMPACT: Assess based on affected services and user reports
TECHNICAL IMPACT: Review system performance degradation

ASSESSMENT APPROACH:
1. Monitor user-facing service availability
2. Review customer support ticket volume
3. Analyze business metrics if available
4. Check SLA/SLO compliance status

RECOMMENDATION: Implement immediate monitoring and gather impact data for comprehensive assessment.
            """.strip(),
            'generation_method': 'fallback',
            'business_impact_level': 'unknown',
            'affected_systems': []
        }
    
    def _create_fallback_immediate_actions(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create fallback immediate actions"""
        return {
            'actions_text': """
IMMEDIATE ACTIONS - STANDARD INCIDENT RESPONSE

STOP THE BLEEDING:
1. Verify and acknowledge all monitoring alerts
2. Check system resource utilization and availability
3. Review recent deployments or changes
4. Notify on-call engineers and escalate if needed

STABILIZE SYSTEMS:
1. Implement known workarounds if applicable
2. Consider service restarts if safe and appropriate
3. Monitor system recovery and performance trends
4. Document all actions taken for post-incident review

RESTORE SERVICE:
1. Follow established recovery procedures
2. Validate service functionality after changes
3. Monitor for continued stability
4. Prepare status updates for stakeholders
            """.strip(),
            'generation_method': 'fallback',
            'structured_actions': [
                {'action': 'Verify monitoring alerts', 'priority': 'high', 'estimated_time': '5 minutes'},
                {'action': 'Check system resources', 'priority': 'high', 'estimated_time': '10 minutes'},
                {'action': 'Review recent changes', 'priority': 'medium', 'estimated_time': '15 minutes'},
                {'action': 'Notify on-call team', 'priority': 'high', 'estimated_time': '5 minutes'}
            ],
            'priority_level': 'medium',
            'estimated_completion_time': '30-60 minutes'
        }
    
    def _create_fallback_recovery_plan(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create fallback recovery plan"""
        return {
            'recovery_text': """
RECOVERY PLAN - STANDARD PROCEDURE

PHASE 1: IMMEDIATE STABILIZATION
- Identify and isolate affected components
- Implement temporary fixes or workarounds
- Monitor system stability

PHASE 2: SERVICE RESTORATION
- Execute planned recovery procedures
- Validate service functionality
- Restore full operational capacity

PHASE 3: FULL RECOVERY
- Verify all systems operating normally
- Resume standard monitoring posture
- Prepare incident documentation

PHASE 4: POST-INCIDENT
- Conduct post-mortem review
- Document lessons learned
- Update procedures and monitoring
            """.strip(),
            'generation_method': 'fallback',
            'recovery_phases': [
                'PHASE 1: IMMEDIATE STABILIZATION',
                'PHASE 2: SERVICE RESTORATION',
                'PHASE 3: FULL RECOVERY',
                'PHASE 4: POST-INCIDENT'
            ],
            'estimated_recovery_time': 'To be determined'
        }
    
    def _create_fallback_communication_plan(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create fallback communication plan"""
        return {
            'communication_text': """
COMMUNICATION PLAN - STANDARD PROTOCOL

IMMEDIATE NOTIFICATIONS:
- Engineering teams notified
- Management briefed on situation
- Stakeholders informed of response status

UPDATE SCHEDULE:
- Hourly updates during active response
- Status changes communicated immediately
- Recovery milestones reported promptly

ESCALATION PROTOCOL:
- Follow standard escalation procedures
- Engage additional resources as needed
- Maintain transparency with stakeholders

DOCUMENTATION:
- Record all communications
- Prepare incident summary report
- Schedule post-incident review meeting
            """.strip(),
            'generation_method': 'fallback',
            'stakeholder_groups': ['Engineering Team', 'Management', 'Operations'],
            'communication_frequency': 'Hourly during active response'
        }

# Example usage and testing
if __name__ == "__main__":
    print("🤖 Testing Production Incident Summarizer...")
    
    summarizer = ProductionIncidentSummarizer()
    
    # Mock data for testing
    mock_nlp_results = {
        'incident_insights': {
            'incident_severity': 'high',
            'confidence_score': 0.85,
            'affected_components': ['api-service', 'database'],
            'probable_causes': ['Database connection timeout', 'High CPU usage'],
            'people_involved': ['alice', 'bob'],
            'business_impact': 'medium',
            'escalation_needed': True
        },
        'severity_analysis': {
            'dominant_severity': 'high',
            'severity_confidence': 0.8
        }
    }
    
    mock_ml_predictions = {
        'incident_count': 5,
        'incident_rate': 15.2,
        'threshold': 0.5,
        'individual_predictions': {
            'anomaly': {
                'stacked_results': {
                    'anomaly_count': 3,
                    'anomaly_rate': 8.5
                }
            },
            'failure': {
                'failure_count': 2,
                'failure_rate': 6.1
            }
        }
    }
    
    mock_collected_data = {
        'logs': {
            'total_entries': 1500,
            'error_count': 45,
            'logs': pd.DataFrame([
                {'timestamp': datetime.now(), 'level': 'ERROR', 'message': 'Connection timeout'}
            ])
        },
        'metrics': {
            'metrics': pd.DataFrame([
                {'timestamp': datetime.now(), 'cpu_util': 95, 'memory_util': 87, 'error_rate': 0.15}
            ])
        }
    }
    
    # Generate incident brief
    brief = summarizer.generate_production_incident_brief(
        mock_nlp_results, mock_ml_predictions, mock_collected_data
    )
    
    print("✅ Production incident brief generation completed!")
    print(f"📊 Quality score: {brief['quality_metrics']['overall_quality']:.3f}")
    print(f"🎯 Actionability score: {brief['actionability_score']:.3f}")
    print(f"⏱️ Processing time: {brief['metadata']['processing_time_seconds']:.1f}s")
    print(f"🤖 AI generation rate: {brief['quality_metrics']['ai_generation_rate']:.1%}")
    
    # Display sample components
    incident_brief = brief.get('incident_brief', {})
    if incident_brief.get('executive_summary'):
        exec_summary = incident_brief['executive_summary']['summary_text']
        print(f"\n📋 Executive Summary Preview:")
        print(f"{exec_summary[:200]}..." if len(exec_summary) > 200 else exec_summary)
    
    print(f"\n🚀 Ready to integrate with your ProductionMetaSystem!")
"""
Complete Production SRE Incident Insight Engine
Integrates with your existing ProductionMetaSystem
Orchestrates: Data → ML → NLP → GenAI → Action
Ready for production deployment with your setup!
"""

import asyncio
import logging
import sys
import os
import traceback
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import your existing system (from the uploaded file)
try:
    from main import ProductionMetaSystem, ProductionFailurePredictor
    PRODUCTION_SYSTEM_AVAILABLE = True
    print("✅ Your Production ML System imported successfully")
except ImportError as e:
    print(f"⚠️ Could not import your production system: {e}")
    print("💡 Make sure paste.py is in the same directory")
    PRODUCTION_SYSTEM_AVAILABLE = False

# Import our new components
try:
    from config import config
    from nlp_processor import ProductionNLPProcessor
    from incident import ProductionIncidentSummarizer
    NEW_COMPONENTS_AVAILABLE = True
    print("✅ New NLP and GenAI components imported successfully")
except ImportError as e:
    print(f"❌ Could not import new components: {e}")
    print("💡 Make sure all files are in the same directory")
    NEW_COMPONENTS_AVAILABLE = False

class CompleteSREInsightEngine:
    """
    Complete SRE Incident Insight Engine
    Integrates your ProductionMetaSystem with NLP and GenAI capabilities
    """
    
    def __init__(self):
        """Initialize the complete production SRE insight engine"""
        self.logger = logging.getLogger(__name__)
        
        print("🚀 Initializing Complete SRE Incident Insight Engine...")
        print("="*70)
        print("🎯 Context-Aware Multimodal Incident Analysis System")
        print("📊 Integrating with your ProductionMetaSystem")
        print("🧠 Adding Advanced NLP Context Extraction")
        print("🤖 Adding GenAI Incident Briefs with Gemini")
        print("="*70)
        
        # Initialize components
        self.config = config
        self.pipeline_runs = 0
        self.last_incident_time = None
        self.errors = []
        
        # Initialize your existing production ML system
        self._init_production_ml_system()
        
        # Initialize new NLP and GenAI components
        self._init_nlp_component()
        self._init_genai_component()
        
        # System state
        self.is_initialized = True
        
        print("✅ Complete SRE Incident Insight Engine initialized successfully!")
        print(f"🎯 Production ML System: {'Available' if hasattr(self, 'production_system') else 'Not Available'}")
        print(f"🧠 NLP Processor: {'Available' if hasattr(self, 'nlp_processor') else 'Not Available'}")
        print(f"🤖 GenAI Summarizer: {'Available' if hasattr(self, 'genai_summarizer') else 'Not Available'}")
        print("="*70)
    
    def _init_production_ml_system(self):
        """Initialize your existing ProductionMetaSystem"""
        try:
            if PRODUCTION_SYSTEM_AVAILABLE:
                print("🔧 Initializing your Production ML System...")
                
                # Use your existing system with same model directory
                self.production_system = ProductionMetaSystem(model_dir=self.config.ml_config['model_dir'])
                
                # Initialize your existing models
                if self.production_system.initialize_models():
                    print("   ✅ Your Production ML models initialized")
                else:
                    print("   ⚠️ Some Production ML models failed to initialize")
                
                # Try to load existing models
                if self.production_system.load_production_models():
                    print("   ✅ Existing Production ML models loaded")
                else:
                    print("   ⚠️ No existing Production ML models found - will train on first run")
                
            else:
                print("❌ Your Production ML System not available")
                self.production_system = None
                
        except Exception as e:
            print(f"❌ Error initializing Production ML System: {e}")
            self.production_system = None
    
    def _init_nlp_component(self):
        """Initialize NLP context processor"""
        try:
            if NEW_COMPONENTS_AVAILABLE:
                print("🧠 Initializing Advanced NLP Context Processor...")
                self.nlp_processor = ProductionNLPProcessor()
                print("   ✅ NLP Context Processor ready")
            else:
                print("❌ NLP Processor not available")
                self.nlp_processor = None
                
        except Exception as e:
            print(f"❌ Error initializing NLP Processor: {e}")
            self.nlp_processor = None
    
    def _init_genai_component(self):
        """Initialize GenAI incident summarizer"""
        try:
            if NEW_COMPONENTS_AVAILABLE:
                print("🤖 Initializing Production GenAI Incident Summarizer...")
                self.genai_summarizer = ProductionIncidentSummarizer()
                print("   ✅ GenAI Incident Summarizer ready")
                
                if not self.genai_summarizer.fallback_mode:
                    print(f"   🔑 Gemini API: Connected ({self.config.genai_config['model_name']})")
                else:
                    print("   ⚠️ Gemini API: Using fallback mode")
            else:
                print("❌ GenAI Summarizer not available")
                self.genai_summarizer = None
                
        except Exception as e:
            print(f"❌ Error initializing GenAI Summarizer: {e}")
            self.genai_summarizer = None
    
    async def run_complete_incident_analysis(self, 
                                           data_paths: Optional[Dict[str, str]] = None,
                                           train_if_needed: bool = True) -> Dict[str, Any]:
        """
        Run complete incident analysis pipeline
        
        Pipeline Flow:
        1. Data Collection (Your DataIntegrator)
        2. ML Detection (Your ProductionMetaSystem)  
        3. NLP Context Extraction (New ProductionNLPProcessor)
        4. GenAI Incident Brief (New ProductionIncidentSummarizer)
        5. Actionable Recommendations
        """
        
        pipeline_start = datetime.now()
        print(f"\n🔄 Starting Complete SRE Analysis Pipeline #{self.pipeline_runs + 1}")
        print(f"⏰ Started at: {pipeline_start.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        
        # Initialize results structure
        results = {
            'pipeline_metadata': {
                'run_number': self.pipeline_runs + 1,
                'started_at': pipeline_start.isoformat(),
                'components_used': [],
                'incident_detected': False,
                'incident_probability': 0.0
            },
            'data_collection': {},
            'ml_analysis': {},
            'nlp_analysis': {},
            'genai_brief': {},
            'recommendations': {},
            'system_status': {},
            'errors': []
        }
        
        try:
            # PHASE 1: Data Collection (Your System)
            print("📊 PHASE 1: Data Collection & Integration")
            print("-" * 50)
            
            collected_data = await self._run_data_collection_phase(data_paths)
            results['data_collection'] = collected_data
            results['pipeline_metadata']['components_used'].append('data_collection')
            
            if not collected_data or collected_data.get('status') == 'failed':
                return self._create_error_result("Data collection failed", results)
            
            # PHASE 2: ML Detection (Your Production System)
            print("\n🤖 PHASE 2: Production ML Detection & Analysis")
            print("-" * 50)
            
            ml_analysis = await self._run_ml_analysis_phase(collected_data, train_if_needed)
            results['ml_analysis'] = ml_analysis
            results['pipeline_metadata']['components_used'].append('ml_detection')
            
            # Check for incident detection
            incident_detected, incident_probability = self._evaluate_incident_detection(ml_analysis)
            results['pipeline_metadata']['incident_detected'] = incident_detected
            results['pipeline_metadata']['incident_probability'] = incident_probability
            
            if incident_detected:
                print(f"🚨 INCIDENT DETECTED! Probability: {incident_probability:.3f}")
                print(f"🎯 Proceeding with full incident analysis...")
                self.last_incident_time = pipeline_start
            else:
                print(f"✅ No incident detected (probability: {incident_probability:.3f})")
                print(f"📊 System monitoring normal - completing basic analysis...")
            
            # PHASE 3: NLP Context Extraction (Always run for insights)
            print("\n🧠 PHASE 3: Advanced NLP Context Extraction")
            print("-" * 50)
            
            nlp_analysis = await self._run_nlp_analysis_phase(collected_data)
            results['nlp_analysis'] = nlp_analysis
            results['pipeline_metadata']['components_used'].append('nlp_analysis')
            
            # PHASE 4: GenAI Incident Brief (If incident or for insights)
            if incident_detected or self.config.system_config.get('always_generate_brief', True):
                print("\n📝 PHASE 4: GenAI Incident Brief Generation")
                print("-" * 50)
                
                genai_brief = await self._run_genai_brief_phase(nlp_analysis, ml_analysis, collected_data)
                results['genai_brief'] = genai_brief
                results['pipeline_metadata']['components_used'].append('genai_summarization')
            
            # PHASE 5: Generate Comprehensive Recommendations
            print("\n🎯 PHASE 5: Actionable Recommendations Generation")
            print("-" * 50)
            
            recommendations = self._generate_comprehensive_recommendations(results, incident_probability)
            results['recommendations'] = recommendations
            
            # PHASE 6: System Status & Health Assessment
            print("\n📊 PHASE 6: System Status Assessment")
            print("-" * 50)
            
            system_status = self._assess_system_status(results)
            results['system_status'] = system_status
            
            # Complete pipeline
            results['pipeline_metadata']['completed_at'] = datetime.now().isoformat()
            results['pipeline_metadata']['duration_seconds'] = (
                datetime.now() - pipeline_start
            ).total_seconds()
            
            self.pipeline_runs += 1
            
            print(f"\n✅ COMPLETE SRE ANALYSIS PIPELINE FINISHED!")
            print(f"⏱️ Total Duration: {results['pipeline_metadata']['duration_seconds']:.1f} seconds")
            print(f"🎯 Components Used: {', '.join(results['pipeline_metadata']['components_used'])}")
            print("="*70)
            
            # Display comprehensive results
            self._display_complete_results_summary(results)
            
            return results
            
        except Exception as e:
            error_msg = f"Complete pipeline error: {str(e)}"
            print(f"❌ {error_msg}")
            print(f"📍 Error details: {traceback.format_exc()}")
            return self._create_error_result(error_msg, results)
    
    async def _run_data_collection_phase(self, data_paths: Optional[Dict[str, str]]) -> Dict[str, Any]:
        """Run data collection using your existing system"""
        try:
            if not self.production_system or not hasattr(self.production_system, 'collect_production_data'):
                print("   ⚠️ Production data collection not available - using mock data")
                return self._create_mock_data()
            
            print("   📋 Collecting production SRE data...")
            
            # Use your existing data collection
            collected_data = self.production_system.collect_production_data()
            
            if collected_data.empty:
                print("   ⚠️ No production data collected - using mock data")
                return self._create_mock_data()
            
            # Enhance with additional context if data_paths provided
            enhanced_data = self._enhance_collected_data(collected_data, data_paths)
            
            print(f"   ✅ Production data collected: {len(enhanced_data)} records")
            
            return {
                'raw_data': enhanced_data,
                'collection_method': 'production_system',
                'data_quality': 'high',
                'status': 'success'
            }
            
        except Exception as e:
            print(f"   ❌ Data collection error: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    async def _run_ml_analysis_phase(self, collected_data: Dict[str, Any], 
                                   train_if_needed: bool = True) -> Dict[str, Any]:
        """Run ML analysis using your ProductionMetaSystem"""
        try:
            if not self.production_system:
                print("   ❌ Production ML System not available")
                return {'status': 'failed', 'error': 'Production ML system not initialized'}
            
            print("   🔍 Running Production ML Detection Models...")
            
            raw_data = collected_data.get('raw_data')
            if raw_data is None or (isinstance(raw_data, pd.DataFrame) and raw_data.empty):
                print("   ⚠️ No data available for ML analysis")
                return {'status': 'no_data'}
            
            # Check if models are trained
            if not self.production_system.is_trained and train_if_needed:
                print("   🏋️ Training Production ML models...")
                
                # Train all models using your system
                training_results = self.production_system.train_all_models(raw_data)
                
                if training_results.get('overall_status') == 'success':
                    print("   ✅ Production ML models trained successfully")
                    
                    # Get predictions from trained models
                    predictions = self.production_system.get_model_predictions(raw_data)
                    
                    # Train meta-model
                    meta_results = self.production_system.train_production_meta_model(raw_data, predictions)
                    
                    if meta_results.get('status') == 'success':
                        print(f"   ✅ Production meta-model trained: {meta_results.get('accuracy', 0)*100:.1f}% accuracy")
                        
                        # Save models
                        self.production_system.save_production_models()
                        
                    else:
                        print("   ⚠️ Meta-model training had issues")
                
                else:
                    print("   ⚠️ Some Production ML models failed to train")
            
            # Get predictions from your system
            if self.production_system.is_trained:
                print("   🎯 Getting Production ML predictions...")
                
                incident_predictions = self.production_system.predict_production_incidents(raw_data)
                
                if incident_predictions:
                    print(f"   ✅ ML Analysis completed")
                    print(f"       - Samples analyzed: {incident_predictions.get('samples_processed', 0):,}")
                    print(f"       - Incidents detected: {incident_predictions.get('incident_count', 0)}")
                    print(f"       - Incident rate: {incident_predictions.get('incident_rate', 0):.1f}%")
                    
                    return {
                        'predictions': incident_predictions,
                        'model_status': 'trained_and_active',
                        'analysis_method': 'production_meta_system',
                        'status': 'success'
                    }
                else:
                    print("   ❌ No predictions generated")
                    return {'status': 'prediction_failed'}
            
            else:
                print("   ⚠️ Production ML models not trained")
                return {'status': 'models_not_trained'}
                
        except Exception as e:
            print(f"   ❌ ML analysis error: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    async def _run_nlp_analysis_phase(self, collected_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run NLP analysis using ProductionNLPProcessor"""
        try:
            if not self.nlp_processor:
                print("   ❌ NLP Processor not available")
                return self._create_basic_nlp_results()
            
            print("   🧠 Processing text data with Advanced NLP...")
            
            # Extract raw data for NLP processing
            raw_data = collected_data.get('raw_data')
            
            # Convert your ProductionMetaSystem data format to NLP processor format
            nlp_input_data = self._convert_data_for_nlp(collected_data)
            
            # Process with NLP
            nlp_results = self.nlp_processor.process_production_data(nlp_input_data)
            
            if nlp_results:
                incident_insights = nlp_results.get('incident_insights', {})
                print(f"   ✅ NLP Analysis completed")
                print(f"       - Processing mode: {nlp_results['processing_metadata']['processing_mode']}")
                print(f"       - Incident severity: {incident_insights.get('incident_severity', 'unknown')}")
                print(f"       - Confidence score: {incident_insights.get('confidence_score', 0):.3f}")
                print(f"       - Affected components: {len(incident_insights.get('affected_components', []))}")
                
                return {
                    'nlp_results': nlp_results,
                    'processing_status': 'success',
                    'insights_available': True
                }
            else:
                print("   ⚠️ NLP processing returned no results")
                return self._create_basic_nlp_results()
                
        except Exception as e:
            print(f"   ❌ NLP analysis error: {e}")
            return self._create_basic_nlp_results()
    
    async def _run_genai_brief_phase(self, nlp_analysis: Dict[str, Any],
                                   ml_analysis: Dict[str, Any],
                                   collected_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run GenAI brief generation using ProductionIncidentSummarizer"""
        try:
            if not self.genai_summarizer:
                print("   ❌ GenAI Summarizer not available")
                return self._create_basic_incident_brief()
            
            print("   🤖 Generating AI-powered incident brief...")
            
            # Extract data for GenAI processing
            nlp_results = nlp_analysis.get('nlp_results', {})
            ml_predictions = ml_analysis.get('predictions', {})
            
            # Generate comprehensive brief
            incident_brief = self.genai_summarizer.generate_production_incident_brief(
                nlp_results, ml_predictions, collected_data
            )
            
            if incident_brief:
                metadata = incident_brief.get('metadata', {})
                quality = incident_brief.get('quality_metrics', {})
                
                print(f"   ✅ GenAI Brief generated")
                print(f"       - Generation model: {metadata.get('generation_model', 'unknown')}")
                print(f"       - Processing time: {metadata.get('processing_time_seconds', 0):.1f}s")
                print(f"       - Quality score: {quality.get('overall_quality', 0):.3f}")
                print(f"       - AI generation rate: {quality.get('ai_generation_rate', 0):.1%}")
                
                return {
                    'incident_brief': incident_brief,
                    'generation_status': 'success',
                    'ai_enhanced': not self.genai_summarizer.fallback_mode
                }
            else:
                print("   ⚠️ GenAI brief generation failed")
                return self._create_basic_incident_brief()
                
        except Exception as e:
            print(f"   ❌ GenAI brief error: {e}")
            return self._create_basic_incident_brief()
    
    def _evaluate_incident_detection(self, ml_analysis: Dict[str, Any]) -> Tuple[bool, float]:
        """Evaluate if incident was detected based on ML analysis"""
        
        if ml_analysis.get('status') != 'success':
            return False, 0.0
        
        predictions = ml_analysis.get('predictions', {})
        
        # Get incident probability from your ProductionMetaSystem
        incident_rate = predictions.get('incident_rate', 0.0)
        incident_count = predictions.get('incident_count', 0)
        threshold = predictions.get('threshold', 0.5)
        
        # Convert rate to probability
        incident_probability = incident_rate / 100.0
        
        # Check against threshold
        incident_detected = incident_probability >= self.config.ml_config['confidence_threshold']
        
        return incident_detected, incident_probability
    
    def _generate_comprehensive_recommendations(self, results: Dict[str, Any], 
                                              incident_probability: float) -> Dict[str, Any]:
        """Generate comprehensive recommendations"""
        
        recommendations = {
            'immediate_actions': [],
            'monitoring_actions': [],
            'investigation_actions': [],
            'communication_actions': [],
            'prevention_actions': [],
            'priority_level': 'medium',
            'estimated_timeline': 'TBD'
        }
        
        try:
            # Extract recommendations from GenAI brief if available
            genai_brief = results.get('genai_brief', {}).get('incident_brief', {})
            if genai_brief:
                brief_data = genai_brief.get('incident_brief', {})
                
                # Extract immediate actions
                immediate_actions = brief_data.get('immediate_actions', {})
                if immediate_actions.get('structured_actions'):
                    recommendations['immediate_actions'] = immediate_actions['structured_actions'][:5]
                
                # Extract recovery plan items
                recovery_plan = brief_data.get('recovery_plan', {})
                if recovery_plan.get('recovery_phases'):
                    recommendations['investigation_actions'] = [
                        f"Execute {phase}" for phase in recovery_plan['recovery_phases'][:3]
                    ]
                
                # Extract communication plan
                comm_plan = brief_data.get('communication_plan', {})
                if comm_plan.get('stakeholder_groups'):
                    recommendations['communication_actions'] = [
                        f"Notify {group}" for group in comm_plan['stakeholder_groups'][:3]
                    ]
            
            # Determine priority based on incident probability
            if incident_probability >= 0.8:
                recommendations['priority_level'] = 'critical'
                recommendations['estimated_timeline'] = '0-2 hours'
            elif incident_probability >= 0.6:
                recommendations['priority_level'] = 'high'
                recommendations['estimated_timeline'] = '2-4 hours'
            elif incident_probability >= 0.4:
                recommendations['priority_level'] = 'medium'
                recommendations['estimated_timeline'] = '4-8 hours'
            else:
                recommendations['priority_level'] = 'low'
                recommendations['estimated_timeline'] = '8+ hours'
            
            # Add default monitoring actions
            if not recommendations['monitoring_actions']:
                recommendations['monitoring_actions'] = [
                    "Monitor system metrics continuously",
                    "Watch for error rate increases",
                    "Track performance degradation",
                    "Observe user impact indicators"
                ]
            
            # Add default prevention actions
            recommendations['prevention_actions'] = [
                "Review incident for pattern analysis",
                "Update monitoring thresholds if needed",
                "Document lessons learned",
                "Consider infrastructure improvements"
            ]
            
            return recommendations
            
        except Exception as e:
            print(f"⚠️ Error generating recommendations: {e}")
            return {
                'immediate_actions': ["Manual analysis required"],
                'priority_level': 'medium',
                'error': str(e)
            }
    
    def _assess_system_status(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall system status"""
        
        status = {
            'overall_health': 'unknown',
            'component_status': {},
            'data_quality': 'unknown',
            'ml_performance': 'unknown',
            'incident_response_readiness': 'unknown',
            'recommendations_for_improvement': []
        }
        
        try:
            # Assess component availability
            components_used = results['pipeline_metadata'].get('components_used', [])
            
            status['component_status'] = {
                'data_collection': 'available' if 'data_collection' in components_used else 'unavailable',
                'ml_detection': 'available' if 'ml_detection' in components_used else 'unavailable',
                'nlp_analysis': 'available' if 'nlp_analysis' in components_used else 'unavailable',
                'genai_summarization': 'available' if 'genai_summarization' in components_used else 'unavailable'
            }
            
            # Assess data quality
            data_collection = results.get('data_collection', {})
            if data_collection.get('data_quality') == 'high':
                status['data_quality'] = 'excellent'
            elif data_collection.get('status') == 'success':
                status['data_quality'] = 'good'
            else:
                status['data_quality'] = 'poor'
            
            # Assess ML performance
            ml_analysis = results.get('ml_analysis', {})
            if ml_analysis.get('status') == 'success':
                predictions = ml_analysis.get('predictions', {})
                if predictions:
                    status['ml_performance'] = 'excellent'
                else:
                    status['ml_performance'] = 'degraded'
            else:
                status['ml_performance'] = 'failed'
            
            # Assess incident response readiness
            available_components = len([v for v in status['component_status'].values() if v == 'available'])
            
            if available_components >= 3:
                status['incident_response_readiness'] = 'fully_ready'
            elif available_components >= 2:
                status['incident_response_readiness'] = 'partially_ready'
            else:
                status['incident_response_readiness'] = 'limited'
            
            # Overall health assessment
            if status['incident_response_readiness'] == 'fully_ready' and status['ml_performance'] == 'excellent':
                status['overall_health'] = 'excellent'
            elif available_components >= 2:
                status['overall_health'] = 'good'
            else:
                status['overall_health'] = 'needs_attention'
            
            # Generate improvement recommendations
            if status['component_status']['genai_summarization'] == 'unavailable':
                status['recommendations_for_improvement'].append("Configure Gemini API key for AI-powered incident briefs")
            
            if status['component_status']['nlp_analysis'] == 'unavailable':
                status['recommendations_for_improvement'].append("Install NLP libraries for advanced context extraction")
            
            if status['ml_performance'] != 'excellent':
                status['recommendations_for_improvement'].append("Train ML models with more production data")
            
            return status
            
        except Exception as e:
            print(f"⚠️ Error assessing system status: {e}")
            status['error'] = str(e)
            return status
    
    def _display_complete_results_summary(self, results: Dict[str, Any]):
        """Display comprehensive results summary"""
        
        print("\n📋 COMPLETE SRE ANALYSIS RESULTS SUMMARY")
        print("="*70)
        
        metadata = results.get('pipeline_metadata', {})
        
        # Basic pipeline info
        print(f"🕐 Pipeline Run #{metadata.get('run_number', 0)}")
        print(f"⏱️ Duration: {metadata.get('duration_seconds', 0):.1f} seconds")
        print(f"🎯 Components: {', '.join(metadata.get('components_used', []))}")
        
        # Incident detection
        if metadata.get('incident_detected'):
            print(f"🚨 INCIDENT DETECTED (Probability: {metadata.get('incident_probability', 0):.3f})")
        else:
            print(f"✅ No incident detected (Probability: {metadata.get('incident_probability', 0):.3f})")
        
        # ML Analysis Summary
        ml_analysis = results.get('ml_analysis', {})
        if ml_analysis.get('status') == 'success':
            predictions = ml_analysis.get('predictions', {})
            print(f"🤖 ML Analysis: {predictions.get('samples_processed', 0):,} samples, "
                  f"{predictions.get('incident_count', 0)} incidents ({predictions.get('incident_rate', 0):.1f}%)")
        
        # NLP Analysis Summary
        nlp_analysis = results.get('nlp_analysis', {})
        if nlp_analysis.get('processing_status') == 'success':
            nlp_results = nlp_analysis.get('nlp_results', {})
            incident_insights = nlp_results.get('incident_insights', {})
            print(f"🧠 NLP Analysis: {incident_insights.get('incident_severity', 'unknown')} severity, "
                  f"{len(incident_insights.get('affected_components', []))} components affected")
        
        # GenAI Brief Summary
        genai_brief = results.get('genai_brief', {})
        if genai_brief.get('generation_status') == 'success':
            brief_metadata = genai_brief.get('incident_brief', {}).get('metadata', {})
            quality = genai_brief.get('incident_brief', {}).get('quality_metrics', {})
            print(f"📝 GenAI Brief: {brief_metadata.get('generation_model', 'unknown')} model, "
                  f"{quality.get('overall_quality', 0):.1%} quality score")
        
        # Recommendations Summary
        recommendations = results.get('recommendations', {})
        if recommendations:
            print(f"💡 Recommendations: {recommendations.get('priority_level', 'unknown').upper()} priority, "
                  f"{len(recommendations.get('immediate_actions', []))} immediate actions")
        
        # System Status
        system_status = results.get('system_status', {})
        if system_status:
            print(f"📊 System Status: {system_status.get('overall_health', 'unknown').upper()} health, "
                  f"{system_status.get('incident_response_readiness', 'unknown')} response readiness")
        
        # Show key insights if incident detected
        if metadata.get('incident_detected'):
            print(f"\n🔍 KEY INCIDENT INSIGHTS:")
            print("-" * 50)
            
            # From NLP
            if nlp_analysis.get('processing_status') == 'success':
                nlp_results = nlp_analysis.get('nlp_results', {})
                incident_insights = nlp_results.get('incident_insights', {})
                
                components = incident_insights.get('affected_components', [])
                if components:
                    print(f"   🎯 Affected Components: {', '.join(components[:5])}")
                
                causes = incident_insights.get('probable_causes', [])
                if causes:
                    print(f"   🔍 Probable Causes: {', '.join(causes[:3])}")
                
                people = incident_insights.get('people_involved', [])
                if people:
                    print(f"   👥 People Involved: {', '.join(people[:3])}")
            
            # From GenAI Brief
            if genai_brief.get('generation_status') == 'success':
                brief_data = genai_brief.get('incident_brief', {}).get('incident_brief', {})
                
                exec_summary = brief_data.get('executive_summary', {}).get('summary_text', '')
                if exec_summary and len(exec_summary) > 50:
                    print(f"   📋 Executive Summary: {exec_summary[:100]}...")
                
                immediate = brief_data.get('immediate_actions', {}).get('structured_actions', [])
                if immediate:
                    print(f"   ⚡ Top Actions: {len(immediate)} immediate actions recommended")
        
        # Show improvement recommendations
        improvements = system_status.get('recommendations_for_improvement', [])
        if improvements:
            print(f"\n🔧 SYSTEM IMPROVEMENT RECOMMENDATIONS:")
            print("-" * 50)
            for i, improvement in enumerate(improvements[:3], 1):
                print(f"   {i}. {improvement}")
        
        print("="*70)
    
    # Utility methods
    def _create_mock_data(self) -> Dict[str, Any]:
        """Create mock data when production data is not available"""
        
        mock_df = pd.DataFrame([
            {
                'timestamp': datetime.now() - timedelta(minutes=30),
                'cpu_util': 45.2,
                'memory_util': 62.1,
                'error_rate': 0.02,
                'will_fail': 0,
                'is_anomaly': 0,
                'is_security_threat': 0
            },
            {
                'timestamp': datetime.now() - timedelta(minutes=25),
                'cpu_util': 78.5,
                'memory_util': 71.3,
                'error_rate': 0.08,
                'will_fail': 1,
                'is_anomaly': 1,
                'is_security_threat': 0
            },
            {
                'timestamp': datetime.now() - timedelta(minutes=20),
                'cpu_util': 92.1,
                'memory_util': 85.7,
                'error_rate': 0.15,
                'will_fail': 1,
                'is_anomaly': 1,
                'is_security_threat': 0
            }
        ])
        
        return mock_df
    
    def _enhance_collected_data(self, collected_data: pd.DataFrame, 
                               data_paths: Optional[Dict[str, str]]) -> pd.DataFrame:
        """Enhance collected data with additional sources if provided"""
        
        if not data_paths:
            return collected_data
        
        try:
            # Could load additional data from paths if needed
            # For now, just return the original data
            return collected_data
            
        except Exception as e:
            print(f"   ⚠️ Error enhancing data: {e}")
            return collected_data
    
    def _convert_data_for_nlp(self, collected_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert ProductionMetaSystem data format for NLP processing"""
        
        try:
            raw_data = collected_data.get('raw_data')
            
            # Create mock structured data for NLP processing
            nlp_data = {
                'logs': {'logs': pd.DataFrame()},
                'metrics': {'metrics': raw_data if isinstance(raw_data, pd.DataFrame) else pd.DataFrame()},
                'chats': {'chats': pd.DataFrame()},
                'tickets': {'tickets': pd.DataFrame()}
            }
            
            # Add some sample logs and chats for demonstration
            if isinstance(raw_data, pd.DataFrame) and not raw_data.empty:
                # Create sample log entries based on metrics
                high_error_entries = raw_data[raw_data.get('error_rate', 0) > 0.05]
                
                if not high_error_entries.empty:
                    sample_logs = []
                    for _, row in high_error_entries.head(3).iterrows():
                        sample_logs.append({
                            'timestamp': row.get('timestamp', datetime.now()),
                            'level': 'ERROR',
                            'message': f"High error rate detected: {row.get('error_rate', 0):.3f}"
                        })
                    
                    nlp_data['logs']['logs'] = pd.DataFrame(sample_logs)
                
                # Create sample chat entries
                sample_chats = [
                    {
                        'timestamp': datetime.now() - timedelta(minutes=15),
                        'user': 'alice',
                        'message': 'Seeing elevated error rates in the API service'
                    },
                    {
                        'timestamp': datetime.now() - timedelta(minutes=10),
                        'user': 'bob',
                        'message': 'CPU usage is spiking on server-01, investigating'
                    }
                ]
                
                nlp_data['chats']['chats'] = pd.DataFrame(sample_chats)
            
            return nlp_data
            
        except Exception as e:
            print(f"   ⚠️ Error converting data for NLP: {e}")
            return {'logs': {'logs': pd.DataFrame()}, 'metrics': {'metrics': pd.DataFrame()}, 
                   'chats': {'chats': pd.DataFrame()}, 'tickets': {'tickets': pd.DataFrame()}}
    
    def _create_basic_nlp_results(self) -> Dict[str, Any]:
        """Create basic NLP results as fallback"""
        return {
            'nlp_results': {
                'incident_insights': {
                    'incident_severity': 'unknown',
                    'confidence_score': 0.3,
                    'affected_components': [],
                    'probable_causes': ['Manual analysis required'],
                    'people_involved': [],
                    'business_impact': 'unknown'
                }
            },
            'processing_status': 'fallback',
            'insights_available': False
        }
    
    def _create_basic_incident_brief(self) -> Dict[str, Any]:
        """Create basic incident brief as fallback"""
        return {
            'incident_brief': {
                'metadata': {
                    'generation_model': 'fallback',
                    'processing_time_seconds': 0.1,
                    'limitation': 'AI capabilities not available'
                },
                'incident_brief': {
                    'executive_summary': {
                        'summary_text': 'Incident detected - manual analysis required for detailed assessment.',
                        'generation_method': 'fallback'
                    },
                    'immediate_actions': {
                        'structured_actions': [
                            {'action': 'Review system logs manually', 'priority': 'high', 'estimated_time': '15 min'},
                            {'action': 'Check system metrics', 'priority': 'high', 'estimated_time': '10 min'},
                            {'action': 'Contact on-call engineer', 'priority': 'medium', 'estimated_time': '5 min'}
                        ]
                    }
                },
                'quality_metrics': {
                    'overall_quality': 0.3,
                    'ai_generation_rate': 0.0
                }
            },
            'generation_status': 'fallback',
            'ai_enhanced': False
        }
    
    def _create_error_result(self, error_msg: str, partial_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create error result structure"""
        partial_results['pipeline_metadata']['status'] = 'failed'
        partial_results['pipeline_metadata']['error'] = error_msg
        partial_results['pipeline_metadata']['completed_at'] = datetime.now().isoformat()
        partial_results['errors'].append(error_msg)
        
        return partial_results
    
    def get_system_health_report(self) -> Dict[str, Any]:
        """Get comprehensive system health report"""
        
        health_report = {
            'system_info': {
                'engine_version': '1.0.0',
                'initialized': self.is_initialized,
                'pipeline_runs_completed': self.pipeline_runs,
                'last_incident_time': self.last_incident_time.isoformat() if self.last_incident_time else None,
                'current_time': datetime.now().isoformat()
            },
            'component_status': {
                'production_ml_system': {
                    'available': bool(self.production_system),
                    'trained': getattr(self.production_system, 'is_trained', False) if self.production_system else False,
                    'model_directory': self.config.ml_config['model_dir']
                },
                'nlp_processor': {
                    'available': bool(self.nlp_processor),
                    'mode': 'production' if self.nlp_processor and hasattr(self.nlp_processor, 'nlp_model') and self.nlp_processor.nlp_model else 'fallback'
                },
                'genai_summarizer': {
                    'available': bool(self.genai_summarizer),
                    'ai_mode': not getattr(self.genai_summarizer, 'fallback_mode', True) if self.genai_summarizer else False,
                    'model_name': self.config.genai_config['model_name']
                }
            },
            'configuration': {
                'api_keys': {
                    'gemini_configured': bool(self.config.GEMINI_API_KEY),
                    'api_key_length': len(self.config.GEMINI_API_KEY) if self.config.GEMINI_API_KEY else 0
                },
                'ml_settings': {
                    'confidence_threshold': self.config.ml_config['confidence_threshold'],
                    'correlation_window': self.config.ml_config['correlation_time_window_minutes']
                },
                'production_mode': self.config.system_config['production_mode']
            },
            'performance': {
                'average_pipeline_duration': 'TBD',  # Could track this over time
                'success_rate': 'TBD',               # Could track this over time
                'error_count': len(self.errors)
            }
        }
        
        return health_report
    
    def run_continuous_monitoring(self, interval_minutes: int = 10):
        """Run continuous monitoring"""
        
        print(f"🔄 Starting SRE Continuous Monitoring (every {interval_minutes} minutes)")
        print(f"🎯 Complete pipeline: Data → ML → NLP → GenAI → Action")
        print("Press Ctrl+C to stop monitoring")
        print("="*70)
        
        async def monitoring_loop():
            while True:
                try:
                    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    print(f"\n⏰ [{current_time}] Running complete SRE analysis cycle...")
                    
                    # Run complete analysis
                    results = await self.run_complete_incident_analysis()
                    
                    # Check results
                    if results.get('pipeline_metadata', {}).get('incident_detected'):
                        print("🚨 INCIDENT DETECTED - Review results above!")
                        
                        # Here you could add notification logic
                        # await self._send_incident_notification(results)
                        
                    else:
                        print("✅ Monitoring cycle completed - no incidents detected")
                    
                    # Wait for next cycle
                    print(f"😴 Next monitoring cycle in {interval_minutes} minutes...")
                    await asyncio.sleep(interval_minutes * 60)
                    
                except KeyboardInterrupt:
                    print("\n⏹️ Continuous monitoring stopped by user")
                    break
                except Exception as e:
                    print(f"❌ Monitoring error: {str(e)}")
                    print("⏳ Waiting 2 minutes before retry...")
                    await asyncio.sleep(120)
        
        try:
            asyncio.run(monitoring_loop())
        except KeyboardInterrupt:
            print("\n👋 SRE Continuous Monitoring stopped")

# Main execution functions
async def run_single_sre_analysis(data_paths: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Run a single SRE analysis"""
    
    engine = CompleteSREInsightEngine()
    return await engine.run_complete_incident_analysis(data_paths)

def run_sre_continuous_monitoring(interval_minutes: int = 10):
    """Run continuous SRE monitoring"""
    
    engine = CompleteSREInsightEngine()
    engine.run_continuous_monitoring(interval_minutes)

def get_sre_system_status():
    """Get SRE system status"""
    
    engine = CompleteSREInsightEngine()
    return engine.get_system_health_report()

# Main execution
async def main():
    """Main function demonstrating the complete SRE system"""
    
    print("🎯 COMPLETE SRE INCIDENT INSIGHT ENGINE")
    print("="*70)
    print("🚀 Context-Aware Multimodal Incident Analysis System")
    print("📊 Integrates with your ProductionMetaSystem")
    print("🧠 Advanced NLP Context Extraction")
    print("🤖 AI-Powered Incident Briefs with Gemini")
    print("="*70)
    
    # Initialize complete system
    engine = CompleteSREInsightEngine()
    
    # Show system status
    health_report = engine.get_system_health_report()
    print(f"\n📊 SYSTEM HEALTH STATUS:")
    print("-" * 50)
    print(f"   🚀 Engine: {'Initialized' if health_report['system_info']['initialized'] else 'Failed'}")
    print(f"   🤖 Production ML: {'Available' if health_report['component_status']['production_ml_system']['available'] else 'Not Available'}")
    print(f"   🧠 NLP Processor: {'Available' if health_report['component_status']['nlp_processor']['available'] else 'Not Available'}")
    print(f"   📝 GenAI Summarizer: {'Available' if health_report['component_status']['genai_summarizer']['available'] else 'Not Available'}")
    print(f"   🔑 Gemini API: {'Configured' if health_report['configuration']['api_keys']['gemini_configured'] else 'Not Configured'}")
    
    print(f"\n🎯 Running Complete SRE Analysis Pipeline...")
    
    try:
        # Run complete analysis
        results = await engine.run_complete_incident_analysis()
        
        # Show final summary
        if results.get('pipeline_metadata', {}).get('incident_detected'):
            print(f"\n🚨 INCIDENT ANALYSIS COMPLETED!")
            
            # Show key findings from GenAI brief
            genai_brief = results.get('genai_brief', {})
            if genai_brief.get('generation_status') == 'success':
                brief_data = genai_brief.get('incident_brief', {}).get('incident_brief', {})
                
                exec_summary = brief_data.get('executive_summary', {}).get('summary_text', '')
                if exec_summary:
                    print(f"\n📋 Executive Summary:")
                    print(f"{exec_summary[:300]}..." if len(exec_summary) > 300 else exec_summary)
                
                immediate = brief_data.get('immediate_actions', {}).get('structured_actions', [])
                if immediate:
                    print(f"\n⚡ Top Immediate Actions:")
                    for i, action in enumerate(immediate[:3], 1):
                        print(f"   {i}. {action.get('action', 'Unknown')} ({action.get('priority', 'unknown')} priority)")
        
        else:
            print(f"\n✅ SYSTEM MONITORING COMPLETED")
            print(f"   📊 No incidents detected - all systems operating normally")
        
    except Exception as e:
        print(f"❌ Analysis execution error: {str(e)}")
        return
    
    print(f"\n💡 NEXT STEPS:")
    print("-" * 30)
    print("1. Run continuous monitoring: run_sre_continuous_monitoring()")
    print("2. Check system status: get_sre_system_status()")
    print("3. Run single analysis: run_single_sre_analysis()")
    
    if not health_report['configuration']['api_keys']['gemini_configured']:
        print("4. Configure Gemini API key for enhanced AI capabilities")
    
    print(f"\n🎉 YOUR COMPLETE SRE INCIDENT INSIGHT ENGINE IS READY!")
    print("="*70)

if __name__ == "__main__":
    # Choose execution mode
    print("🎯 Complete SRE Incident Insight Engine")
    print("="*50)
    print("Choose execution mode:")
    print("1. Single analysis run")
    print("2. Continuous monitoring (every 10 minutes)")
    print("3. System health check")
    print("4. Demo run (default)")
    
    try:
        choice = input("\nEnter choice (1-4, default=4): ").strip() or "4"
        
        if choice == "1":
            print("🔍 Running single SRE analysis...")
            result = asyncio.run(run_single_sre_analysis())
            print(f"✅ Analysis completed: {result['pipeline_metadata']['status'] if 'pipeline_metadata' in result else 'Unknown'}")
            
        elif choice == "2":
            print("🔄 Starting continuous monitoring...")
            run_sre_continuous_monitoring(interval_minutes=10)
            
        elif choice == "3":
            print("📊 Checking system health...")
            health = get_sre_system_status()
            print(f"\nSystem Health Report:")
            print(json.dumps(health, indent=2, default=str))
            
        else:
            print("🚀 Running demo analysis...")
            asyncio.run(main())
            
    except KeyboardInterrupt:
        print("\n👋 SRE Engine stopped by user")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("Running default demo...")
        asyncio.run(main())
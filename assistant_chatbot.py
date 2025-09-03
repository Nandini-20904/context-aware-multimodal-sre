"""
SRE Assistant Chatbot for Incident Response
Interactive AI assistant that helps SRE engineers with incident analysis,
troubleshooting, and recommendations using your existing GenAI system
"""

import asyncio
import json
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd

# Import existing components
try:
    from config import config
    from incident import ProductionIncidentSummarizer
    from sre import CompleteSREInsightEngine
    COMPONENTS_AVAILABLE = True
    print("✅ SRE components imported successfully")
except ImportError as e:
    print(f"❌ Could not import SRE components: {e}")
    COMPONENTS_AVAILABLE = False

# GenAI Libraries
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    print("⚠️ Google GenerativeAI not available")
    GENAI_AVAILABLE = False

class SREAssistantChatbot:
    """
    Interactive SRE Assistant Chatbot
    Helps SRE engineers with incident response, troubleshooting, and system analysis
    """
    
    def __init__(self):
        """Initialize SRE Assistant Chatbot"""
        self.logger = logging.getLogger(__name__)
        
        print("🤖 Initializing SRE Assistant Chatbot...")
        print("="*60)
        print("🎯 Your AI-Powered SRE Expert Assistant")
        print("💬 Interactive incident response and troubleshooting")
        print("🧠 Powered by your Gemini AI system")
        print("="*60)
        
        # Configuration
        self.config = config if COMPONENTS_AVAILABLE else None
        self.conversation_history = []
        self.current_incident_context = None
        self.last_analysis_results = None
        
        # Initialize core components
        self.sre_engine = None
        self.incident_summarizer = None
        self.model = None
        self.is_ready = False
        
        # Initialize chatbot
        self._init_chatbot_system()
        
        # Conversation state
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.total_messages = 0
        
        if self.is_ready:
            print("✅ SRE Assistant Chatbot initialized successfully!")
            print(f"🎯 Session ID: {self.session_id}")
            print("💬 Ready to help with your SRE tasks!")
        else:
            print("⚠️ SRE Assistant Chatbot initialized with limited capabilities")
        
        print("="*60)
    
    def _init_chatbot_system(self):
        """Initialize chatbot system components"""
        try:
            if not COMPONENTS_AVAILABLE or not GENAI_AVAILABLE:
                print("❌ Required components not available - using fallback mode")
                self.is_ready = False
                return
            
            # Initialize SRE Engine for system integration
            print("🔧 Initializing SRE Engine integration...")
            try:
                self.sre_engine = CompleteSREInsightEngine()
                print("   ✅ SRE Engine integrated")
            except Exception as e:
                print(f"   ⚠️ SRE Engine not available: {e}")
                self.sre_engine = None
            
            # Initialize Incident Summarizer for structured responses
            print("📝 Initializing Incident Summarizer...")
            try:
                self.incident_summarizer = ProductionIncidentSummarizer()
                print("   ✅ Incident Summarizer ready")
            except Exception as e:
                print(f"   ⚠️ Incident Summarizer not available: {e}")
                self.incident_summarizer = None
            
            # Initialize direct Gemini model for conversational responses
            print("🤖 Initializing conversational AI model...")
            try:
                api_key = self.config.GEMINI_API_KEY if self.config else None
                if not api_key:
                    raise Exception("No API key configured")
                
                genai.configure(api_key=api_key)
                
                # Optimized settings for conversation
                generation_config = {
                    "temperature": 0.4,  # Slightly more creative for conversation
                    "top_p": 0.8,
                    "top_k": 40,
                    "max_output_tokens": 1500,  # Reasonable length for chat
                }
                
                safety_settings = [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                ]
                
                self.model = genai.GenerativeModel(
                    model_name="gemini-1.5-flash",
                    generation_config=generation_config,
                    safety_settings=safety_settings,
                    system_instruction=self._get_system_instruction()
                )
                
                # Test connection
                test_response = self.model.generate_content("Test connection - respond with 'SRE_CHATBOT_READY'")
                if test_response and test_response.text and 'SRE_CHATBOT_READY' in test_response.text:
                    print("   ✅ Conversational AI model ready")
                    self.is_ready = True
                else:
                    raise Exception("API test failed")
                    
            except Exception as e:
                print(f"   ❌ Conversational AI model failed: {e}")
                self.model = None
                self.is_ready = False
                
        except Exception as e:
            print(f"❌ Chatbot system initialization failed: {e}")
            self.is_ready = False
    
    def _get_system_instruction(self) -> str:
        """Get system instruction for the SRE chatbot"""
        return """
You are an expert SRE (Site Reliability Engineering) Assistant with deep knowledge of:

**CORE EXPERTISE:**
- Incident response and management
- System troubleshooting and debugging
- Performance monitoring and alerting
- Infrastructure management and scalability
- DevOps practices and automation
- Database performance and optimization
- Network and security issues
- Cloud platforms (AWS, GCP, Azure)
- Kubernetes and containerization
- CI/CD pipelines and deployment strategies

**COMMUNICATION STYLE:**
- Professional but friendly and approachable
- Clear, actionable advice with specific steps
- Use emojis appropriately to make responses engaging
- Prioritize urgent issues and provide time estimates
- Ask clarifying questions when needed
- Provide both immediate fixes and long-term solutions

**RESPONSE FORMAT:**
- Start with a brief acknowledgment of the issue
- Provide immediate action items for urgent issues
- Explain the reasoning behind recommendations
- Offer follow-up steps and monitoring advice
- Include relevant commands, scripts, or configurations when helpful

**SPECIAL CAPABILITIES:**
- Integrate with production ML incident detection system
- Access real-time system analysis and recommendations
- Generate structured incident reports and post-mortems
- Provide context-aware troubleshooting based on current system state
- Correlate incidents across multiple data sources

**PRIORITIES:**
1. System stability and availability
2. Data integrity and security
3. Performance optimization
4. Team collaboration and knowledge sharing
5. Process improvement and automation

Always provide practical, actionable guidance that helps SRE engineers resolve issues efficiently and prevent future occurrences.
"""
    
    async def chat(self, user_message: str) -> str:
        """
        Main chat interface - processes user message and returns AI response
        """
        try:
            if not self.is_ready:
                return self._get_fallback_response(user_message)
            
            self.total_messages += 1
            current_time = datetime.now().strftime('%H:%M:%S')
            
            print(f"[{current_time}] 👤 User: {user_message}")
            
            # Add user message to conversation history
            self.conversation_history.append({
                'role': 'user',
                'content': user_message,
                'timestamp': datetime.now().isoformat()
            })
            
            # Detect intent and enhance with system context
            enhanced_message = await self._enhance_message_with_context(user_message)
            
            # Generate AI response
            response = await self._generate_ai_response(enhanced_message)
            
            if not response:
                response = "I apologize, but I'm having trouble generating a response right now. Please try asking your question differently, or let me know if you need immediate assistance with a critical incident."
            
            # Add AI response to conversation history
            self.conversation_history.append({
                'role': 'assistant', 
                'content': response,
                'timestamp': datetime.now().isoformat()
            })
            
            # Keep conversation history manageable
            if len(self.conversation_history) > 20:  # Keep last 20 messages
                self.conversation_history = self.conversation_history[-20:]
            
            print(f"[{current_time}] 🤖 SRE Assistant: {response[:100]}..." if len(response) > 100 else f"[{current_time}] 🤖 SRE Assistant: {response}")
            
            return response
            
        except Exception as e:
            error_response = f"I encountered an error while processing your request: {str(e)}. Please try again or contact your SRE team if this is urgent."
            print(f"❌ Chat error: {e}")
            return error_response
    
    async def _enhance_message_with_context(self, user_message: str) -> str:
        """Enhance user message with system context and capabilities"""
        
        try:
            # Detect if user is asking for system analysis
            system_keywords = ['incident', 'analysis', 'status', 'health', 'alert', 'error', 'outage', 'performance']
            
            if any(keyword in user_message.lower() for keyword in system_keywords):
                # Check if we should run system analysis
                if self._should_run_system_analysis(user_message):
                    print("   🔍 Running system analysis for context...")
                    try:
                        # Run quick system analysis
                        if self.sre_engine:
                            analysis_results = await self.sre_engine.run_complete_incident_analysis()
                            self.last_analysis_results = analysis_results
                            
                            # Extract key insights
                            context_summary = self._extract_context_summary(analysis_results)
                            
                            enhanced_message = f"""
USER QUESTION: {user_message}

CURRENT SYSTEM CONTEXT:
{context_summary}

Please provide a response that takes into account the current system state and any relevant insights from the analysis.
"""
                            return enhanced_message
                    except Exception as e:
                        print(f"   ⚠️ System analysis failed: {e}")
            
            # Add conversation context for continuity
            if len(self.conversation_history) > 0:
                recent_context = self.conversation_history[-4:]  # Last 2 exchanges
                context_summary = "RECENT CONVERSATION:\n"
                for msg in recent_context:
                    role = "User" if msg['role'] == 'user' else "Assistant"
                    context_summary += f"{role}: {msg['content'][:100]}...\n" if len(msg['content']) > 100 else f"{role}: {msg['content']}\n"
                
                enhanced_message = f"""
{context_summary}

CURRENT USER MESSAGE: {user_message}

Please respond in context of our conversation, providing helpful SRE guidance.
"""
                return enhanced_message
            
            return user_message
            
        except Exception as e:
            print(f"⚠️ Error enhancing message: {e}")
            return user_message
    
    def _should_run_system_analysis(self, message: str) -> bool:
        """Determine if system analysis should be run based on user message"""
        
        analysis_triggers = [
            'current status', 'system health', 'any incidents', 'what\'s happening',
            'check system', 'analysis', 'alerts', 'errors', 'problems',
            'how are things', 'system state', 'investigate'
        ]
        
        message_lower = message.lower()
        return any(trigger in message_lower for trigger in analysis_triggers)
    
    def _extract_context_summary(self, analysis_results: Dict[str, Any]) -> str:
        """Extract key context from system analysis results"""
        
        try:
            summary = []
            
            # Pipeline metadata
            metadata = analysis_results.get('pipeline_metadata', {})
            if metadata.get('incident_detected'):
                summary.append(f"🚨 ACTIVE INCIDENT DETECTED (Probability: {metadata.get('incident_probability', 0):.3f})")
            else:
                summary.append(f"✅ No incidents detected (Probability: {metadata.get('incident_probability', 0):.3f})")
            
            # ML Analysis
            ml_analysis = analysis_results.get('ml_analysis', {})
            if ml_analysis.get('status') == 'success':
                predictions = ml_analysis.get('predictions', {})
                summary.append(f"📊 ML Analysis: {predictions.get('incident_count', 0)} incidents detected from {predictions.get('samples_processed', 0)} samples")
            
            # NLP Analysis
            nlp_analysis = analysis_results.get('nlp_analysis', {})
            if nlp_analysis.get('processing_status') == 'success':
                nlp_results = nlp_analysis.get('nlp_results', {})
                incident_insights = nlp_results.get('incident_insights', {})
                summary.append(f"🧠 NLP Analysis: {incident_insights.get('incident_severity', 'unknown')} severity")
                
                components = incident_insights.get('affected_components', [])
                if components:
                    summary.append(f"🎯 Affected Components: {', '.join(components[:3])}")
            
            # Recommendations
            recommendations = analysis_results.get('recommendations', {})
            if recommendations:
                priority = recommendations.get('priority_level', 'unknown')
                action_count = len(recommendations.get('immediate_actions', []))
                summary.append(f"💡 Recommendations: {priority.upper()} priority, {action_count} immediate actions")
            
            # System Status
            system_status = analysis_results.get('system_status', {})
            if system_status:
                health = system_status.get('overall_health', 'unknown')
                readiness = system_status.get('incident_response_readiness', 'unknown')
                summary.append(f"📊 System Status: {health.upper()} health, {readiness} response readiness")
            
            return '\n'.join(summary) if summary else "System analysis completed - no significant issues detected"
            
        except Exception as e:
            return f"System analysis available but context extraction failed: {str(e)}"
    
    async def _generate_ai_response(self, enhanced_message: str) -> Optional[str]:
        """Generate AI response using Gemini"""
        
        try:
            if not self.model:
                return self._get_fallback_response(enhanced_message)
            
            # Generate response with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.model.generate_content(enhanced_message)
                    
                    if response and response.text:
                        return response.text.strip()
                    else:
                        print(f"   ⚠️ Empty response from AI on attempt {attempt + 1}")
                        
                except Exception as e:
                    print(f"   ⚠️ AI generation error on attempt {attempt + 1}: {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1)  # Brief wait before retry
            
            return None
            
        except Exception as e:
            print(f"❌ Error generating AI response: {e}")
            return None
    
    def _get_fallback_response(self, user_message: str) -> str:
        """Provide fallback response when AI is not available"""
        
        message_lower = user_message.lower()
        
        # Intent-based fallback responses
        if any(word in message_lower for word in ['hello', 'hi', 'hey']):
            return """
👋 Hello! I'm your SRE Assistant. I'm here to help with:

🚨 **Incident Response & Troubleshooting**
📊 **System Health & Performance**  
🔧 **Infrastructure & Configuration**
📝 **Documentation & Best Practices**

How can I assist you today? (Note: I'm currently running in limited mode - some advanced features may not be available)
"""
        
        elif any(word in message_lower for word in ['help', 'commands', 'what can you do']):
            return """
🤖 **SRE Assistant Capabilities:**

**🚨 Incident Management:**
- Incident analysis and troubleshooting
- Root cause investigation guidance
- Recovery procedures and rollback strategies
- Post-incident review assistance

**📊 System Monitoring:**
- Performance analysis and optimization
- Alert configuration and tuning
- Capacity planning recommendations
- Health check implementations

**🔧 Infrastructure:**
- Server and database troubleshooting
- Network connectivity issues
- Container and Kubernetes problems
- Cloud platform guidance (AWS, GCP, Azure)

**📝 Best Practices:**
- SRE process improvements
- Monitoring and alerting strategies
- Documentation and runbook creation
- Team collaboration enhancement

Ask me about any SRE-related topic or describe your current issue!
"""
        
        elif any(word in message_lower for word in ['incident', 'outage', 'down', 'error']):
            return """
🚨 **Incident Response Guidance:**

**Immediate Steps:**
1. **Assess Impact** - Determine user/business impact
2. **Stabilize** - Stop the bleeding, implement workarounds
3. **Investigate** - Gather logs, metrics, recent changes
4. **Communicate** - Update stakeholders, set expectations
5. **Resolve** - Apply fix, monitor recovery
6. **Follow-up** - Post-incident review, improvements

**Key Questions:**
- What symptoms are you observing?
- When did the issue start?
- What recent changes were made?
- What's the user impact?

Please provide more details about your specific incident for targeted guidance!
"""
        
        elif any(word in message_lower for word in ['performance', 'slow', 'latency']):
            return """
📊 **Performance Troubleshooting:**

**Investigation Steps:**
1. **Metrics Review** - CPU, memory, disk, network utilization
2. **Application Performance** - Response times, throughput, error rates  
3. **Database Performance** - Query times, connection pools, locks
4. **Infrastructure** - Load balancers, CDN, network latency
5. **Recent Changes** - Deployments, configuration changes

**Common Causes:**
- Resource exhaustion (CPU, memory, disk)
- Database query inefficiencies
- Network bottlenecks or timeouts
- Inefficient code or algorithms
- Infrastructure scaling issues

Share your specific performance metrics or symptoms for more targeted advice!
"""
        
        else:
            return """
🤖 I'm here to help with your SRE challenges! 

Currently running in limited mode, but I can still provide guidance on:
- Incident response and troubleshooting
- System performance optimization  
- Infrastructure and configuration
- Best practices and procedures

Please describe your specific issue or question, and I'll do my best to help!

For enhanced AI capabilities, ensure the Gemini API integration is properly configured.
"""
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of current conversation"""
        
        return {
            'session_info': {
                'session_id': self.session_id,
                'started_at': self.session_id,  # Using session_id as timestamp
                'total_messages': self.total_messages,
                'ai_available': self.is_ready
            },
            'conversation_length': len(self.conversation_history),
            'last_analysis': {
                'available': bool(self.last_analysis_results),
                'incident_detected': self.last_analysis_results.get('pipeline_metadata', {}).get('incident_detected', False) if self.last_analysis_results else False
            },
            'system_integration': {
                'sre_engine': bool(self.sre_engine),
                'incident_summarizer': bool(self.incident_summarizer),
                'genai_model': bool(self.model)
            }
        }
    
    def export_conversation(self, filename: Optional[str] = None) -> str:
        """Export conversation history to file"""
        
        if not filename:
            filename = f"sre_chat_session_{self.session_id}.json"
        
        try:
            export_data = {
                'session_summary': self.get_conversation_summary(),
                'conversation_history': self.conversation_history,
                'exported_at': datetime.now().isoformat()
            }
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            return f"✅ Conversation exported to {filename}"
            
        except Exception as e:
            return f"❌ Export failed: {str(e)}"

class SREChatInterface:
    """
    Interactive chat interface for SRE Assistant
    """
    
    def __init__(self):
        """Initialize chat interface"""
        self.chatbot = SREAssistantChatbot()
        self.running = False
    
    async def start_interactive_chat(self):
        """Start interactive chat session"""
        
        print("\n" + "="*60)
        print("🤖 **SRE ASSISTANT CHATBOT - INTERACTIVE MODE**")
        print("="*60)
        print("💬 Your AI-powered SRE expert assistant is ready!")
        print("🎯 Ask about incidents, troubleshooting, system health, etc.")
        print("📝 Type 'help' for capabilities, 'exit' to quit")
        print("="*60)
        
        if not self.chatbot.is_ready:
            print("⚠️  Running in limited mode - some features may not be available")
            print("💡 Install dependencies and configure API keys for full capabilities")
            print("-"*60)
        
        self.running = True
        
        try:
            while self.running:
                # Get user input
                user_input = input("\n👤 You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle special commands
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    print("\n👋 SRE Assistant: Goodbye! Stay reliable! 🚀")
                    break
                
                elif user_input.lower() == 'clear':
                    print("\n" * 50)  # Clear screen
                    print("🧹 Chat history cleared locally (conversation memory preserved)")
                    continue
                
                elif user_input.lower() == 'summary':
                    summary = self.chatbot.get_conversation_summary()
                    print(f"\n📊 **Session Summary:**")
                    print(f"   Messages: {summary['session_info']['total_messages']}")
                    print(f"   AI Mode: {'Full' if summary['session_info']['ai_available'] else 'Limited'}")
                    print(f"   System Integration: {summary['system_integration']}")
                    continue
                
                elif user_input.lower() == 'export':
                    filename = f"sre_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    result = self.chatbot.export_conversation(filename)
                    print(f"\n📁 {result}")
                    continue
                
                # Get AI response
                print("🤖 Thinking...")
                response = await self.chatbot.chat(user_input)
                print(f"\n🤖 **SRE Assistant:** {response}")
                
        except KeyboardInterrupt:
            print("\n\n👋 SRE Assistant: Session ended. Take care!")
        except Exception as e:
            print(f"\n❌ Chat interface error: {str(e)}")
        
        # Session summary
        summary = self.chatbot.get_conversation_summary()
        print(f"\n📊 **Final Session Summary:**")
        print(f"   Total messages: {summary['session_info']['total_messages']}")
        print(f"   Session duration: Interactive")
        print("   Thanks for using SRE Assistant! 🚀")
    
    def stop_chat(self):
        """Stop chat interface"""
        self.running = False

# Convenience functions for different usage patterns
async def quick_sre_question(question: str) -> str:
    """Ask a quick SRE question and get a response"""
    chatbot = SREAssistantChatbot()
    response = await chatbot.chat(question)
    return response

async def sre_chat_session():
    """Start an interactive SRE chat session"""
    interface = SREChatInterface()
    await interface.start_interactive_chat()

def run_sre_chatbot():
    """Run SRE chatbot in interactive mode"""
    try:
        asyncio.run(sre_chat_session())
    except KeyboardInterrupt:
        print("\n👋 SRE Chatbot stopped by user")
    except Exception as e:
        print(f"❌ Error running chatbot: {str(e)}")

# Main execution
if __name__ == "__main__":
    print("🤖 SRE Assistant Chatbot")
    print("="*50)
    print("Choose mode:")
    print("1. Interactive chat session")
    print("2. Quick question demo")
    print("3. System integration test")
    
    try:
        choice = input("Enter choice (1-3, default=1): ").strip() or "1"
        
        if choice == "2":
            async def demo():
                questions = [
                    "Hello! What can you help me with?",
                    "My API is returning 500 errors. What should I check first?",
                    "How do I troubleshoot high CPU usage on a production server?"
                ]
                
                chatbot = SREAssistantChatbot()
                
                for q in questions:
                    print(f"\n👤 Question: {q}")
                    response = await chatbot.chat(q)
                    print(f"🤖 Response: {response[:200]}..." if len(response) > 200 else f"🤖 Response: {response}")
            
            asyncio.run(demo())
        
        elif choice == "3":
            print("🔧 Testing system integration...")
            chatbot = SREAssistantChatbot()
            summary = chatbot.get_conversation_summary()
            print("\n📊 System Integration Status:")
            print(json.dumps(summary, indent=2))
        
        else:
            print("🚀 Starting interactive SRE chat session...")
            run_sre_chatbot()
            
    except KeyboardInterrupt:
        print("\n👋 SRE Assistant stopped")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
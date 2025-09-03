#!/usr/bin/env python3
"""
Quick Demo and Test Script for SRE Assistant Chatbot
Shows how to use the chatbot with your existing SRE system
"""

import asyncio
import json
from datetime import datetime

# Import the chatbot
try:
    from assistant_chatbot import SREAssistantChatbot, quick_sre_question, sre_chat_session
    CHATBOT_AVAILABLE = True
    print("✅ SRE Assistant Chatbot imported successfully")
except ImportError as e:
    print(f"❌ Could not import SRE Assistant Chatbot: {e}")
    CHATBOT_AVAILABLE = False

async def demo_quick_questions():
    """Demo: Quick questions without interactive session"""
    
    print("\n🚀 DEMO: Quick SRE Questions")
    print("="*50)
    
    demo_questions = [
        "Hello! What can you help me with?",
        "My application is showing high latency. What should I check first?", 
        "How do I investigate a database connection issue?",
        "What are the best practices for incident response?",
        "Can you help me understand why my API is returning 500 errors?",
        "How do I check current system status and health?"
    ]
    
    if not CHATBOT_AVAILABLE:
        print("❌ Chatbot not available for demo")
        return
    
    chatbot = SREAssistantChatbot()
    
    for i, question in enumerate(demo_questions, 1):
        print(f"\n📝 Question {i}: {question}")
        print("🤖 Thinking...")
        
        try:
            response = await chatbot.chat(question)
            print(f"💬 Response: {response[:300]}..." if len(response) > 300 else f"💬 Response: {response}")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        
        print("-" * 50)
    
    # Show session summary
    summary = chatbot.get_conversation_summary()
    print(f"\n📊 Demo Session Summary:")
    print(f"   Questions asked: {summary['session_info']['total_messages']}")
    print(f"   AI mode: {'Full' if summary['session_info']['ai_available'] else 'Limited'}")
    print(f"   System integration: {summary['system_integration']}")

async def demo_incident_scenario():
    """Demo: Incident response scenario"""
    
    print("\n🚨 DEMO: Incident Response Scenario")
    print("="*50)
    
    incident_questions = [
        "We have a critical incident! Our main API is down and users can't access the service.",
        "The API started failing 15 minutes ago with 500 errors. What immediate steps should I take?",
        "I've checked the logs and see database connection timeouts. How do I troubleshoot this?",
        "The database seems overloaded with CPU at 95%. What are the immediate remediation steps?",
        "We've restarted the database and it's recovering. How do I monitor the recovery?",
        "The incident is resolved. What should I document for the post-mortem?"
    ]
    
    if not CHATBOT_AVAILABLE:
        print("❌ Chatbot not available for demo")
        return
    
    chatbot = SREAssistantChatbot()
    
    print("🎬 **Scenario**: Critical API outage with database issues")
    print("📅 **Timeline**: Simulating real incident response conversation")
    
    for i, question in enumerate(incident_questions, 1):
        print(f"\n⏰ **Step {i}**: {question}")
        print("🤖 SRE Assistant thinking...")
        
        try:
            response = await chatbot.chat(question)
            print(f"📋 **SRE Assistant**: {response}")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        
        print("\n" + "="*50)
    
    # Export the incident conversation
    export_result = chatbot.export_conversation(f"incident_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    print(f"\n📁 {export_result}")

async def test_system_integration():
    """Test integration with your existing SRE system"""
    
    print("\n🔧 SYSTEM INTEGRATION TEST")
    print("="*50)
    
    if not CHATBOT_AVAILABLE:
        print("❌ Chatbot not available for integration test")
        return
    
    chatbot = SREAssistantChatbot()
    
    # Test system integration
    print("1️⃣ Testing chatbot initialization...")
    summary = chatbot.get_conversation_summary()
    print(f"   ✅ Chatbot initialized: {summary['session_info']['ai_available']}")
    print(f"   ✅ SRE Engine: {summary['system_integration']['sre_engine']}")
    print(f"   ✅ Incident Summarizer: {summary['system_integration']['incident_summarizer']}")
    print(f"   ✅ GenAI Model: {summary['system_integration']['genai_model']}")
    
    # Test system status inquiry
    print("\n2️⃣ Testing system status inquiry...")
    system_question = "Can you check the current system status and tell me if there are any incidents?"
    
    try:
        response = await chatbot.chat(system_question)
        print(f"   💬 Response received: {len(response)} characters")
        print(f"   📝 Sample: {response[:200]}...")
        
        if "analysis" in response.lower():
            print("   ✅ System integration working - real analysis performed")
        else:
            print("   ⚠️ Limited mode - using fallback responses")
            
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
    
    # Test conversation memory
    print("\n3️⃣ Testing conversation memory...")
    followup_question = "Based on what you just told me, what should I monitor most closely?"
    
    try:
        response = await chatbot.chat(followup_question)
        print(f"   💬 Follow-up response: {response[:150]}...")
        
        if len(chatbot.conversation_history) >= 4:  # 2 exchanges
            print("   ✅ Conversation memory working")
        else:
            print("   ⚠️ Conversation memory limited")
            
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
    
    print(f"\n📊 Final Integration Status:")
    final_summary = chatbot.get_conversation_summary()
    print(json.dumps(final_summary, indent=2))

def show_usage_examples():
    """Show different ways to use the SRE chatbot"""
    
    print("\n📚 SRE CHATBOT USAGE EXAMPLES")
    print("="*50)
    
    print("""
🔥 **1. Interactive Chat Session:**
```python
from sre_assistant_chatbot import run_sre_chatbot
run_sre_chatbot()
```

💬 **2. Quick Questions:**
```python
import asyncio
from sre_assistant_chatbot import quick_sre_question

response = asyncio.run(quick_sre_question("How do I troubleshoot high CPU usage?"))
print(response)
```

🤖 **3. Programmatic Integration:**
```python
from sre_assistant_chatbot import SREAssistantChatbot

async def main():
    chatbot = SREAssistantChatbot()
    
    # Ask multiple questions
    questions = [
        "What's the current system status?",
        "How do I investigate API latency issues?",
        "What are immediate steps for database outage?"
    ]
    
    for question in questions:
        response = await chatbot.chat(question)
        print(f"Q: {question}")
        print(f"A: {response}")
        print("-" * 50)

asyncio.run(main())
```

🚨 **4. Incident Response Integration:**
```python
from sre_assistant_chatbot import SREAssistantChatbot
from complete_sre_engine import CompleteSREInsightEngine

async def incident_response_workflow():
    # Initialize systems
    sre_engine = CompleteSREInsightEngine()
    chatbot = SREAssistantChatbot()
    
    # Run analysis
    analysis = await sre_engine.run_complete_incident_analysis()
    
    # Get AI guidance
    if analysis['pipeline_metadata']['incident_detected']:
        response = await chatbot.chat(
            f"We detected an incident with {analysis['pipeline_metadata']['incident_probability']:.1%} confidence. "
            "What immediate actions should we take?"
        )
        print(f"🤖 AI Guidance: {response}")

asyncio.run(incident_response_workflow())
```

📱 **5. Command Line Usage:**
```bash
# Interactive mode
python sre_assistant_chatbot.py

# Quick demo
python demo_sre_chatbot.py demo

# Integration test  
python demo_sre_chatbot.py test
```

🔧 **6. Custom Integration:**
```python
class CustomSREWorkflow:
    def __init__(self):
        self.chatbot = SREAssistantChatbot()
    
    async def get_troubleshooting_advice(self, issue_description):
        prompt = f'''
        Issue: {issue_description}
        
        Please provide:
        1. Immediate investigation steps
        2. Common causes to check
        3. Remediation strategies
        4. Prevention measures
        '''
        
        return await self.chatbot.chat(prompt)
```

🎯 **7. Integration with Monitoring:**
```python
# When your monitoring detects an issue
async def on_alert_triggered(alert_data):
    chatbot = SREAssistantChatbot()
    
    guidance = await chatbot.chat(f'''
    Alert triggered: {alert_data['alert_name']}
    Severity: {alert_data['severity']}
    Metrics: {alert_data['metrics']}
    
    What should our on-call engineer do first?
    ''')
    
    # Send guidance to on-call team
    await send_to_oncall_channel(guidance)
```
""")

async def main():
    """Main demo function"""
    
    print("🤖 **SRE ASSISTANT CHATBOT DEMO & TEST SUITE**")
    print("="*60)
    print("🎯 Testing your new AI-powered SRE assistant")
    print("💬 Integrated with your existing incident response system")
    print("🧠 Powered by Gemini AI with your configured API key")
    print("="*60)
    
    if not CHATBOT_AVAILABLE:
        print("❌ SRE Assistant Chatbot not available")
        print("💡 Make sure all required files are in place and dependencies installed")
        return
    
    print("\n🚀 Choose demo mode:")
    print("1. Quick questions demo")
    print("2. Incident response scenario")  
    print("3. System integration test")
    print("4. Usage examples")
    print("5. Interactive chat session")
    print("6. All demos")
    
    try:
        choice = input("\nEnter choice (1-6, default=1): ").strip() or "1"
        
        if choice == "1":
            await demo_quick_questions()
        elif choice == "2":
            await demo_incident_scenario()
        elif choice == "3":
            await test_system_integration()
        elif choice == "4":
            show_usage_examples()
        elif choice == "5":
            print("🚀 Starting interactive chat session...")
            await sre_chat_session()
        elif choice == "6":
            print("🎬 Running all demos...")
            await demo_quick_questions()
            await demo_incident_scenario() 
            await test_system_integration()
            show_usage_examples()
        else:
            print("Invalid choice, running quick questions demo...")
            await demo_quick_questions()
            
    except KeyboardInterrupt:
        print("\n👋 Demo stopped by user")
    except Exception as e:
        print(f"❌ Demo error: {str(e)}")
    
    print("\n✅ **SRE ASSISTANT CHATBOT DEMO COMPLETE!**")
    print("🚀 Your AI-powered SRE assistant is ready for production use!")
    print("💡 Run 'python sre_assistant_chatbot.py' for interactive mode")

if __name__ == "__main__":
    # Command line arguments
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "demo":
            print("🚀 Running chatbot demo...")
            asyncio.run(demo_quick_questions())
        elif sys.argv[1] == "incident":
            print("🚨 Running incident scenario...")
            asyncio.run(demo_incident_scenario())
        elif sys.argv[1] == "test":
            print("🔧 Running system integration test...")
            asyncio.run(test_system_integration())
        elif sys.argv[1] == "examples":
            show_usage_examples()
        else:
            print("❌ Unknown command. Use: demo, incident, test, or examples")
    else:
        asyncio.run(main())
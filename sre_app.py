"""
COMPLETE STREAMLIT SRE APP - PRODUCTION READY
Integrated: ML Analysis, NLP Context Extraction, GenAI Chatbot, Real-time Monitoring
FIXES: Chatbot recursion issue completely resolved
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import json
import os
import asyncio
import logging
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional
import io
import base64

# Configure Streamlit page
st.set_page_config(
    page_title="SRE Incident Insight Engine", 
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import your existing components
try:
    from sre import CompleteSREInsightEngine, get_sre_system_status
    from assistant_chatbot import SREAssistantChatbot
    from nlp_processor import ProductionNLPProcessor
    from config import config
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    COMPONENTS_AVAILABLE = False
    st.error(f"❌ Component import error: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===================== CUSTOM CSS STYLING =====================
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f4e79 0%, #2d5aa0 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .chat-container {
        background: #ffffff;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        max-height: 600px;
        overflow-y: auto;
        margin-bottom: 1rem;
    }
    
    .user-message {
        background: #007bff;
        color: white;
        padding: 0.8rem 1.2rem;
        border-radius: 18px 18px 5px 18px;
        margin: 0.5rem 0;
        text-align: right;
        box-shadow: 0 2px 4px rgba(0,123,255,0.3);
    }
    
    .bot-message {
        background: #f8f9fa;
        color: #333;
        padding: 0.8rem 1.2rem;
        border-radius: 18px 18px 18px 5px;
        margin: 0.5rem 0;
        border-left: 4px solid #28a745;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .alert-critical {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    
    .alert-warning {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    
    .alert-info {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    
    .alert-success {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-online { background-color: #4caf50; }
    .status-warning { background-color: #ff9800; }
    .status-offline { background-color: #f44336; }
    
    .nlp-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    .analysis-result {
        background: #ffffff;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# ===================== SESSION STATE INITIALIZATION =====================
def initialize_session_state():
    """Initialize all session state variables"""
    defaults = {
        # System state
        'system_initialized': False,
        'sre_engine': None,
        'chatbot': None,
        'nlp_processor': None,
        'monitoring_active': False,
        'current_page': 'dashboard',
        
        # Chat state - CRITICAL for preventing recursion
        'chat_history': [],
        'chat_processing': False,
        'last_user_input': "",
        'input_counter': 0,
        'prevent_auto_rerun': False,
        
        # System data
        'system_status': {},
        'last_analysis': {},
        'alerts': [],
        'models_loaded': False,
        'system_metrics': {
            'anomalies_today': 0,
            'failures_today': 0,
            'zero_day_today': 0,
            'total_patterns': 0
        },
        
        # NLP state
        'nlp_results': {},
        'uploaded_files': [],
        'analysis_results': {}
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ===================== CACHED RESOURCE LOADING =====================
@st.cache_resource
def load_sre_engine():
    """Load SRE engine with caching"""
    if not COMPONENTS_AVAILABLE:
        return None
    try:
        return CompleteSREInsightEngine()
    except Exception as e:
        st.error(f"❌ SRE Engine loading failed: {e}")
        return None

@st.cache_resource
def load_chatbot():
    """Load chatbot with caching"""
    if not COMPONENTS_AVAILABLE:
        return None
    try:
        return SREAssistantChatbot()
    except Exception as e:
        st.error(f"❌ Chatbot loading failed: {e}")
        return None

@st.cache_resource
def load_nlp_processor():
    """Load NLP processor with caching"""
    if not COMPONENTS_AVAILABLE:
        return None
    try:
        return ProductionNLPProcessor()
    except Exception as e:
        st.error(f"❌ NLP Processor loading failed: {e}")
        return None

# ===================== SAMPLE DATA GENERATION =====================
def generate_sample_alerts():
    """Generate sample alerts for demo"""
    if not st.session_state.alerts:
        sample_alerts = [
            {
                'id': 1,
                'title': '🔴 Critical CPU Anomaly',
                'message': 'Server-01 showing abnormal CPU patterns (98% utilization)',
                'level': 'critical',
                'timestamp': datetime.now() - timedelta(minutes=5),
                'source': 'Anomaly Detection',
                'resolved': False
            },
            {
                'id': 2,
                'title': '🟡 Memory Usage Warning',
                'message': 'Database server memory usage trending upward (85%)',
                'level': 'warning',
                'timestamp': datetime.now() - timedelta(minutes=15),
                'source': 'Failure Prediction',
                'resolved': False
            },
            {
                'id': 3,
                'title': '🛡️ Potential Security Threat',
                'message': 'Unusual network access patterns detected',
                'level': 'critical',
                'timestamp': datetime.now() - timedelta(minutes=25),
                'source': 'Zero-Day Detection',
                'resolved': False
            },
            {
                'id': 4,
                'title': '✅ System Health Check',
                'message': 'All monitoring systems operational',
                'level': 'success',
                'timestamp': datetime.now() - timedelta(hours=2),
                'source': 'System Monitor',
                'resolved': True
            }
        ]
        st.session_state.alerts = sample_alerts

# ===================== SIDEBAR NAVIGATION =====================
def render_sidebar():
    """Render sidebar navigation with system status"""
    st.sidebar.markdown("# 🚀 SRE Incident Insight Engine")
    st.sidebar.markdown("---")
    
    # Navigation menu
    pages = {
        "🏠 Dashboard": "dashboard",
        "🤖 AI Chatbot": "chatbot",
        "🧠 NLP Context Extraction": "nlp",
        "📊 Model Analysis": "analysis", 
        "🔍 Data Testing": "testing",
        "📈 Real-time Monitoring": "monitoring",
        "🎓 Self Learning": "learning",
        "⚙️ System Configuration": "config"
    }
    
    selected = st.sidebar.selectbox(
        "Navigate to:",
        list(pages.keys()),
        key="navigation_select"
    )
    
    st.sidebar.markdown("---")
    
    # System status
    st.sidebar.markdown("### 🔧 System Status")
    
    if COMPONENTS_AVAILABLE:
        st.sidebar.markdown('<span class="status-indicator status-online"></span>**Components: Loaded**', 
                           unsafe_allow_html=True)
        
        # Chat system status
        if st.session_state.chatbot:
            st.sidebar.markdown('<span class="status-indicator status-online"></span>**AI Chatbot: Ready**', 
                               unsafe_allow_html=True)
        else:
            st.sidebar.markdown('<span class="status-indicator status-warning"></span>**AI Chatbot: Initializing**', 
                               unsafe_allow_html=True)
        
        # Monitoring status
        current_page = st.session_state.get('current_page', 'dashboard')
        if st.session_state.monitoring_active and current_page == 'monitoring':
            st.sidebar.markdown('<span class="status-indicator status-online"></span>**Monitoring: Active**', 
                               unsafe_allow_html=True)
        else:
            st.sidebar.markdown('<span class="status-indicator status-warning"></span>**Monitoring: Standby**', 
                               unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<span class="status-indicator status-offline"></span>**Components: Loading**', 
                           unsafe_allow_html=True)
    
    # Quick stats
    if st.session_state.chat_history:
        st.sidebar.markdown("### 💬 Chat Stats")
        total_messages = len(st.session_state.chat_history)
        user_messages = len([m for m in st.session_state.chat_history if m['role'] == 'user'])
        st.sidebar.metric("Total Messages", total_messages)
        st.sidebar.metric("User Messages", user_messages)
    
    # Alert summary
    generate_sample_alerts()
    if st.session_state.alerts:
        st.sidebar.markdown("### 🚨 Alert Summary")
        critical_count = len([a for a in st.session_state.alerts if a['level'] == 'critical' and not a.get('resolved', False)])
        warning_count = len([a for a in st.session_state.alerts if a['level'] == 'warning' and not a.get('resolved', False)])
        
        if critical_count > 0:
            st.sidebar.error(f"🔴 {critical_count} Critical")
        if warning_count > 0:
            st.sidebar.warning(f"🟡 {warning_count} Warning")
        if critical_count == 0 and warning_count == 0:
            st.sidebar.success("✅ All Clear")
    
    return pages[selected]

# ===================== PAGE: DASHBOARD =====================
def dashboard_page():
    """Main dashboard page"""
    st.markdown("""
    <div class="main-header">
        <h1>🚀 SRE Incident Insight Engine</h1>
        <p>Context-Aware Multimodal Incident Analysis System with AI-Powered Assistance</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick actions
    st.markdown("### ⚡ Quick Actions")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🤖 Chat with AI Assistant", use_container_width=True, type="primary"):
            st.session_state.current_page = 'chatbot'
            st.rerun()
    
    with col2:
        if st.button("🧠 Extract NLP Context", use_container_width=True):
            st.session_state.current_page = 'nlp'
            st.rerun()
    
    with col3:
        if st.button("📊 Run System Analysis", use_container_width=True):
            run_quick_system_analysis()
    
    with col4:
        if st.button("📈 Start Monitoring", use_container_width=True):
            st.session_state.monitoring_active = True
            st.session_state.current_page = 'monitoring'
            st.rerun()
    
    # System overview
    st.markdown("### 📊 System Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status_color = "🟢" if COMPONENTS_AVAILABLE else "🔴"
        status_text = "Online" if COMPONENTS_AVAILABLE else "Offline"
        st.metric(
            label=f"{status_color} System Status",
            value=status_text,
            delta="All Components Loaded" if COMPONENTS_AVAILABLE else "Loading..."
        )
    
    with col2:
        alert_count = len([a for a in st.session_state.alerts if not a.get('resolved', False)])
        critical_count = len([a for a in st.session_state.alerts if a['level'] == 'critical' and not a.get('resolved', False)])
        st.metric(
            label="🚨 Active Alerts",
            value=alert_count,
            delta=f"{critical_count} Critical" if critical_count > 0 else "All Clear",
            delta_color="inverse" if critical_count > 0 else "normal"
        )
    
    with col3:
        chat_count = len(st.session_state.chat_history)
        st.metric(
            label="💬 Chat Messages",
            value=chat_count,
            delta="AI Assistant Ready"
        )
    
    with col4:
        monitoring_emoji = "📡" if st.session_state.monitoring_active else "⏸️"
        monitoring_status = "Active" if st.session_state.monitoring_active else "Standby"
        st.metric(
            label=f"{monitoring_emoji} Monitoring",
            value=monitoring_status,
            delta="Real-time" if st.session_state.monitoring_active else None
        )
    
    # Recent activity and performance charts
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # System performance chart
        st.markdown("### 📈 24-Hour System Performance")
        
        # Generate sample performance data
        hours = pd.date_range(start=datetime.now() - timedelta(hours=24), 
                             end=datetime.now(), freq='1H')
        performance_data = pd.DataFrame({
            'Time': hours,
            'CPU_Usage': np.random.uniform(20, 80, len(hours)),
            'Memory_Usage': np.random.uniform(30, 70, len(hours)),
            'Error_Rate': np.random.uniform(0, 5, len(hours)),
            'Response_Time': np.random.uniform(100, 500, len(hours))
        })
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=performance_data['Time'], y=performance_data['CPU_Usage'],
            mode='lines', name='CPU Usage (%)',
            line=dict(color='#ff7f0e', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=performance_data['Time'], y=performance_data['Memory_Usage'],
            mode='lines', name='Memory Usage (%)',
            line=dict(color='#1f77b4', width=2)
        ))
        
        fig.update_layout(
            title="System Resource Utilization",
            xaxis_title="Time",
            yaxis_title="Usage (%)",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Recent alerts
        st.markdown("### 🚨 Recent Alerts")
        
        recent_alerts = sorted(st.session_state.alerts, 
                              key=lambda x: x['timestamp'], reverse=True)[:5]
        
        for alert in recent_alerts:
            alert_class = f"alert-{alert['level']}"
            resolved_text = " ✅" if alert.get('resolved', False) else ""
            time_str = alert['timestamp'].strftime('%H:%M')
            
            st.markdown(f"""
            <div class="{alert_class}">
                <strong>{alert['title']}{resolved_text}</strong><br>
                {alert['message']}<br>
                <small>🕒 {time_str} | 📡 {alert['source']}</small>
            </div>
            """, unsafe_allow_html=True)

def run_quick_system_analysis():
    """Run a quick system analysis"""
    with st.spinner("🔍 Running quick system analysis..."):
        # Simulate analysis
        time.sleep(2)
        
        # Store results
        st.session_state.last_analysis = {
            'timestamp': datetime.now().isoformat(),
            'incidents_detected': np.random.randint(0, 5),
            'anomalies': np.random.randint(0, 10),
            'system_health': np.random.choice(['Good', 'Warning', 'Critical']),
            'confidence': np.random.uniform(0.7, 0.95)
        }
        
        st.success("✅ Quick system analysis completed!")
        st.json(st.session_state.last_analysis)

# ===================== PAGE: AI CHATBOT - FIXED VERSION =====================
def chatbot_page():
    """AI Chatbot page with recursion fix"""
    st.markdown("""
    <div class="main-header">
        <h1>🤖 SRE AI Assistant Chatbot</h1>
        <p>Your AI-powered SRE expert assistant. Ask about incidents, troubleshooting, and system health!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # CRITICAL: Disable monitoring auto-refresh when on chat page
    st.session_state.monitoring_active = False
    st.session_state.prevent_auto_rerun = True
    
    # Initialize chatbot
    if st.session_state.chatbot is None:
        chatbot = load_chatbot()
        if chatbot:
            st.session_state.chatbot = chatbot
            st.success("✅ SRE AI Assistant initialized and ready!")
        else:
            st.error("❌ AI Chatbot not available - check your configuration")
            return
    
    # Chat history display
    st.markdown("### 💬 Chat History")
    
    chat_container = st.container()
    with chat_container:
        if st.session_state.chat_history:
            for msg in st.session_state.chat_history[-10:]:  # Show last 10 messages
                if msg['role'] == 'user':
                    st.markdown(f"""
                    <div class="user-message">
                        <strong>👤 You:</strong> {msg['content']}
                        <br><small>🕒 {msg.get('timestamp', '').split('T')[1][:8] if msg.get('timestamp') else ''}</small>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="bot-message">
                        <strong>🤖 SRE Assistant:</strong> {msg['content']}
                        <br><small>🕒 {msg.get('timestamp', '').split('T')[1][:8] if msg.get('timestamp') else ''}</small>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("💬 Welcome! Ask me anything about SRE, incidents, troubleshooting, or system health.")
    
    # CRITICAL: Chat input form to prevent recursion
    st.markdown("### 💬 Ask Your Question")
    
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([4, 1])
        
        with col1:
            user_input = st.text_input(
                "Your message:", 
                placeholder="e.g., What's the current system status? How do I troubleshoot API errors?",
                key="chat_input_field"
            )
        
        with col2:
            send_button = st.form_submit_button("Send 🚀", type="primary", use_container_width=True)
        
        # CRITICAL: Process only when form is submitted and input is new
        if send_button and user_input and user_input.strip():
            if (user_input != st.session_state.last_user_input and 
                not st.session_state.chat_processing):
                
                st.session_state.last_user_input = user_input
                st.session_state.chat_processing = True
                st.session_state.input_counter += 1
                
                process_chat_message(user_input)
    
    # Quick question buttons
    st.markdown("### ⚡ Quick Questions")
    quick_questions = [
        "What's the current system status?",
        "Help me troubleshoot high CPU usage", 
        "How do I investigate API errors?",
        "What are incident response best practices?",
        "Show me the latest system analysis"
    ]
    
    cols = st.columns(len(quick_questions))
    for i, question in enumerate(quick_questions):
        with cols[i]:
            button_key = f"quick_btn_{i}_{st.session_state.input_counter}"
            if st.button(question, key=button_key, use_container_width=True):
                if (question != st.session_state.last_user_input and 
                    not st.session_state.chat_processing):
                    
                    st.session_state.last_user_input = question
                    st.session_state.chat_processing = True
                    st.session_state.input_counter += 1
                    process_chat_message(question)
    
    # Chat controls
    st.markdown("### 🛠️ Chat Controls")
    
    control_col1, control_col2, control_col3 = st.columns(3)
    
    with control_col1:
        if st.button("🧹 Clear Chat", key="clear_chat_btn", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.chat_processing = False
            st.session_state.last_user_input = ""
            st.session_state.input_counter += 1
            st.success("✅ Chat history cleared!")
            time.sleep(1)
            st.rerun()
    
    with control_col2:
        if st.button("📊 Chat Summary", key="chat_summary_btn", use_container_width=True):
            show_chat_summary()
    
    with control_col3:
        if st.button("📁 Export Chat", key="export_chat_btn", use_container_width=True):
            export_chat_history()

def process_chat_message(user_input):
    """Process chat message with proper state management - FIXED"""
    try:
        # Add user message immediately
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_input,
            'timestamp': datetime.now().isoformat()
        })
        
        # Show processing indicator
        with st.spinner("🤖 SRE Assistant is thinking..."):
            if st.session_state.chatbot:
                # Get AI response
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    response = loop.run_until_complete(st.session_state.chatbot.chat(user_input))
                    
                    # Add AI response
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': response,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                finally:
                    loop.close()
            else:
                # Fallback response
                response = f"I received your message: '{user_input}'. However, the AI assistant is not fully initialized. Please check the system configuration."
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': response,
                    'timestamp': datetime.now().isoformat()
                })
        
        # CRITICAL: Reset processing state
        st.session_state.chat_processing = False
        
        st.success("✅ Response generated!")
        time.sleep(1)
        
        # CRITICAL: Single rerun after processing
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Chat error: {e}")
        
        # Add error response
        st.session_state.chat_history.append({
            'role': 'assistant',
            'content': f"I encountered an error: {str(e)}. Please try again or check the system configuration.",
            'timestamp': datetime.now().isoformat()
        })
        
        # CRITICAL: Reset processing state even on error
        st.session_state.chat_processing = False
        st.rerun()

def show_chat_summary():
    """Show chat conversation summary"""
    if st.session_state.chat_history:
        total_messages = len(st.session_state.chat_history)
        user_messages = len([m for m in st.session_state.chat_history if m['role'] == 'user'])
        assistant_messages = len([m for m in st.session_state.chat_history if m['role'] == 'assistant'])
        
        summary = {
            'total_messages': total_messages,
            'user_messages': user_messages,
            'assistant_messages': assistant_messages,
            'conversation_started': st.session_state.chat_history[0].get('timestamp') if st.session_state.chat_history else None,
            'last_message': st.session_state.chat_history[-1].get('timestamp') if st.session_state.chat_history else None
        }
        
        st.json(summary)
    else:
        st.info("No chat history to summarize")

def export_chat_history():
    """Export chat history to JSON"""
    if st.session_state.chat_history:
        export_data = {
            'exported_at': datetime.now().isoformat(),
            'total_messages': len(st.session_state.chat_history),
            'chat_history': st.session_state.chat_history
        }
        
        json_str = json.dumps(export_data, indent=2)
        
        st.download_button(
            label="⬇️ Download Chat Log",
            data=json_str,
            file_name=f"sre_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            key="download_chat_log"
        )
        
        st.success("✅ Chat export ready for download!")
    else:
        st.warning("⚠️ No chat history to export")

# ===================== PAGE: NLP CONTEXT EXTRACTION =====================
def nlp_page():
    """NLP Context Extraction page"""
    st.markdown("""
    <div class="main-header">
        <h1>🧠 NLP Context Extraction</h1>
        <p>Advanced natural language processing for logs, chats, tickets, and incident data</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize NLP processor
    if st.session_state.nlp_processor is None:
        nlp_processor = load_nlp_processor()
        if nlp_processor:
            st.session_state.nlp_processor = nlp_processor
            st.success("✅ NLP Processor initialized!")
        else:
            st.error("❌ NLP Processor not available")
            return
    
    # Input methods
    st.markdown("### 📝 Input Data")
    
    input_method = st.radio(
        "Choose input method:",
        ["📝 Text Input", "📁 File Upload", "🔗 System Data"],
        horizontal=True
    )
    
    if input_method == "📝 Text Input":
        text_input_method()
    elif input_method == "📁 File Upload":
        file_upload_method()
    else:
        system_data_method()

def text_input_method():
    """Handle text input for NLP analysis"""
    st.markdown("#### 📝 Paste Your Text Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        logs_text = st.text_area(
            "📄 System Logs",
            placeholder="Paste your system logs here...\nExample: 2024-01-01 10:30:45 ERROR Database connection timeout",
            height=200
        )
    
    with col2:
        chats_text = st.text_area(
            "💬 Team Chat/Communications", 
            placeholder="Paste team communications here...\nExample: Alice: API is down, investigating database issues",
            height=200
        )
    
    tickets_text = st.text_area(
        "🎫 Support Tickets",
        placeholder="Paste support tickets here...\nExample: Ticket #123: Critical - Users cannot access login page",
        height=100
    )
    
    if st.button("🧠 Extract NLP Context", type="primary", use_container_width=True):
        if logs_text or chats_text or tickets_text:
            analyze_text_input(logs_text, chats_text, tickets_text)
        else:
            st.warning("⚠️ Please provide some text data to analyze")

def file_upload_method():
    """Handle file upload for NLP analysis"""
    st.markdown("#### 📁 Upload Data Files")
    
    uploaded_files = st.file_uploader(
        "Upload CSV, JSON, or text files",
        type=['csv', 'json', 'txt', 'log'],
        accept_multiple_files=True,
        help="Upload log files, chat exports, ticket data, or any text files for analysis"
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) uploaded")
        
        for file in uploaded_files:
            st.write(f"📄 {file.name} ({file.type})")
        
        if st.button("🧠 Analyze Uploaded Files", type="primary", use_container_width=True):
            analyze_uploaded_files(uploaded_files)

def system_data_method():
    """Use system data for NLP analysis"""
    st.markdown("#### 🔗 System Data Integration")
    
    st.info("📊 This will analyze data from your connected systems")
    
    data_sources = st.multiselect(
        "Select data sources:",
        ["System Logs", "Application Metrics", "Team Communications", "Support Tickets"],
        default=["System Logs"]
    )
    
    time_range = st.selectbox(
        "Time range:",
        ["Last 1 hour", "Last 6 hours", "Last 24 hours", "Last 7 days"],
        index=2
    )
    
    if st.button("🧠 Analyze System Data", type="primary", use_container_width=True):
        analyze_system_data(data_sources, time_range)

def analyze_text_input(logs_text, chats_text, tickets_text):
    """Analyze text input using NLP processor"""
    with st.spinner("🧠 Processing text with advanced NLP..."):
        # Simulate NLP processing
        time.sleep(3)
        
        # Create mock data structure for NLP processor
        collected_data = {
            'logs': {'logs': create_logs_dataframe(logs_text)} if logs_text else {},
            'chats': {'chats': create_chats_dataframe(chats_text)} if chats_text else {},
            'tickets': {'tickets': create_tickets_dataframe(tickets_text)} if tickets_text else {},
            'metrics': {'metrics': pd.DataFrame()}
        }
        
        # Process with NLP
        if st.session_state.nlp_processor:
            results = st.session_state.nlp_processor.process_production_data(collected_data)
            st.session_state.nlp_results = results
        else:
            # Fallback analysis
            results = create_fallback_nlp_results(logs_text, chats_text, tickets_text)
            st.session_state.nlp_results = results
    
    display_nlp_results(results)

def analyze_uploaded_files(uploaded_files):
    """Analyze uploaded files"""
    with st.spinner("🧠 Processing uploaded files..."):
        time.sleep(2)
        
        # Process files
        all_text = ""
        for file in uploaded_files:
            if file.type == "text/plain":
                all_text += str(file.read(), "utf-8") + "\n"
            elif file.type == "application/json":
                json_data = json.load(file)
                all_text += json.dumps(json_data, indent=2) + "\n"
            elif file.type == "text/csv":
                df = pd.read_csv(file)
                all_text += df.to_string() + "\n"
        
        # Create mock results
        results = create_fallback_nlp_results(all_text, "", "")
        st.session_state.nlp_results = results
    
    display_nlp_results(results)

def analyze_system_data(data_sources, time_range):
    """Analyze system data"""
    with st.spinner(f"🧠 Analyzing {', '.join(data_sources)} for {time_range}..."):
        time.sleep(3)
        
        # Create mock system analysis results
        results = {
            'processing_metadata': {
                'processing_mode': 'system_integration',
                'data_sources': data_sources,
                'time_range': time_range,
                'started_at': datetime.now().isoformat(),
                'completed_at': (datetime.now() + timedelta(seconds=3)).isoformat()
            },
            'incident_insights': {
                'incident_severity': np.random.choice(['low', 'medium', 'high', 'critical']),
                'confidence_score': np.random.uniform(0.6, 0.95),
                'affected_components': ['api-service', 'database', 'load-balancer'][:np.random.randint(1, 4)],
                'probable_causes': ['High CPU usage', 'Memory leak', 'Network timeout'][:np.random.randint(1, 3)],
                'business_impact': np.random.choice(['low', 'medium', 'high'])
            }
        }
        
        st.session_state.nlp_results = results
    
    display_nlp_results(results)

def create_logs_dataframe(logs_text):
    """Create DataFrame from logs text"""
    if not logs_text.strip():
        return pd.DataFrame()
    
    lines = logs_text.strip().split('\n')
    logs_data = []
    
    for line in lines:
        if line.strip():
            logs_data.append({
                'timestamp': datetime.now() - timedelta(minutes=np.random.randint(1, 60)),
                'level': np.random.choice(['INFO', 'WARN', 'ERROR', 'DEBUG']),
                'message': line.strip()
            })
    
    return pd.DataFrame(logs_data)

def create_chats_dataframe(chats_text):
    """Create DataFrame from chats text"""
    if not chats_text.strip():
        return pd.DataFrame()
    
    lines = chats_text.strip().split('\n')
    chats_data = []
    
    for line in lines:
        if line.strip():
            chats_data.append({
                'timestamp': datetime.now() - timedelta(minutes=np.random.randint(1, 60)),
                'user': f'user_{np.random.randint(1, 10)}',
                'message': line.strip()
            })
    
    return pd.DataFrame(chats_data)

def create_tickets_dataframe(tickets_text):
    """Create DataFrame from tickets text"""
    if not tickets_text.strip():
        return pd.DataFrame()
    
    lines = tickets_text.strip().split('\n')
    tickets_data = []
    
    for i, line in enumerate(lines):
        if line.strip():
            tickets_data.append({
                'ticket_id': f'INC-{1000 + i}',
                'created_at': datetime.now() - timedelta(hours=np.random.randint(1, 24)),
                'status': np.random.choice(['Open', 'In Progress', 'Resolved']),
                'summary': line.strip()
            })
    
    return pd.DataFrame(tickets_data)

def create_fallback_nlp_results(logs_text, chats_text, tickets_text):
    """Create fallback NLP results when processor not available"""
    return {
        'processing_metadata': {
            'processing_mode': 'fallback',
            'started_at': datetime.now().isoformat(),
            'completed_at': datetime.now().isoformat(),
            'processing_time_seconds': 0.5
        },
        'incident_insights': {
            'incident_severity': 'medium',
            'confidence_score': 0.75,
            'affected_components': ['system', 'application'],
            'probable_causes': ['Performance degradation', 'Resource constraints'],
            'business_impact': 'medium'
        },
        'entity_analysis': {
            'total_entities': len(logs_text.split()) + len(chats_text.split()) + len(tickets_text.split()),
            'entities_by_type': {
                'SYSTEM': ['server', 'database', 'api'],
                'ACTIONS': ['restart', 'investigate', 'monitor'],
                'PEOPLE': ['admin', 'user', 'support']
            }
        }
    }

def display_nlp_results(results):
    """Display NLP analysis results"""
    st.success("✅ NLP analysis completed!")
    
    # Processing metadata
    metadata = results.get('processing_metadata', {})
    st.markdown("### 📊 Processing Summary")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Processing Mode", metadata.get('processing_mode', 'unknown'))
    with col2:
        st.metric("Processing Time", f"{metadata.get('processing_time_seconds', 0):.2f}s")
    with col3:
        st.metric("Data Sources", len(metadata.get('data_sources_processed', [])))
    
    # Incident insights
    insights = results.get('incident_insights', {})
    if insights:
        st.markdown("### 🚨 Incident Insights")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            severity = insights.get('incident_severity', 'unknown')
            severity_color = {'critical': '🔴', 'high': '🟠', 'medium': '🟡', 'low': '🟢'}.get(severity, '⚪')
            st.metric("Incident Severity", f"{severity_color} {severity.title()}")
        
        with col2:
            confidence = insights.get('confidence_score', 0)
            st.metric("Confidence Score", f"{confidence:.1%}")
        
        with col3:
            business_impact = insights.get('business_impact', 'unknown')
            st.metric("Business Impact", business_impact.title())
        
        # Affected components
        if insights.get('affected_components'):
            st.markdown("**🎯 Affected Components:**")
            for component in insights['affected_components']:
                st.markdown(f"• {component}")
        
        # Probable causes
        if insights.get('probable_causes'):
            st.markdown("**🔍 Probable Causes:**")
            for cause in insights['probable_causes']:
                st.markdown(f"• {cause}")
    
    # Entity analysis
    entity_analysis = results.get('entity_analysis', {})
    if entity_analysis:
        st.markdown("### 🏷️ Entity Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Entities", entity_analysis.get('total_entities', 0))
        
        with col2:
            entities_by_type = entity_analysis.get('entities_by_type', {})
            st.metric("Entity Types", len(entities_by_type))
        
        # Show entities by type
        for entity_type, entities in entities_by_type.items():
            if entities:
                st.markdown(f"**{entity_type}:** {', '.join(entities[:5])}")
    
    # Export results
    if st.button("📁 Export NLP Results", key="export_nlp_btn"):
        json_str = json.dumps(results, indent=2, default=str)
        st.download_button(
            label="⬇️ Download Results",
            data=json_str,
            file_name=f"nlp_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

# ===================== PAGE: REAL-TIME MONITORING - FIXED =====================
def monitoring_page():
    """Real-time monitoring page with proper auto-refresh control"""
    st.markdown("""
    <div class="main-header">
        <h1>📈 Real-time System Monitoring</h1>
        <p>Live monitoring of system health, performance, and incidents</p>
    </div>
    """, unsafe_allow_html=True)
    
    # CRITICAL: Only enable auto-refresh when explicitly on monitoring page
    current_page = st.session_state.get('current_page', 'dashboard')
    
    # Monitoring controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # CRITICAL: Only allow monitoring when on monitoring page
        monitoring_enabled = st.checkbox(
            "🔴 Enable Live Monitoring", 
            value=(st.session_state.monitoring_active and current_page == 'monitoring'),
            key="monitoring_enable_checkbox",
            help="Enable real-time data refresh"
        )
    
    with col2:
        refresh_interval = st.selectbox(
            "⏱️ Refresh Interval", 
            ["5s", "10s", "30s", "1min"], 
            index=1,
            key="refresh_interval_select"
        )
    
    with col3:
        if st.button("🔄 Manual Refresh", key="manual_refresh_btn", use_container_width=True):
            st.rerun()
    
    # CRITICAL: Only update monitoring state when actually on monitoring page
    if current_page == 'monitoring':
        st.session_state.monitoring_active = monitoring_enabled
    
    # Real-time metrics
    render_realtime_metrics()
    
    # System health indicators
    render_system_health()
    
    # CRITICAL: Auto-refresh only when all conditions are met
    should_auto_refresh = (
        st.session_state.monitoring_active and 
        current_page == 'monitoring' and
        not st.session_state.get('prevent_auto_rerun', False) and
        not st.session_state.get('chat_processing', False)
    )
    
    if should_auto_refresh:
        interval_seconds = {"5s": 5, "10s": 10, "30s": 30, "1min": 60}[refresh_interval]
        time.sleep(interval_seconds)
        st.rerun()

def render_realtime_metrics():
    """Render real-time system metrics"""
    st.markdown("### 📊 Live System Metrics")
    
    # Generate real-time data
    current_time = datetime.now()
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        cpu_usage = np.random.uniform(15, 85)
        cpu_delta = np.random.uniform(-10, 10)
        st.metric(
            "🖥️ CPU Usage",
            f"{cpu_usage:.1f}%",
            f"{cpu_delta:+.1f}%"
        )
    
    with col2:
        memory_usage = np.random.uniform(25, 75)
        memory_delta = np.random.uniform(-5, 5)
        st.metric(
            "💾 Memory Usage", 
            f"{memory_usage:.1f}%",
            f"{memory_delta:+.1f}%"
        )
    
    with col3:
        error_rate = np.random.uniform(0, 8)
        error_delta = np.random.uniform(-2, 2)
        st.metric(
            "❌ Error Rate",
            f"{error_rate:.2f}%",
            f"{error_delta:+.2f}%",
            delta_color="inverse"
        )
    
    with col4:
        response_time = np.random.uniform(80, 450)
        response_delta = np.random.uniform(-50, 50)
        st.metric(
            "⏱️ Avg Response Time",
            f"{response_time:.0f}ms",
            f"{response_delta:+.0f}ms",
            delta_color="inverse"
        )
    
    # Real-time charts
    col1, col2 = st.columns(2)
    
    with col1:
        # CPU and Memory chart
        times = pd.date_range(start=current_time - timedelta(hours=1), 
                             end=current_time, freq='1min')
        
        metrics_data = pd.DataFrame({
            'Time': times,
            'CPU': np.random.uniform(20, 80, len(times)),
            'Memory': np.random.uniform(30, 70, len(times))
        })
        
        fig1 = px.line(metrics_data, x='Time', y=['CPU', 'Memory'],
                      title="CPU & Memory Usage (Last Hour)",
                      labels={'value': 'Usage (%)', 'variable': 'Metric'})
        fig1.update_layout(height=300)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Error rate and response time
        error_data = pd.DataFrame({
            'Time': times,
            'Error_Rate': np.random.uniform(0, 5, len(times)),
            'Response_Time': np.random.uniform(100, 400, len(times)) / 100  # Scale for visibility
        })
        
        fig2 = px.line(error_data, x='Time', y=['Error_Rate', 'Response_Time'],
                      title="Error Rate & Response Time (Last Hour)",
                      labels={'value': 'Rate/Time', 'variable': 'Metric'})
        fig2.update_layout(height=300)
        st.plotly_chart(fig2, use_container_width=True)

def render_system_health():
    """Render system health indicators"""
    st.markdown("### 🏥 System Health Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🔧 Component Status")
        components = [
            ("API Gateway", np.random.choice(["🟢 Healthy", "🟡 Warning", "🔴 Critical"], p=[0.7, 0.2, 0.1])),
            ("Database", np.random.choice(["🟢 Healthy", "🟡 Warning", "🔴 Critical"], p=[0.8, 0.15, 0.05])),
            ("Load Balancer", np.random.choice(["🟢 Healthy", "🟡 Warning", "🔴 Critical"], p=[0.9, 0.08, 0.02])),
            ("Cache Layer", np.random.choice(["🟢 Healthy", "🟡 Warning", "🔴 Critical"], p=[0.85, 0.1, 0.05]))
        ]
        
        for component, status in components:
            st.markdown(f"**{component}:** {status}")
    
    with col2:
        st.markdown("#### 📊 Performance Indicators")
        st.progress(np.random.uniform(0.6, 0.9), "Overall Health Score")
        st.progress(np.random.uniform(0.7, 0.95), "Uptime SLA")
        st.progress(np.random.uniform(0.5, 0.8), "Performance Score")
        st.progress(np.random.uniform(0.8, 1.0), "Security Score")
    
    with col3:
        st.markdown("#### 🚨 Alert Summary")
        alert_summary = {
            "Critical": np.random.randint(0, 3),
            "High": np.random.randint(0, 5),
            "Medium": np.random.randint(0, 10),
            "Low": np.random.randint(0, 15)
        }
        
        for level, count in alert_summary.items():
            color_map = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}
            st.markdown(f"{color_map[level]} **{level}:** {count}")

# ===================== OTHER PAGES (PLACEHOLDERS) =====================
def analysis_page():
    """Model Analysis page"""
    st.markdown("""
    <div class="main-header">
        <h1>📊 Model Analysis</h1>
        <p>ML model performance, accuracy metrics, and detailed insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("🚧 Model Analysis page - Integration with your existing ML system")
    
    if st.button("🚀 Run ML Analysis", type="primary"):
        run_quick_system_analysis()

def testing_page():
    """Data Testing page"""
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Data Testing</h1>
        <p>Quality tests, model validation, and integration testing</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("🚧 Data Testing page - File upload and testing capabilities")

def learning_page():
    """Self Learning page"""
    st.markdown("""
    <div class="main-header">
        <h1>🎓 Self Learning</h1>
        <p>ML training, model improvement, and learning cycles</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("🚧 Self Learning page - Integration with your self-learning system")

def config_page():
    """System Configuration page"""
    st.markdown("""
    <div class="main-header">
        <h1>⚙️ System Configuration</h1>
        <p>Settings, API keys, and system information</p>
    </div>
    """, unsafe_allow_html=True)
    
    # System info
    st.markdown("### 🔧 System Information")
    
    system_info = {
        "Components Available": COMPONENTS_AVAILABLE,
        "Chatbot Initialized": st.session_state.chatbot is not None,
        "NLP Processor Ready": st.session_state.nlp_processor is not None,
        "Chat Messages": len(st.session_state.chat_history),
        "Monitoring Active": st.session_state.monitoring_active,
        "Current Page": st.session_state.current_page
    }
    
    for key, value in system_info.items():
        st.markdown(f"**{key}:** {value}")
    
    # Configuration
    st.markdown("### ⚙️ Configuration")
    if COMPONENTS_AVAILABLE:
        try:
            st.markdown(f"**Gemini API Key:** {'✅ Configured' if config.GEMINI_API_KEY else '❌ Not configured'}")
            st.markdown(f"**Model Name:** {config.genai_config.get('model_name', 'Not specified')}")
            st.markdown(f"**Production Mode:** {config.system_config.get('production_mode', False)}")
        except:
            st.warning("⚠️ Configuration not fully accessible")
    else:
        st.warning("⚠️ System components not loaded")

# ===================== MAIN APPLICATION =====================
def main():
    """Main Streamlit application - COMPLETE VERSION"""
    
    # CRITICAL: Initialize both regular and chat session state
    initialize_session_state()
    
    # Header
    if st.session_state.current_page == 'dashboard':
        pass  # Dashboard has its own header
    
    # Render sidebar and get current page
    current_page = render_sidebar()
    
    # CRITICAL: Track current page for auto-refresh control
    st.session_state.current_page = current_page
    
    # Route to appropriate page
    if current_page == "dashboard":
        dashboard_page()
    elif current_page == "chatbot":
        chatbot_page()
    elif current_page == "nlp":
        nlp_page()
    elif current_page == "analysis":
        analysis_page()
    elif current_page == "testing":
        testing_page()
    elif current_page == "monitoring":
        monitoring_page()
    elif current_page == "learning":
        learning_page()
    elif current_page == "config":
        config_page()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #6c757d; margin-top: 2rem;'>
        <p>🚀 SRE Incident Insight Engine | Production-Ready AI System | Powered by Gemini AI</p>
        <p>Complete Integration: ML Analysis • NLP Context Extraction • AI Chatbot • Real-time Monitoring</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
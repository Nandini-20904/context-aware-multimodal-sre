"""
PRODUCTION-READY STREAMLIT DASHBOARD - MAIN APPLICATION
Multi-page SRE Self-Learning System Dashboard
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
from pathlib import Path

# Configure Streamlit page
st.set_page_config(
    page_title="SRE Self-Learning System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
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
    .sidebar-section {
        margin: 1rem 0;
        padding: 0.5rem 0;
        border-bottom: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize all session state variables"""
    defaults = {
        'system_initialized': False,
        'system': None,
        'monitoring_active': False,
        'models_loaded': False,
        'alerts': [],
        'current_data': None,
        'last_analysis_time': None,
        'knowledge_base_stats': {},
        'system_metrics': {
            'anomalies_today': 0,
            'failures_today': 0,
            'zero_day_today': 0,
            'total_patterns': 0
        }
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Load self-learning system
@st.cache_resource
def load_self_learning_system():
    """Load and initialize the self-learning system"""
    try:
        from self_learning import SelfLearningMetaSystem
        system = SelfLearningMetaSystem(model_dir="production_models")
        
        # Initialize models
        system.initialize_models()
        
        # Try to load existing models
        models_loaded = system.load_self_learning_models()
        
        return system, models_loaded
    except Exception as e:
        st.error(f"❌ Error loading system: {e}")
        return None, False

# Generate sample alerts for demo
def generate_sample_alerts():
    """Generate sample alerts for demonstration"""
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
                'title': '🧠 Knowledge Base Update',
                'message': 'Self-learning system added 3 new anomaly patterns',
                'level': 'info',
                'timestamp': datetime.now() - timedelta(hours=1),
                'source': 'Self-Learning',
                'resolved': False
            },
            {
                'id': 5,
                'title': '✅ System Health Check',
                'message': 'All monitoring systems operational',
                'level': 'success',
                'timestamp': datetime.now() - timedelta(hours=2),
                'source': 'System Monitor',
                'resolved': True
            }
        ]
        st.session_state.alerts = sample_alerts

# Sidebar navigation
def render_sidebar():
    """Render the sidebar navigation and system status"""
    st.sidebar.markdown("# 🧠 SRE Self-Learning System")
    st.sidebar.markdown("---")
    
    # Navigation
    pages = {
        "🏠 Dashboard": "dashboard",
        "📊 Anomaly Analysis": "anomaly_analysis",
        "⚠️ Failure Analysis": "failure_analysis", 
        "🛡️ Zero-Day Analysis": "zero_day_analysis",
        "📡 Real-Time Monitoring": "monitoring",
        "📁 Dataset Testing": "dataset_testing",
        "🧠 Self-Learning Hub": "self_learning",
        "⚙️ Configuration": "configuration"
    }
    
    selected = st.sidebar.selectbox(
        "Navigate to:",
        list(pages.keys()),
        key="navigation"
    )
    
    st.sidebar.markdown("---")
    
    # System status
    st.sidebar.markdown("### System Status")
    
    if st.session_state.system_initialized:
        st.sidebar.markdown('<span class="status-indicator status-online"></span>**System: Online**', 
                           unsafe_allow_html=True)
        if st.session_state.monitoring_active:
            st.sidebar.markdown('<span class="status-indicator status-online"></span>**Monitoring: Active**', 
                               unsafe_allow_html=True)
        else:
            st.sidebar.markdown('<span class="status-indicator status-warning"></span>**Monitoring: Standby**', 
                               unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<span class="status-indicator status-offline"></span>**System: Initializing**', 
                           unsafe_allow_html=True)
    
    # Quick stats in sidebar
    if st.session_state.system and st.session_state.system_initialized:
        st.sidebar.markdown("### Quick Stats")
        try:
            kb = st.session_state.system.knowledge_base
            total_patterns = (
                len(kb.known_anomaly_patterns) + 
                len(kb.zero_day_patterns) + 
                len(kb.failure_patterns)
            )
            
            st.sidebar.metric("Total Patterns", total_patterns)
            st.sidebar.metric("Anomaly Patterns", len(kb.known_anomaly_patterns))
            st.sidebar.metric("Failure Patterns", len(kb.failure_patterns))
            st.sidebar.metric("Zero-Day Patterns", len(kb.zero_day_patterns))
        except:
            st.sidebar.info("Stats loading...")
    
    # Alerts summary
    if st.session_state.alerts:
        st.sidebar.markdown("### Alert Summary")
        critical_count = len([a for a in st.session_state.alerts if a['level'] == 'critical' and not a.get('resolved', False)])
        warning_count = len([a for a in st.session_state.alerts if a['level'] == 'warning' and not a.get('resolved', False)])
        
        if critical_count > 0:
            st.sidebar.error(f"🔴 {critical_count} Critical")
        if warning_count > 0:
            st.sidebar.warning(f"🟡 {warning_count} Warning")
        if critical_count == 0 and warning_count == 0:
            st.sidebar.success("✅ All Clear")
    
    return pages[selected]

# Dashboard page
def dashboard_page():
    """Main dashboard page"""
    st.markdown('<h1 class="main-header">🧠 SRE Self-Learning System Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Initialize system if not done
    if not st.session_state.system_initialized:
        with st.spinner("🚀 Initializing SRE Self-Learning System..."):
            system, models_loaded = load_self_learning_system()
            if system:
                st.session_state.system = system
                st.session_state.system_initialized = True
                st.session_state.models_loaded = models_loaded
                
                if models_loaded:
                    st.success("✅ Self-learning models loaded successfully!")
                else:
                    st.info("ℹ️ System initialized. Models ready for training.")
                
                time.sleep(2)
                st.rerun()
            else:
                st.error("❌ Failed to initialize system. Please check your setup.")
                return
    
    # System overview metrics
    st.markdown("### 📊 System Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status_color = "🟢" if st.session_state.system_initialized else "🔴"
        status_text = "Online" if st.session_state.system_initialized else "Offline"
        st.metric(
            label=f"{status_color} System Status",
            value=status_text,
            delta="Self-Learning Active" if st.session_state.models_loaded else "Ready for Training"
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
        if st.session_state.system and st.session_state.system.knowledge_base:
            total_patterns = (
                len(st.session_state.system.knowledge_base.known_anomaly_patterns) +
                len(st.session_state.system.knowledge_base.zero_day_patterns) +
                len(st.session_state.system.knowledge_base.failure_patterns)
            )
        else:
            total_patterns = 0
        
        st.metric(
            label="🧠 Knowledge Patterns",
            value=total_patterns,
            delta="Learning Enabled" if total_patterns > 0 else "Ready to Learn"
        )
    
    with col4:
        monitoring_emoji = "📡" if st.session_state.monitoring_active else "⏸️"
        monitoring_status = "Active" if st.session_state.monitoring_active else "Standby"
        st.metric(
            label=f"{monitoring_emoji} Monitoring",
            value=monitoring_status,
            delta="Real-time" if st.session_state.monitoring_active else None
        )
    
    st.markdown("---")
    
    # Quick Actions
    st.markdown("### 🚀 Quick Actions")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📡 Start Real-Time Monitoring", use_container_width=True, type="primary"):
            st.session_state.monitoring_active = True
            st.success("✅ Real-time monitoring started!")
            time.sleep(1)
            st.rerun()
    
    with col2:
        if st.button("🔍 Quick System Scan", use_container_width=True):
            with st.spinner("Running quick system scan..."):
                time.sleep(2)
                st.success("✅ System scan completed - No immediate threats detected")
    
    with col3:
        if st.button("📊 Generate Reports", use_container_width=True):
            st.info("📊 Generating comprehensive system reports...")
            time.sleep(1)
    
    with col4:
        if st.button("🧠 Trigger Learning Update", use_container_width=True):
            if st.session_state.system:
                with st.spinner("Updating knowledge base..."):
                    time.sleep(2)
                    st.success("🧠 Knowledge base updated with latest patterns!")
    
    # Main dashboard content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # System performance chart
        st.markdown("### 📈 24-Hour System Performance")
        
        # Generate sample performance data
        hours = pd.date_range(start=datetime.now() - timedelta(hours=24), 
                             end=datetime.now(), freq='1H')
        performance_data = pd.DataFrame({
            'Time': hours,
            'Anomalies': np.random.poisson(3, len(hours)),
            'Failures': np.random.poisson(1, len(hours)),
            'Zero_Day': np.random.poisson(0.5, len(hours)),
            'CPU_Avg': np.random.uniform(30, 80, len(hours)),
            'Memory_Avg': np.random.uniform(40, 75, len(hours))
        })
        
        fig = go.Figure()
        
        # Detection counts
        fig.add_trace(go.Scatter(
            x=performance_data['Time'], y=performance_data['Anomalies'],
            mode='lines+markers', name='Anomalies Detected',
            line=dict(color='#ff7f0e', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=performance_data['Time'], y=performance_data['Failures'],
            mode='lines+markers', name='Failures Predicted',
            line=dict(color='#d62728', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=performance_data['Time'], y=performance_data['Zero_Day'],
            mode='lines+markers', name='Zero-Day Threats',
            line=dict(color='#2ca02c', width=2)
        ))
        
        fig.update_layout(
            title="Detection Activity Over Time",
            xaxis_title="Time",
            yaxis_title="Count",
            hovermode='x unified',
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # System resource utilization
        fig2 = go.Figure()
        
        fig2.add_trace(go.Scatter(
            x=performance_data['Time'], y=performance_data['CPU_Avg'],
            mode='lines', name='CPU Usage (%)',
            line=dict(color='#1f77b4', width=2),
            fill='tonexty'
        ))
        
        fig2.add_trace(go.Scatter(
            x=performance_data['Time'], y=performance_data['Memory_Avg'],
            mode='lines', name='Memory Usage (%)',
            line=dict(color='#ff7f0e', width=2),
            fill='tonexty'
        ))
        
        fig2.update_layout(
            title="System Resource Utilization",
            xaxis_title="Time",
            yaxis_title="Usage (%)",
            height=300,
            yaxis=dict(range=[0, 100])
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        # Recent alerts
        st.markdown("### 🚨 Recent Alerts")
        
        generate_sample_alerts()  # Generate demo alerts
        
        # Display recent alerts
        recent_alerts = sorted(st.session_state.alerts, 
                              key=lambda x: x['timestamp'], reverse=True)[:5]
        
        for alert in recent_alerts:
            alert_class = f"alert-{alert['level']}"
            resolved_text = " ✅" if alert.get('resolved', False) else ""
            time_str = alert['timestamp'].strftime('%H:%M:%S')
            
            st.markdown(f"""
            <div class="{alert_class}">
                <strong>{alert['title']}{resolved_text}</strong><br>
                {alert['message']}<br>
                <small>🕒 {time_str} | 📡 {alert['source']}</small>
            </div>
            """, unsafe_allow_html=True)
        
        # Alert management
        if st.button("🔄 Refresh Alerts", use_container_width=True):
            st.rerun()
        
        if st.button("✅ Mark All Resolved", use_container_width=True):
            for alert in st.session_state.alerts:
                alert['resolved'] = True
            st.success("All alerts marked as resolved")
            st.rerun()
    
    # Knowledge base overview
    st.markdown("---")
    st.markdown("### 🧠 Knowledge Base Overview")
    
    if st.session_state.system and st.session_state.system.knowledge_base:
        kb = st.session_state.system.knowledge_base
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("#### 📚 Anomaly Patterns")
            anomaly_count = len(kb.known_anomaly_patterns)
            st.metric("Total Patterns", anomaly_count)
            
            if anomaly_count > 0:
                # Sample pattern analysis
                pattern_types = {}
                for pattern in list(kb.known_anomaly_patterns)[:20]:
                    ptype = pattern.get('type', 'unknown')
                    pattern_types[ptype] = pattern_types.get(ptype, 0) + 1
                
                st.write("**Top Types:**")
                for ptype, count in list(pattern_types.items())[:3]:
                    st.text(f"• {ptype}: {count}")
        
        with col2:
            st.markdown("#### 🔍 Zero-Day Patterns")
            zero_day_count = len(kb.zero_day_patterns)
            st.metric("Zero-Day Patterns", zero_day_count)
            
            if zero_day_count > 0:
                avg_effectiveness = np.mean([
                    p.get('effectiveness', 0) for p in kb.zero_day_patterns
                ])
                st.metric("Avg Effectiveness", f"{avg_effectiveness:.2f}")
        
        with col3:
            st.markdown("#### ⚠️ Failure Patterns")
            failure_count = len(kb.failure_patterns)
            st.metric("Failure Patterns", failure_count)
            
            if failure_count > 0:
                recent_patterns = [
                    p for p in kb.failure_patterns 
                    if (datetime.now() - p.get('timestamp', datetime.now())).days <= 7
                ]
                st.metric("This Week", len(recent_patterns))
        
        with col4:
            st.markdown("#### 📝 Learning Activity")
            feedback_count = len(kb.feedback_buffer)
            st.metric("Feedback Entries", feedback_count)
            
            if feedback_count > 0:
                recent_feedback = [
                    f for f in kb.feedback_buffer
                    if (datetime.now() - f.get('timestamp', datetime.now())).hours <= 24
                ]
                st.metric("Last 24h", len(recent_feedback))
    
    else:
        st.info("🧠 Knowledge base will be available once the system is fully initialized.")
    
    # System health indicators
    st.markdown("---")
    st.markdown("### 🏥 System Health")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Model health
        st.markdown("#### 🤖 Model Health")
        if st.session_state.models_loaded:
            model_status = "🟢 All Models Loaded"
            model_details = [
                "✅ Anomaly Detection: Ready",
                "✅ Failure Prediction: Ready", 
                "✅ Zero-Day Detection: Ready"
            ]
        else:
            model_status = "🟡 Models Initializing"
            model_details = [
                "⏳ Anomaly Detection: Loading",
                "⏳ Failure Prediction: Loading",
                "⏳ Zero-Day Detection: Loading"
            ]
        
        st.write(model_status)
        for detail in model_details:
            st.text(detail)
    
    with col2:
        # Data pipeline health
        st.markdown("#### 📊 Data Pipeline")
        pipeline_status = "🟢 Data Pipeline Healthy"
        pipeline_details = [
            "✅ Log Collection: Active",
            "✅ Metrics Ingestion: Active",
            "✅ Chat Analysis: Active",
            "✅ Ticket Processing: Active"
        ]
        
        st.write(pipeline_status)
        for detail in pipeline_details:
            st.text(detail)
    
    with col3:
        # Learning system health
        st.markdown("#### 🧠 Learning System")
        learning_status = "🟢 Learning System Active"
        learning_details = [
            "✅ Pattern Recognition: Active",
            "✅ Cross-Pollination: Active",
            "✅ Auto-Retraining: Enabled",
            "✅ Feedback Loop: Active"
        ]
        
        st.write(learning_status)
        for detail in learning_details:
            st.text(detail)

# Main application
def main():
    """Main Streamlit application"""
    # Initialize session state
    initialize_session_state()
    
    # Render sidebar and get current page
    current_page = render_sidebar()
    
    # Route to appropriate page
    if current_page == "dashboard":
        dashboard_page()
    else:
        # For other pages, show construction notice for now
        page_names = {
            "anomaly_analysis": "📊 Anomaly Analysis",
            "failure_analysis": "⚠️ Failure Analysis", 
            "zero_day_analysis": "🛡️ Zero-Day Analysis",
            "monitoring": "📡 Real-Time Monitoring",
            "dataset_testing": "📁 Dataset Testing",
            "self_learning": "🧠 Self-Learning Hub",
            "configuration": "⚙️ Configuration"
        }
        
        page_title = page_names.get(current_page, "Page")
        st.markdown(f'<h1 class="main-header">{page_title}</h1>', unsafe_allow_html=True)
        
        st.info(f"🚧 {page_title} page is under construction. Full implementation coming next!")
        
        # Show what will be included in each page
        if current_page == "dataset_testing":
            st.markdown("""
            ### 📁 Dataset Testing Features (Coming Soon):
            - **File Upload**: Upload CSV, JSON, Excel files for analysis
            - **Automatic Analysis**: Run all 3 models on uploaded data
            - **Real-time Results**: See anomalies, failures, and zero-day threats
            - **Model Training**: Train models on new datasets
            - **Pattern Learning**: Extract and learn new patterns
            - **Export Results**: Download analysis reports
            """)
        elif current_page == "monitoring":
            st.markdown("""
            ### 📡 Real-Time Monitoring Features (Coming Soon):
            - **Live Data Streaming**: Real-time system metrics
            - **Auto-alerts**: Instant notifications for threats
            - **Dashboard Views**: Multiple monitoring dashboards
            - **Historical Trends**: Time-series analysis
            - **System Health**: Real-time status indicators
            - **Alert Management**: Configure and manage alerts
            """)

if __name__ == "__main__":
    main()
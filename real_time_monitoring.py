"""
FIXED REAL-TIME MONITORING PAGE - Error handling for datetime operations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import json

def real_time_monitoring_page():
    """Complete Real-Time Monitoring Page - FIXED"""
    st.markdown('<h1 class="main-header">📡 Real-Time System Monitoring</h1>', unsafe_allow_html=True)
    
    if not st.session_state.get('system_initialized', False):
        st.error("❌ System not initialized. Please go to Dashboard first.")
        return
    
    system = st.session_state.system
    
    # Monitoring controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📡 Monitoring Settings")
        auto_refresh = st.checkbox("Auto Refresh", value=st.session_state.get('monitoring_active', False))
        refresh_interval = st.selectbox("Refresh Interval", ["5s", "10s", "30s", "1min"], index=2)
        alert_threshold = st.slider("Alert Threshold", 0.1, 1.0, 0.7, 0.1)
        
        if auto_refresh != st.session_state.get('monitoring_active', False):
            st.session_state.monitoring_active = auto_refresh
    
    with col2:
        st.markdown("### 🎯 Monitoring Scope")
        monitor_anomalies = st.checkbox("Anomaly Detection", value=True)
        monitor_failures = st.checkbox("Failure Prediction", value=True) 
        monitor_security = st.checkbox("Security Threats", value=True)
        monitor_performance = st.checkbox("Performance Metrics", value=True)
    
    with col3:
        st.markdown("### 📊 System Status")
        if st.session_state.monitoring_active:
            st.success("🟢 Monitoring Active")
            
            # FIXED: Handle None monitoring_start properly
            monitoring_start = st.session_state.get('monitoring_start')
            if monitoring_start is None:
                # Set monitoring start time if it's None
                st.session_state.monitoring_start = datetime.now()
                monitoring_start = st.session_state.monitoring_start
            
            uptime = datetime.now() - monitoring_start
            uptime_hours = uptime.seconds // 3600
            uptime_minutes = (uptime.seconds // 60) % 60
            st.metric("Uptime", f"{uptime_hours}h {uptime_minutes}m")
        else:
            st.warning("🟡 Monitoring Inactive")
            if st.button("▶️ Start Monitoring", type="primary"):
                st.session_state.monitoring_active = True
                st.session_state.monitoring_start = datetime.now()
                st.rerun()
        
        # Live alerts count
        active_alerts = len([a for a in st.session_state.get('alerts', []) if not a.get('resolved', False)])
        st.metric("Active Alerts", active_alerts)
    
    st.markdown("---")
    
    # Main monitoring dashboard
    if st.session_state.monitoring_active:
        show_live_monitoring_dashboard(system, monitor_anomalies, monitor_failures, 
                                     monitor_security, monitor_performance, alert_threshold)
        
        # Auto-refresh logic
        if auto_refresh:
            refresh_seconds = {"5s": 5, "10s": 10, "30s": 30, "1min": 60}
            time.sleep(refresh_seconds.get(refresh_interval, 30))
            st.rerun()
    else:
        show_monitoring_setup()

def show_live_monitoring_dashboard(system, monitor_anomalies, monitor_failures, monitor_security, monitor_performance, alert_threshold):
    """Show live monitoring dashboard"""
    
    # Generate real-time data
    current_time = datetime.now()
    live_data = generate_live_monitoring_data(current_time)
    
    # Process data through monitoring systems
    monitoring_results = process_live_monitoring(system, live_data, monitor_anomalies, 
                                               monitor_failures, monitor_security, alert_threshold)
    
    # Display live metrics
    show_live_metrics_overview(live_data, monitoring_results)
    
    # Live charts
    show_live_monitoring_charts(live_data, monitoring_results, monitor_performance)
    
    # Process and display alerts
    process_live_alerts(monitoring_results, alert_threshold)
    
    # Activity log
    show_monitoring_activity_log()

def generate_live_monitoring_data(current_time):
    """Generate realistic live monitoring data"""
    
    # Base system metrics with realistic patterns
    time_of_day = current_time.hour + current_time.minute / 60.0
    
    # CPU usage with daily patterns (higher during business hours)
    base_cpu = 35 + 15 * np.sin((time_of_day - 6) * np.pi / 12)
    cpu_util = max(10, min(95, base_cpu + np.random.normal(0, 8)))
    
    # Memory usage with gradual increase pattern
    base_memory = 45 + 10 * np.sin((time_of_day - 8) * np.pi / 14)
    memory_util = max(20, min(90, base_memory + np.random.normal(0, 5)))
    
    # Network traffic
    network_in = np.random.uniform(200, 1200)
    network_out = np.random.uniform(100, 800)
    
    # Error rates (usually low, occasional spikes)
    error_rate = np.random.exponential(0.01) if np.random.random() > 0.05 else np.random.uniform(0.05, 0.15)
    
    # Response time
    response_time = np.random.exponential(0.15) + (cpu_util / 100) * 0.3
    
    # Active connections
    active_connections = np.random.poisson(45 + int(cpu_util / 4))
    
    # Security metrics
    failed_logins = np.random.poisson(1) if np.random.random() > 0.1 else np.random.poisson(8)
    firewall_blocks = np.random.poisson(2)
    
    # Disk and queue metrics
    disk_util = np.random.uniform(25, 75)
    queue_depth = np.random.poisson(3 + int(cpu_util / 20))
    
    live_data = {
        'timestamp': current_time,
        'cpu_util': cpu_util,
        'memory_util': memory_util,
        'network_in': network_in,
        'network_out': network_out,
        'error_rate': error_rate,
        'response_time': response_time,
        'active_connections': active_connections,
        'failed_logins': failed_logins,
        'firewall_blocks': firewall_blocks,
        'disk_util': disk_util,
        'queue_depth': queue_depth,
        
        # Derived metrics
        'resource_pressure': (cpu_util + memory_util) / 2,
        'network_ratio': network_out / max(network_in, 1),
        'system_health': calculate_system_health(cpu_util, memory_util, error_rate, response_time)
    }
    
    return live_data

def calculate_system_health(cpu_util, memory_util, error_rate, response_time):
    """Calculate overall system health score"""
    
    # Health scoring (0-100 scale)
    cpu_health = max(0, 100 - (cpu_util - 60)) if cpu_util > 60 else 100
    memory_health = max(0, 100 - (memory_util - 70)) if memory_util > 70 else 100
    error_health = max(0, 100 - (error_rate * 1000)) if error_rate > 0.01 else 100
    response_health = max(0, 100 - (response_time * 100)) if response_time > 0.5 else 100
    
    # Weighted average
    overall_health = (cpu_health * 0.3 + memory_health * 0.3 + 
                     error_health * 0.2 + response_health * 0.2)
    
    return min(100, max(0, overall_health))

def process_live_monitoring(system, live_data, monitor_anomalies, monitor_failures, monitor_security, alert_threshold):
    """Process live data through monitoring systems"""
    
    results = {
        'timestamp': live_data['timestamp'],
        'anomaly_detection': None,
        'failure_prediction': None,
        'security_analysis': None,
        'alerts': []
    }
    
    # Convert single data point to DataFrame for processing
    df = pd.DataFrame([live_data])
    
    try:
        # Anomaly detection
        if monitor_anomalies:
            anomaly_score = detect_live_anomaly(live_data)
            results['anomaly_detection'] = {
                'score': anomaly_score,
                'is_anomaly': anomaly_score > 0.5,
                'severity': get_anomaly_severity(anomaly_score)
            }
            
            if anomaly_score > alert_threshold:
                results['alerts'].append({
                    'type': 'anomaly',
                    'severity': get_anomaly_severity(anomaly_score),
                    'message': f"Anomaly detected (score: {anomaly_score:.3f})",
                    'timestamp': live_data['timestamp']
                })
        
        # Failure prediction
        if monitor_failures:
            failure_prob = predict_live_failure(live_data)
            results['failure_prediction'] = {
                'probability': failure_prob,
                'risk_level': get_risk_level(failure_prob),
                'time_to_failure': estimate_time_to_failure(failure_prob)
            }
            
            if failure_prob > alert_threshold:
                results['alerts'].append({
                    'type': 'failure',
                    'severity': get_risk_level(failure_prob),
                    'message': f"High failure probability ({failure_prob:.1%})",
                    'timestamp': live_data['timestamp']
                })
        
        # Security analysis
        if monitor_security:
            threat_score = analyze_live_security(live_data)
            results['security_analysis'] = {
                'threat_score': threat_score,
                'threat_level': get_threat_level_name(threat_score),
                'security_events': live_data['failed_logins'] + live_data['firewall_blocks']
            }
            
            if threat_score > alert_threshold:
                results['alerts'].append({
                    'type': 'security',
                    'severity': get_threat_level_name(threat_score),
                    'message': f"Security threat detected (score: {threat_score:.3f})",
                    'timestamp': live_data['timestamp']
                })
        
    except Exception as e:
        st.error(f"Error in live monitoring: {e}")
    
    return results

def detect_live_anomaly(data):
    """Detect anomalies in live data point"""
    
    anomaly_factors = []
    
    # CPU anomaly
    if data['cpu_util'] > 85:
        anomaly_factors.append((data['cpu_util'] - 85) / 15)
    elif data['cpu_util'] < 15:
        anomaly_factors.append((15 - data['cpu_util']) / 15)
    
    # Memory anomaly
    if data['memory_util'] > 85:
        anomaly_factors.append((data['memory_util'] - 85) / 15)
    
    # Error rate anomaly
    if data['error_rate'] > 0.05:
        anomaly_factors.append(min(1.0, data['error_rate'] * 10))
    
    # Response time anomaly
    if data['response_time'] > 1.0:
        anomaly_factors.append(min(1.0, (data['response_time'] - 1.0) / 2.0))
    
    # Network anomaly
    if data['network_ratio'] > 2.0:
        anomaly_factors.append(min(1.0, (data['network_ratio'] - 2.0) / 3.0))
    
    # Overall anomaly score
    if anomaly_factors:
        return min(1.0, max(anomaly_factors) + np.random.normal(0, 0.05))
    else:
        return np.random.uniform(0, 0.3)  # Normal baseline

def predict_live_failure(data):
    """Predict failure probability from live data"""
    
    failure_factors = []
    
    # Resource-based failure risk
    resource_risk = data['resource_pressure'] / 100
    failure_factors.append(resource_risk * 0.4)
    
    # Error-based failure risk
    error_risk = min(1.0, data['error_rate'] * 20)
    failure_factors.append(error_risk * 0.3)
    
    # Performance degradation risk
    performance_risk = min(1.0, data['response_time'] / 2.0)
    failure_factors.append(performance_risk * 0.2)
    
    # Queue buildup risk
    queue_risk = min(1.0, data['queue_depth'] / 20)
    failure_factors.append(queue_risk * 0.1)
    
    # Combined failure probability
    base_prob = sum(failure_factors)
    return min(1.0, max(0, base_prob + np.random.normal(0, 0.05)))

def analyze_live_security(data):
    """Analyze security threats in live data"""
    
    security_factors = []
    
    # Authentication threats
    if data['failed_logins'] > 5:
        security_factors.append(min(1.0, data['failed_logins'] / 20))
    
    # Network threats
    if data['network_out'] > 2000:
        security_factors.append(min(1.0, (data['network_out'] - 2000) / 3000))
    
    # Firewall activity
    if data['firewall_blocks'] > 5:
        security_factors.append(min(1.0, data['firewall_blocks'] / 15))
    
    # Resource abuse
    if data['cpu_util'] > 95 and data['memory_util'] > 90:
        security_factors.append(0.7)
    
    # Combined threat score
    if security_factors:
        return min(1.0, max(security_factors) + np.random.normal(0, 0.03))
    else:
        return np.random.uniform(0, 0.2)  # Baseline security noise

def get_anomaly_severity(score):
    """Get anomaly severity level"""
    if score > 0.8:
        return "Critical"
    elif score > 0.6:
        return "High"
    elif score > 0.4:
        return "Medium"
    else:
        return "Low"

def get_risk_level(prob):
    """Get risk level from probability"""
    if prob > 0.8:
        return "Critical"
    elif prob > 0.6:
        return "High"
    elif prob > 0.4:
        return "Medium"
    else:
        return "Low"

def get_threat_level_name(score):
    """Get threat level name"""
    if score > 0.8:
        return "Critical"
    elif score > 0.6:
        return "High"
    elif score > 0.4:
        return "Medium"
    else:
        return "Low"

def estimate_time_to_failure(prob):
    """Estimate time to failure based on probability"""
    if prob > 0.9:
        return "< 1 hour"
    elif prob > 0.8:
        return "1-6 hours"
    elif prob > 0.6:
        return "6-24 hours"
    elif prob > 0.4:
        return "1-7 days"
    else:
        return "> 1 week"

def show_live_metrics_overview(live_data, monitoring_results):
    """Show live metrics overview"""
    
    st.markdown("### 📊 Live System Metrics")
    
    # Top row - core system metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        cpu_color = "inverse" if live_data['cpu_util'] > 85 else "normal"
        st.metric("CPU Usage", f"{live_data['cpu_util']:.1f}%", 
                 delta="High" if live_data['cpu_util'] > 85 else "Normal",
                 delta_color=cpu_color)
    
    with col2:
        memory_color = "inverse" if live_data['memory_util'] > 85 else "normal"
        st.metric("Memory Usage", f"{live_data['memory_util']:.1f}%",
                 delta="High" if live_data['memory_util'] > 85 else "Normal",
                 delta_color=memory_color)
    
    with col3:
        error_color = "inverse" if live_data['error_rate'] > 0.05 else "normal"
        st.metric("Error Rate", f"{live_data['error_rate']:.3f}",
                 delta="Elevated" if live_data['error_rate'] > 0.05 else "Normal",
                 delta_color=error_color)
    
    with col4:
        response_color = "inverse" if live_data['response_time'] > 1.0 else "normal"
        st.metric("Response Time", f"{live_data['response_time']:.2f}s",
                 delta="Slow" if live_data['response_time'] > 1.0 else "Good",
                 delta_color=response_color)
    
    with col5:
        health = live_data['system_health']
        health_color = "normal" if health > 80 else "inverse" if health < 60 else "off"
        health_status = "Good" if health > 80 else "Poor" if health < 60 else "Fair"
        st.metric("System Health", f"{health:.1f}%", delta=health_status, delta_color=health_color)
    
    # Second row - monitoring results
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if monitoring_results['anomaly_detection']:
            anom = monitoring_results['anomaly_detection']
            severity_color = "inverse" if anom['severity'] in ["Critical", "High"] else "normal"
            st.metric("Anomaly Score", f"{anom['score']:.3f}",
                     delta=anom['severity'], delta_color=severity_color)
        else:
            st.metric("Anomaly Score", "N/A")
    
    with col2:
        if monitoring_results['failure_prediction']:
            fail = monitoring_results['failure_prediction']
            risk_color = "inverse" if fail['risk_level'] in ["Critical", "High"] else "normal"
            st.metric("Failure Risk", f"{fail['probability']:.1%}",
                     delta=fail['risk_level'], delta_color=risk_color)
        else:
            st.metric("Failure Risk", "N/A")
    
    with col3:
        if monitoring_results['security_analysis']:
            sec = monitoring_results['security_analysis']
            threat_color = "inverse" if sec['threat_level'] in ["Critical", "High"] else "normal"
            st.metric("Security Threats", f"{sec['threat_score']:.3f}",
                     delta=sec['threat_level'], delta_color=threat_color)
        else:
            st.metric("Security Threats", "N/A")
    
    with col4:
        alert_count = len(monitoring_results['alerts'])
        alert_color = "inverse" if alert_count > 0 else "normal"
        st.metric("Active Alerts", alert_count,
                 delta="New" if alert_count > 0 else "None", delta_color=alert_color)

def show_live_monitoring_charts(live_data, monitoring_results, monitor_performance):
    """Show live monitoring charts"""
    
    st.markdown("---")
    st.markdown("### 📈 Live Monitoring Charts")
    
    # Get historical data for charts
    if 'monitoring_history' not in st.session_state:
        st.session_state.monitoring_history = []
    
    # Add current data to history
    st.session_state.monitoring_history.append(live_data)
    
    # Keep only last 50 data points (for performance)
    if len(st.session_state.monitoring_history) > 50:
        st.session_state.monitoring_history = st.session_state.monitoring_history[-50:]
    
    history_df = pd.DataFrame(st.session_state.monitoring_history)
    
    if len(history_df) > 1:
        col1, col2 = st.columns(2)
        
        with col1:
            # System resource utilization
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=history_df['timestamp'], y=history_df['cpu_util'],
                                   mode='lines+markers', name='CPU %', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=history_df['timestamp'], y=history_df['memory_util'],
                                   mode='lines+markers', name='Memory %', line=dict(color='green')))
            fig.add_hline(y=85, line_dash="dash", line_color="red", annotation_text="High Threshold")
            fig.update_layout(title="System Resource Utilization", xaxis_title="Time", 
                            yaxis_title="Percentage", height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Error rate and response time
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig.add_trace(
                go.Scatter(x=history_df['timestamp'], y=history_df['error_rate'],
                          mode='lines+markers', name='Error Rate', line=dict(color='red')),
                secondary_y=False
            )
            
            fig.add_trace(
                go.Scatter(x=history_df['timestamp'], y=history_df['response_time'],
                          mode='lines+markers', name='Response Time', line=dict(color='orange')),
                secondary_y=True
            )
            
            fig.update_xaxes(title_text="Time")
            fig.update_yaxes(title_text="Error Rate", secondary_y=False)
            fig.update_yaxes(title_text="Response Time (s)", secondary_y=True)
            fig.update_layout(title="Performance Metrics", height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Network activity
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=history_df['timestamp'], y=history_df['network_in'],
                                   mode='lines+markers', name='Network In', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=history_df['timestamp'], y=history_df['network_out'],
                                   mode='lines+markers', name='Network Out', line=dict(color='red')))
            fig.update_layout(title="Network Activity", xaxis_title="Time", 
                            yaxis_title="MB/s", height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # System health trend
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=history_df['timestamp'], y=history_df['system_health'],
                                   mode='lines+markers', name='System Health', 
                                   line=dict(color='green'), fill='tonexty'))
            fig.add_hline(y=80, line_dash="dash", line_color="green", annotation_text="Good Threshold")
            fig.add_hline(y=60, line_dash="dash", line_color="orange", annotation_text="Warning Threshold")
            fig.update_layout(title="System Health Trend", xaxis_title="Time", 
                            yaxis_title="Health Score", height=350, yaxis=dict(range=[0, 100]))
            st.plotly_chart(fig, use_container_width=True)

def process_live_alerts(monitoring_results, alert_threshold):
    """Process and display live alerts"""
    
    new_alerts = monitoring_results.get('alerts', [])
    
    if new_alerts:
        # Add to session state alerts
        if 'alerts' not in st.session_state:
            st.session_state.alerts = []
        
        for alert in new_alerts:
            # Add unique ID and additional metadata
            alert['id'] = len(st.session_state.alerts) + 1
            alert['source'] = 'Real-time Monitoring'
            alert['resolved'] = False
            
            st.session_state.alerts.append(alert)
        
        # Show new alerts
        st.markdown("---")
        st.markdown("### 🚨 New Alerts")
        
        for alert in new_alerts:
            severity_colors = {
                'Critical': 'alert-critical',
                'High': 'alert-warning', 
                'Medium': 'alert-info',
                'Low': 'alert-info'
            }
            
            alert_class = severity_colors.get(alert.get('severity', 'Low'), 'alert-info')
            
            st.markdown(f"""
            <div class="{alert_class}">
                <strong>🚨 {alert['severity']} Alert</strong><br>
                {alert['message']}<br>
                <small>🕒 {alert['timestamp'].strftime('%H:%M:%S')} | 📡 {alert['source']}</small>
            </div>
            """, unsafe_allow_html=True)

def show_monitoring_activity_log():
    """Show monitoring activity log"""
    
    st.markdown("---")
    st.markdown("### 📜 Activity Log")
    
    # Get recent monitoring activities
    activities = get_monitoring_activities()
    
    if activities:
        for activity in activities[-10:]:  # Show last 10 activities
            timestamp = activity['timestamp'].strftime('%H:%M:%S')
            
            if activity['type'] == 'alert':
                icon = "🚨"
                color = "red"
            elif activity['type'] == 'detection':
                icon = "🔍"
                color = "orange"
            elif activity['type'] == 'normal':
                icon = "✅"
                color = "green"
            else:
                icon = "ℹ️"
                color = "blue"
            
            st.markdown(f"<span style='color:{color}'>{icon} {timestamp}: {activity['message']}</span>", 
                       unsafe_allow_html=True)
    else:
        st.info("No recent activities")

def get_monitoring_activities():
    """Get monitoring activities (simulated)"""
    
    if 'monitoring_activities' not in st.session_state:
        st.session_state.monitoring_activities = []
    
    # Add some sample activities
    current_time = datetime.now()
    
    # Add periodic activities
    if len(st.session_state.monitoring_activities) < 20:
        sample_activities = [
            {'type': 'normal', 'message': 'System health check completed - all systems normal', 'timestamp': current_time - timedelta(minutes=5)},
            {'type': 'detection', 'message': 'Anomaly detection scan completed - 0 anomalies found', 'timestamp': current_time - timedelta(minutes=10)},
            {'type': 'normal', 'message': 'Performance metrics updated', 'timestamp': current_time - timedelta(minutes=15)},
            {'type': 'info', 'message': 'Real-time monitoring started', 'timestamp': current_time - timedelta(minutes=20)}
        ]
        st.session_state.monitoring_activities.extend(sample_activities)
    
    return st.session_state.monitoring_activities

def show_monitoring_setup():
    """Show monitoring setup page when not active"""
    
    st.markdown("### 📡 Real-Time Monitoring Setup")
    
    st.info("Real-time monitoring is currently inactive. Configure and start monitoring to view live system metrics.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Monitoring Features")
        features = [
            "📊 **Live System Metrics** - CPU, Memory, Network, Disk",
            "🔍 **Real-time Anomaly Detection** - Instant anomaly alerts",
            "⚠️ **Failure Prediction** - Proactive failure warnings", 
            "🛡️ **Security Monitoring** - Live threat detection",
            "📈 **Performance Tracking** - Response time and throughput",
            "🚨 **Smart Alerting** - Configurable alert thresholds",
            "📜 **Activity Logging** - Complete monitoring history",
            "🔄 **Auto-refresh** - Customizable refresh intervals"
        ]
        
        for feature in features:
            st.markdown(feature)
    
    with col2:
        st.markdown("#### ⚙️ Configuration Options")
        
        st.markdown("**Refresh Intervals:**")
        st.markdown("• 5 seconds - High frequency monitoring")
        st.markdown("• 10 seconds - Balanced monitoring")
        st.markdown("• 30 seconds - Standard monitoring")
        st.markdown("• 1 minute - Low frequency monitoring")
        
        st.markdown("**Alert Thresholds:**")
        st.markdown("• 0.3 - Sensitive (more alerts)")
        st.markdown("• 0.5 - Balanced alerting")
        st.markdown("• 0.7 - Conservative (fewer alerts)")
        st.markdown("• 0.9 - Critical only")
        
        st.markdown("**Monitoring Scope:**")
        st.markdown("• Anomaly Detection")
        st.markdown("• Failure Prediction")
        st.markdown("• Security Threats")
        st.markdown("• Performance Metrics")
    
    # Quick start
    st.markdown("---")
    st.markdown("#### 🚀 Quick Start")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🟢 Start Basic Monitoring", use_container_width=True, type="primary"):
            st.session_state.monitoring_active = True
            st.session_state.monitoring_start = datetime.now()
            st.success("✅ Basic monitoring started!")
            time.sleep(1)
            st.rerun()
    
    with col2:
        if st.button("🔍 Start Full Monitoring", use_container_width=True):
            st.session_state.monitoring_active = True
            st.session_state.monitoring_start = datetime.now()
            st.success("✅ Full monitoring started!")
            time.sleep(1)
            st.rerun()
    
    with col3:
        if st.button("📊 View Demo", use_container_width=True):
            show_monitoring_demo()

def show_monitoring_demo():
    """Show monitoring demo"""
    
    st.markdown("---")
    st.markdown("### 📊 Monitoring Demo")
    
    st.info("This is a preview of what real-time monitoring looks like when active.")
    
    # Demo data
    demo_time = datetime.now()
    demo_data = generate_live_monitoring_data(demo_time)
    
    # Demo metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("CPU Usage", f"{demo_data['cpu_util']:.1f}%")
    with col2:
        st.metric("Memory Usage", f"{demo_data['memory_util']:.1f}%")
    with col3:
        st.metric("Error Rate", f"{demo_data['error_rate']:.3f}")
    with col4:
        st.metric("System Health", f"{demo_data['system_health']:.1f}%")
    
    # Demo chart
    demo_times = [demo_time - timedelta(minutes=i*5) for i in range(12, 0, -1)]
    demo_cpu_data = [np.random.uniform(30, 80) for _ in range(12)]
    demo_memory_data = [np.random.uniform(40, 75) for _ in range(12)]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=demo_times, y=demo_cpu_data, mode='lines+markers', name='CPU %'))
    fig.add_trace(go.Scatter(x=demo_times, y=demo_memory_data, mode='lines+markers', name='Memory %'))
    fig.update_layout(title="Demo: System Resource Utilization", height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("**Start monitoring to see live, interactive charts and real-time alerts!**")

# Export the function
__all__ = ['real_time_monitoring_page']
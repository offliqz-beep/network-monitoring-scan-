# -*- coding: utf-8 -*-
"""
AI Network Monitor with MySQL Database Integration
Created for XAMPP MySQL Database
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import time
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import mysql.connector
from mysql.connector import Error
import plotly.graph_objects as go
import plotly.express as px
import socket

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="AI Network Monitor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# Constants for Time Thresholds
# -------------------------
ONLINE_THRESHOLD_SECONDS = 60  # Device is online if data received within last 60 seconds
STALE_THRESHOLD_SECONDS = 120  # Data is stale if older than 120 seconds (2 minutes)
OFFLINE_THRESHOLD_SECONDS = 300  # Consider device offline if no data for 5 minutes

# -------------------------
# Database Connection Function
# -------------------------
def get_db_connection():
    """Create connection to MySQL database"""
    try:
        connection = mysql.connector.connect(
            host='localhost',
            database='network_monitor',
            user='root',
            password='',  # XAMPP default password is empty
            connection_timeout=5
        )
        return connection
    except Error as e:
        return None

# -------------------------
# Save Metrics to Database (Only when data is valid and recent)
# -------------------------
def save_to_database(devices, latency, packet_loss, bandwidth, prediction, data_age_seconds):
    """Save network metrics to database - only if data is valid and recent"""
    # Don't save if data is stale or offline (older than STALE_THRESHOLD_SECONDS)
    if data_age_seconds > STALE_THRESHOLD_SECONDS:
        return False
    
    # Don't save if all metrics are zero (indicates offline/error state)
    if devices == 0 and latency == 0 and packet_loss == 0 and bandwidth == 0:
        return False
    
    connection = get_db_connection()
    if connection:
        try:
            cursor = connection.cursor()
            
            # Convert numpy types to Python native types
            devices = int(devices)
            latency = float(latency)
            packet_loss = float(packet_loss)
            bandwidth = float(bandwidth)
            prediction = int(prediction)
            
            # Insert into network_metrics
            query = """
                INSERT INTO network_metrics 
                (devices, latency, packet_loss, bandwidth, congestion_prediction, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s)
            """
            current_time = datetime.now()
            cursor.execute(query, (devices, latency, packet_loss, bandwidth, prediction, current_time))
            metric_id = cursor.lastrowid
            
            # Generate and save recommendations
            advice_list, _ = network_advice(devices, latency, packet_loss, bandwidth, prediction)
            
            for advice in advice_list:
                rec_query = "INSERT INTO recommendations (metric_id, recommendation) VALUES (%s, %s)"
                cursor.execute(rec_query, (metric_id, advice))
            
            # Log the action
            log_query = "INSERT INTO system_logs (log_type, message) VALUES (%s, %s)"
            log_message = f'Network metrics saved - Devices: {devices}, Latency: {latency}, Prediction: {prediction}'
            cursor.execute(log_query, ('INFO', log_message))
            
            connection.commit()
            cursor.close()
            connection.close()
            return True
        except Error as e:
            return False
    return False

# -------------------------
# Predict Network Congestion
# -------------------------
def predict_network(devices, latency, packet_loss, bandwidth):
    """Predict network congestion with proper type conversion"""
    # Convert to float/int to ensure correct types
    devices = float(devices) if not isinstance(devices, (int, float)) else devices
    latency = float(latency)
    packet_loss = float(packet_loss)
    bandwidth = float(bandwidth)
    
    if model is None:
        # Demo logic
        if latency > 100 or packet_loss > 2 or bandwidth < 50 or devices > 15:
            return 1
        return 0
    
    sample = [[devices, latency, packet_loss, bandwidth]]
    prediction = model.predict(sample)[0]
    return int(prediction)

# -------------------------
# Fetch ThingSpeak Data with Timestamp Check
# -------------------------
@st.cache_data(ttl=5)
def fetch_thingspeak_data():
    """Fetch data from ThingSpeak and check if data is fresh"""
    try:
        url = f"http://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json?api_key={READ_API_KEY}&results=1"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        if 'feeds' in data and len(data['feeds']) > 0:
            latest = data['feeds'][0]
            
            # Get the timestamp of the last update
            last_update_str = latest.get('created_at')
            
            if last_update_str:
                # Parse the timestamp
                last_update = datetime.strptime(last_update_str, '%Y-%m-%dT%H:%M:%SZ')
                current_time = datetime.utcnow()
                
                # Calculate time difference in seconds
                time_diff = (current_time - last_update).total_seconds()
                
                # Check if data is too old (offline)
                if time_diff > OFFLINE_THRESHOLD_SECONDS:
                    # Device is offline - return zeros with age
                    return 0, 0.0, 0.0, 0.0, time_diff, last_update, "offline"
                elif time_diff > STALE_THRESHOLD_SECONDS:
                    # Data is stale but not completely offline
                    return 0, 0.0, 0.0, 0.0, time_diff, last_update, "stale"
            
            # Check if the feed has valid data (not None or empty)
            field1 = latest.get('field1')
            field2 = latest.get('field2')
            field3 = latest.get('field3')
            field4 = latest.get('field4')
            
            # If any field is None or empty, consider device offline
            if field1 is None or field2 is None or field3 is None or field4 is None:
                return 0, 0.0, 0.0, 0.0, time_diff if last_update_str else OFFLINE_THRESHOLD_SECONDS, last_update if last_update_str else None, "offline"
            
            # Convert to Python native types
            devices = int(field1) if field1 else 0
            latency = float(field2) if field2 else 0.0
            packet_loss = float(field3) if field3 else 0.0
            bandwidth = float(field4) if field4 else 0.0
            
            # If all values are zero, consider device offline
            if devices == 0 and latency == 0 and packet_loss == 0 and bandwidth == 0:
                return 0, 0.0, 0.0, 0.0, time_diff, last_update, "offline"
            
            # Determine status based on data freshness
            if time_diff <= ONLINE_THRESHOLD_SECONDS:
                status = "online"
            elif time_diff <= STALE_THRESHOLD_SECONDS:
                status = "recent"
            else:
                status = "stale"
                
            return devices, latency, packet_loss, bandwidth, time_diff, last_update, status
        else:
            return 0, 0.0, 0.0, 0.0, OFFLINE_THRESHOLD_SECONDS, None, "offline"
            
    except Exception as e:
        # Any exception (connection error, timeout, etc.) means device is offline
        return 0, 0.0, 0.0, 0.0, OFFLINE_THRESHOLD_SECONDS, None, "offline"

# -------------------------
# Check if ThingSpeak Device is Online (Based on Timestamp)
# -------------------------
def get_thingspeak_status():
    """Check ThingSpeak device status based on last update time"""
    try:
        url = f"http://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json?api_key={READ_API_KEY}&results=1"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        if 'feeds' in data and len(data['feeds']) > 0:
            latest = data['feeds'][0]
            
            # Get the timestamp of the last update
            last_update_str = latest.get('created_at')
            
            if last_update_str:
                # Parse the timestamp
                last_update = datetime.strptime(last_update_str, '%Y-%m-%dT%H:%M:%SZ')
                current_time = datetime.utcnow()
                
                # Calculate time difference in seconds
                time_diff = (current_time - last_update).total_seconds()
                
                # Check status based on time difference
                if time_diff <= ONLINE_THRESHOLD_SECONDS:
                    return "online", time_diff, last_update
                elif time_diff <= STALE_THRESHOLD_SECONDS:
                    return "recent", time_diff, last_update
                elif time_diff <= OFFLINE_THRESHOLD_SECONDS:
                    return "stale", time_diff, last_update
                else:
                    return "offline", time_diff, last_update
            
            return "offline", OFFLINE_THRESHOLD_SECONDS, None
        else:
            return "offline", OFFLINE_THRESHOLD_SECONDS, None
            
    except Exception as e:
        return "offline", OFFLINE_THRESHOLD_SECONDS, None

# -------------------------
# Get Database Statistics
# -------------------------
def get_db_statistics():
    """Get statistics from database"""
    connection = get_db_connection()
    if connection:
        try:
            cursor = connection.cursor()
            
            # Total records
            cursor.execute("SELECT COUNT(*) FROM network_metrics")
            total_records = cursor.fetchone()[0]
            
            # Congestion predictions count
            cursor.execute("SELECT COUNT(*) FROM network_metrics WHERE congestion_prediction = 1")
            congestion_count = cursor.fetchone()[0]
            
            # Average metrics
            cursor.execute("""
                SELECT AVG(latency), AVG(packet_loss), AVG(bandwidth), AVG(devices)
                FROM network_metrics
            """)
            avg_data = cursor.fetchone()
            
            cursor.close()
            connection.close()
            
            # Convert to Python native types
            return {
                'total_records': int(total_records) if total_records else 0,
                'congestion_count': int(congestion_count) if congestion_count else 0,
                'avg_latency': float(avg_data[0]) if avg_data[0] else 0.0,
                'avg_packet_loss': float(avg_data[1]) if avg_data[1] else 0.0,
                'avg_bandwidth': float(avg_data[2]) if avg_data[2] else 0.0,
                'avg_devices': float(avg_data[3]) if avg_data[3] else 0.0
            }
        except Error as e:
            return {}
    return {}

# -------------------------
# Load Historical Data
# -------------------------
@st.cache_data(ttl=60)
def load_historical_data(limit=100):
    """Load historical network metrics from database"""
    connection = get_db_connection()
    if connection:
        try:
            query = """
                SELECT id, timestamp, devices, latency, packet_loss, bandwidth, 
                       congestion_prediction, created_at
                FROM network_metrics
                ORDER BY timestamp DESC
                LIMIT %s
            """
            df = pd.read_sql(query, connection, params=(int(limit),))
            connection.close()
            return df
        except Error as e:
            return pd.DataFrame()
    return pd.DataFrame()

# -------------------------
# Load Recommendations History
# -------------------------
def load_recommendations_history(limit=50):
    """Load recommendations with metrics data"""
    connection = get_db_connection()
    if connection:
        try:
            query = """
                SELECT r.id, r.recommendation, r.created_at,
                       n.timestamp, n.devices, n.latency, n.packet_loss, n.bandwidth,
                       n.congestion_prediction
                FROM recommendations r
                JOIN network_metrics n ON r.metric_id = n.id
                ORDER BY r.created_at DESC
                LIMIT %s
            """
            df = pd.read_sql(query, connection, params=(limit,))
            connection.close()
            return df
        except Error as e:
            return pd.DataFrame()
    return pd.DataFrame()

# -------------------------
# Load System Logs
# -------------------------
def load_system_logs(limit=100):
    """Load system logs from database"""
    connection = get_db_connection()
    if connection:
        try:
            query = """
                SELECT id, log_type, message, created_at
                FROM system_logs
                ORDER BY created_at DESC
                LIMIT %s
            """
            df = pd.read_sql(query, connection, params=(limit,))
            connection.close()
            return df
        except Error as e:
            return pd.DataFrame()
    return pd.DataFrame()

# -------------------------
# Custom CSS
# -------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #fff 0%, #e0e0e0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
    }
    
    .metric-card {
        background: rgba(255,255,255,0.95);
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 8px 20px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .metric-label {
        font-size: 0.9rem;
        font-weight: 600;
        color: #667eea;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #2d3748;
        margin-bottom: 0.25rem;
    }
    
    .prediction-risk {
        background: linear-gradient(135deg, #f56565 0%, #e53e3e 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px;
        text-align: center;
    }
    
    .prediction-normal {
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px;
        text-align: center;
    }
    
    .recommendation-card {
        background: rgba(255,255,255,0.95);
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    
    .footer {
        text-align: center;
        padding: 2rem;
        color: rgba(255,255,255,0.7);
        font-size: 0.9rem;
    }
    
    .status-online {
        background: rgba(255,255,255,0.95);
        border-radius: 15px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid #48bb78;
        text-align: center;
    }
    
    .status-recent {
        background: rgba(255,255,255,0.95);
        border-radius: 15px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid #4299e1;
        text-align: center;
    }
    
    .status-stale {
        background: rgba(255,255,255,0.95);
        border-radius: 15px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid #ed8936;
        text-align: center;
    }
    
    .status-offline {
        background: rgba(255,255,255,0.95);
        border-radius: 15px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid #f56565;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------
# Load Model
# -------------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load("network_congestion_model.pkl")
        return model
    except:
        st.warning("⚠ Model file not found. Using demo mode.")
        return None

model = load_model()

# -------------------------
# ThingSpeak Configuration
# -------------------------
CHANNEL_ID = "3272879"
READ_API_KEY = "DVHBFJFGLFO80Y2N"

# -------------------------
# Generate Recommendations
# -------------------------
def network_advice(devices, latency, packet_loss, bandwidth, prediction):
    advice = []
    severity_levels = []
    
    # Don't generate advice if all metrics are zero (offline state)
    if devices == 0 and latency == 0 and packet_loss == 0 and bandwidth == 0:
        advice.append("⚠️ Network monitoring device is offline - No data available")
        severity_levels.append("warning")
        return advice, severity_levels
    
    if latency > 100:
        advice.append(f"⚠ High Latency ({latency:.1f}ms): Check router config, Enable QoS, Optimize routing")
        severity_levels.append("high")
    elif latency > 50:
        advice.append(f"⚠ Moderate Latency ({latency:.1f}ms): Monitor network traffic, consider optimization")
        severity_levels.append("medium")
    
    if packet_loss > 2:
        advice.append(f"⚠ Critical Packet Loss ({packet_loss:.2f}%): Check cables, switch ports, inspect interference")
        severity_levels.append("high")
    elif packet_loss > 1:
        advice.append(f"⚠ Packet Loss Detected ({packet_loss:.2f}%): Investigate network stability")
        severity_levels.append("medium")
    
    if bandwidth < 50:
        advice.append(f"⚠ Low Bandwidth ({bandwidth:.1f}Mbps): Upgrade ISP, limit heavy traffic apps")
        severity_levels.append("high")
    elif bandwidth < 100:
        advice.append(f"⚠ Moderate Bandwidth ({bandwidth:.1f}Mbps): Monitor usage patterns")
        severity_levels.append("medium")
    
    if devices > 15:
        advice.append(f"⚠ High Device Count ({devices} devices): Add APs, implement VLAN segmentation")
        severity_levels.append("high")
    elif devices > 10:
        advice.append(f"⚠ Growing Device Count ({devices} devices): Plan for network expansion")
        severity_levels.append("medium")
    
    if not advice:
        advice.append("✅ Network Operating Normally - All metrics within optimal ranges")
        severity_levels.append("good")
    
    if prediction == 1:
        advice.insert(0, "🚨 CRITICAL: AI predicts network congestion risk! Immediate action required.")
        severity_levels.insert(0, "critical")
    
    return advice, severity_levels

# -------------------------
# Format Time Difference
# -------------------------
def format_time_diff(seconds):
    """Format time difference in human readable format"""
    if seconds < 60:
        return f"{int(seconds)} seconds ago"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    elif seconds < 86400:
        hours = int(seconds / 3600)
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    else:
        days = int(seconds / 86400)
        return f"{days} day{'s' if days != 1 else ''} ago"

# -------------------------
# Main App
# -------------------------
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <div class="main-title">📡 AI Network Congestion Monitor</div>
        <div class="subtitle">Real-time network analytics with MySQL database integration</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 System Status")
        
        # Check ThingSpeak device status
        status, time_diff, last_update = get_thingspeak_status()
        
        if status == "online":
            st.success(f"✅ ThingSpeak Device: ONLINE")
            if last_update:
                st.info(f"📡 Last update: {format_time_diff(time_diff)}")
        elif status == "recent":
            st.info(f"🟢 ThingSpeak Device: RECENT DATA")
            if last_update:
                st.info(f"📡 Last update: {format_time_diff(time_diff)}")
        elif status == "stale":
            st.warning(f"⚠️ ThingSpeak Device: STALE DATA")
            if last_update:
                st.warning(f"⏰ No data for: {format_time_diff(time_diff)}")
        else:
            st.error(f"❌ ThingSpeak Device: OFFLINE")
            if last_update:
                st.error(f"⏰ Last data: {format_time_diff(time_diff)}")
            else:
                st.error("⏰ No data received from device")
        
        # Check database connection
        db_connection = get_db_connection()
        if db_connection:
            st.success("✅ MySQL Database: Connected")
            db_connection.close()
        else:
            st.error("❌ MySQL Database: Disconnected")
        
        st.markdown("---")
        
        # Database statistics
        stats = get_db_statistics()
        if stats and stats['total_records'] > 0:
            st.markdown("### 📈 Database Statistics")
            st.metric("Total Records", stats['total_records'])
            st.metric("Congestion Events", stats['congestion_count'])
            st.metric("Avg Devices", f"{stats['avg_devices']:.1f}")
            st.metric("Avg Latency", f"{stats['avg_latency']:.1f} ms")
        else:
            st.info("No data in database yet")
        
        st.markdown("---")
        st.markdown("### 🤖 AI Model")
        st.markdown("Machine Learning model analyzing network patterns to predict congestion.")
        
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📡 Live Monitor", "📊 Historical Data", "💡 Recommendations", "📝 System Logs"])
    
    with tab1:
        # Live monitoring
        placeholder = st.empty()
        
        # Auto-refresh loop
        refresh_count = 0
        
        while True:
            with placeholder.container():
                # Fetch data from ThingSpeak with timestamp
                devices, latency, packet_loss, bandwidth, time_diff, last_update, data_status = fetch_thingspeak_data()
                
                # Determine if data is usable (online or recent)
                data_usable = data_status in ["online", "recent"] and not (devices == 0 and latency == 0 and packet_loss == 0 and bandwidth == 0)
                
                # Display status message based on data freshness
                if data_status == "online":
                    st.markdown(f"""
                    <div class="status-online">
                        <strong>✅ DEVICE ONLINE - RECEIVING LIVE DATA</strong><br>
                        Data received {format_time_diff(time_diff)}<br>
                        <small>Last update: {last_update.strftime('%Y-%m-%d %H:%M:%S') if last_update else 'Unknown'} UTC</small>
                    </div>
                    """, unsafe_allow_html=True)
                elif data_status == "recent":
                    st.markdown(f"""
                    <div class="status-recent">
                        <strong>🟢 DEVICE RECENT - DATA AVAILABLE</strong><br>
                        Data received {format_time_diff(time_diff)}<br>
                        <small>Last update: {last_update.strftime('%Y-%m-%d %H:%M:%S') if last_update else 'Unknown'} UTC</small>
                    </div>
                    """, unsafe_allow_html=True)
                elif data_status == "stale":
                    st.markdown(f"""
                    <div class="status-stale">
                        <strong>⚠️ STALE DATA - DEVICE MAY BE OFFLINE</strong><br>
                        No data received for {format_time_diff(time_diff)}<br>
                        <small>Last data: {last_update.strftime('%Y-%m-%d %H:%M:%S') if last_update else 'Unknown'} UTC</small>
                    </div>
                    """, unsafe_allow_html=True)
                else:  # offline
                    st.markdown(f"""
                    <div class="status-offline">
                        <strong>❌ DEVICE OFFLINE - NO DATA RECEIVED</strong><br>
                        {f'Last data received {format_time_diff(time_diff)}' if last_update else 'No data ever received'}<br>
                        <small>Waiting for device to come online...</small>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div style="text-align: right; color: rgba(255,255,255,0.8); margin-bottom: 1rem;">
                    Dashboard updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                </div>
                """, unsafe_allow_html=True)
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Active Devices</div>
                        <div class="metric-value">{devices if data_usable else 0}</div>
                        <div class="metric-unit">connected devices</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    color = "#f56565" if latency > 100 else "#48bb78" if latency < 50 else "#ed8936"
                    if not data_usable:
                        color = "#a0aec0"
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Network Latency</div>
                        <div class="metric-value" style="color: {color}">{latency if data_usable else 0:.1f}</div>
                        <div class="metric-unit">milliseconds</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    color = "#f56565" if packet_loss > 2 else "#48bb78" if packet_loss < 1 else "#ed8936"
                    if not data_usable:
                        color = "#a0aec0"
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Packet Loss</div>
                        <div class="metric-value" style="color: {color}">{packet_loss if data_usable else 0:.2f}</div>
                        <div class="metric-unit">percentage</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    color = "#f56565" if bandwidth < 50 else "#48bb78" if bandwidth > 100 else "#ed8936"
                    if not data_usable:
                        color = "#a0aec0"
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Bandwidth</div>
                        <div class="metric-value" style="color: {color}">{bandwidth if data_usable else 0:.1f}</div>
                        <div class="metric-unit">Mbps</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Predict (only if data is usable)
                if data_usable:
                    prediction = predict_network(devices, latency, packet_loss, bandwidth)
                else:
                    prediction = 0
                
                # Save to database only if data is fresh and every 5th refresh
                refresh_count += 1
                if data_usable and refresh_count % 5 == 0 and time_diff <= STALE_THRESHOLD_SECONDS:
                    if save_to_database(devices, latency, packet_loss, bandwidth, prediction, time_diff):
                        st.toast("✅ Fresh data saved to database!", icon="💾")
                elif not data_usable and refresh_count % 10 == 0:
                    if data_status == "stale":
                        st.toast(f"⚠️ Stale data detected - Last update {format_time_diff(time_diff)}", icon="⚠️")
                    elif data_status == "offline":
                        st.toast("❌ Device offline - No data to save", icon="❌")
                
                # Display prediction
                st.markdown("---")
                st.markdown("### 🔮 AI Prediction")
                
                if not data_usable:
                    if data_status == "stale":
                        st.markdown(f"""
                        <div class="status-stale">
                            <div class="prediction-title">⚠️ PREDICTION UNAVAILABLE - STALE DATA</div>
                            <div>Waiting for fresh data from monitoring device...<br>
                            <small>Last data received: {format_time_diff(time_diff)}</small></div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="status-offline">
                            <div class="prediction-title">⚠️ NO DATA AVAILABLE</div>
                            <div>Waiting for network monitoring device to come online...</div>
                        </div>
                        """, unsafe_allow_html=True)
                elif prediction == 1:
                    st.markdown("""
                    <div class="prediction-risk">
                        <div class="prediction-title">🚨 NETWORK CONGESTION RISK DETECTED</div>
                        <div>AI model predicts high probability of network congestion.</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="prediction-normal">
                        <div class="prediction-title">✅ NETWORK OPERATING NORMALLY</div>
                        <div>AI model indicates stable network conditions.</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Recommendations
                st.markdown("### 💡 IT Recommendations")
                display_devices = devices if data_usable else 0
                display_latency = latency if data_usable else 0
                display_packet_loss = packet_loss if data_usable else 0
                display_bandwidth = bandwidth if data_usable else 0
                
                advice_list, severity_levels = network_advice(display_devices, display_latency, 
                                                             display_packet_loss, display_bandwidth, prediction)
                
                for advice, severity in zip(advice_list, severity_levels):
                    if severity == "warning":
                        bg_color = "#fef3c7"
                        border_color = "#f59e0b"
                    elif severity in ["critical", "high"]:
                        bg_color = "#fee2e2"
                        border_color = "#dc2626"
                    elif severity == "medium":
                        bg_color = "#fef3c7"
                        border_color = "#f59e0b"
                    else:
                        bg_color = "#d1fae5"
                        border_color = "#10b981"
                    
                    st.markdown(f"""
                    <div class="recommendation-card" style="background: {bg_color}; border-left-color: {border_color};">
                        <div class="recommendation-text">{advice}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            time.sleep(5)
    
    with tab2:
        st.markdown("### 📊 Historical Network Data")
        
        historical_df = load_historical_data(100)
        
        if not historical_df.empty:
            # Filters
            col1, col2 = st.columns(2)
            with col1:
                date_range = st.date_input("Select Date Range", [])
            with col2:
                show_congestion_only = st.checkbox("Show only congestion events")
            
            # Apply filters
            df_filtered = historical_df.copy()
            if show_congestion_only:
                df_filtered = df_filtered[df_filtered['congestion_prediction'] == 1]
            
            # Display metrics over time
            if not df_filtered.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_filtered['timestamp'], y=df_filtered['latency'], 
                                         mode='lines+markers', name='Latency (ms)'))
                fig.add_trace(go.Scatter(x=df_filtered['timestamp'], y=df_filtered['bandwidth'], 
                                         mode='lines+markers', name='Bandwidth (Mbps)', yaxis='y2'))
                
                fig.update_layout(
                    title="Network Metrics Over Time",
                    xaxis_title="Time",
                    yaxis_title="Latency (ms)",
                    yaxis2=dict(title="Bandwidth (Mbps)", overlaying='y', side='right'),
                    template="plotly_white",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Data table
                st.markdown("### 📋 Data Table")
                st.dataframe(df_filtered, use_container_width=True)
                
                # Download button
                csv = df_filtered.to_csv(index=False)
                st.download_button("📥 Download Data as CSV", csv, "network_metrics.csv", "text/csv")
            else:
                st.info("No data matches the selected filters.")
        else:
            st.info("No historical data available yet. Data will appear as monitoring continues.")
    
    with tab3:
        st.markdown("### 💡 IT Recommendations History")
        
        recommendations_df = load_recommendations_history(50)
        
        if not recommendations_df.empty:
            for idx, row in recommendations_df.iterrows():
                with st.expander(f"Recommendation from {row['created_at']} - Devices: {row['devices']}"):
                    st.write(f"**Recommendation:** {row['recommendation']}")
                    st.write(f"**Metrics:** Latency: {row['latency']:.1f}ms, Packet Loss: {row['packet_loss']:.2f}%, Bandwidth: {row['bandwidth']:.1f}Mbps")
                    st.write(f"**Congestion Prediction:** {'Yes' if row['congestion_prediction'] == 1 else 'No'}")
        else:
            st.info("No recommendations available yet.")
    
    with tab4:
        st.markdown("### 📝 System Logs")
        
        logs_df = load_system_logs(100)
        
        if not logs_df.empty:
            # Color code log types
            def color_log_type(log_type):
                if log_type == 'ERROR':
                    return '🔴'
                elif log_type == 'WARNING':
                    return '🟡'
                else:
                    return '🟢'
            
            logs_df['icon'] = logs_df['log_type'].apply(color_log_type)
            logs_df['display'] = logs_df['icon'] + ' ' + logs_df['log_type']
            
            st.dataframe(
                logs_df[['created_at', 'display', 'message']],
                use_container_width=True,
                column_config={
                    'created_at': 'Timestamp',
                    'display': 'Type',
                    'message': 'Message'
                }
            )
            
            # Clear logs button
            if st.button("🗑️ Clear Logs", type="secondary"):
                connection = get_db_connection()
                if connection:
                    cursor = connection.cursor()
                    cursor.execute("DELETE FROM system_logs")
                    connection.commit()
                    cursor.close()
                    connection.close()
                    st.success("Logs cleared!")
                    st.rerun()
        else:
            st.info("No logs available yet.")

if __name__ == "__main__":
    main()
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# Configuration
LOGS_FILE = 'parsed_access_log.csv'
ANOMALY_FILE = 'diecast_ip_anomaly_scores.csv'

st.set_page_config(page_title="Diecast Bot Monitor", layout="wide")

@st.cache_data
def load_data():
    """Loads and preprocesses the data."""
    # Load logs
    try:
        df_req = pd.read_csv(LOGS_FILE)
        df_req['timestamp'] = pd.to_datetime(df_req['timestamp'])
    except FileNotFoundError:
        st.error(f"File not found: {LOGS_FILE}")
        return None, None

    # Load anomaly scores
    try:
        df_ip = pd.read_csv(ANOMALY_FILE)
    except FileNotFoundError:
        st.error(f"File not found: {ANOMALY_FILE}")
        return None, None
        
    return df_req, df_ip

def overview_page(df_req, df_ip, suspicious_ips):
    """Displays the Overview page."""
    st.header("Overview")
    
    # Metrics
    total_requests = len(df_req)
    total_ips = df_ip['ip'].nunique()
    num_suspicious = len(suspicious_ips)
    
    # Calculate traffic from suspicious IPs
    suspicious_ip_set = set(suspicious_ips['ip'])
    suspicious_reqs = df_req[df_req['ip'].isin(suspicious_ip_set)]
    num_suspicious_reqs = len(suspicious_reqs)
    suspicious_traffic_pct = (num_suspicious_reqs / total_requests) * 100 if total_requests > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Requests", f"{total_requests:,}")
    col2.metric("Total Unique IPs", f"{total_ips:,}")
    col3.metric("Suspicious IPs", f"{num_suspicious:,}")
    col4.metric("Suspicious Traffic", f"{suspicious_traffic_pct:.1f}%")
    
    # Time Series Chart
    st.subheader("Traffic Over Time")
    
    # Resample to 10 minutes
    df_req['time_bin'] = df_req['timestamp'].dt.floor('10min')
    
    # Tag requests as suspicious
    df_req['is_suspicious'] = df_req['ip'].isin(suspicious_ip_set)
    
    # Group by time
    traffic_over_time = df_req.groupby('time_bin').size().reset_index(name='Total Requests')
    suspicious_over_time = df_req[df_req['is_suspicious']].groupby('time_bin').size().reset_index(name='Suspicious Requests')
    
    # Merge
    chart_data = pd.merge(traffic_over_time, suspicious_over_time, on='time_bin', how='left').fillna(0)
    
    fig = px.line(chart_data, x='time_bin', y=['Total Requests', 'Suspicious Requests'], 
                  title="Requests per 10 Minutes", labels={'time_bin': 'Time', 'value': 'Requests'})
    st.plotly_chart(fig, use_container_width=True)

def suspicious_ips_page(df_req, df_ip_filtered):
    """Displays the Top Suspicious IPs page."""
    st.header("Top Suspicious IPs")
    
    # Sort by anomaly score (ascending = more anomalous)
    df_display = df_ip_filtered.sort_values('anomaly_score', ascending=True)
    
    # Main Table
    st.dataframe(df_display, use_container_width=True)
    
    # IP Drilldown
    st.subheader("IP Drilldown")
    selected_ip = st.selectbox("Select an IP to analyze:", df_display['ip'].head(50))
    
    if selected_ip:
        ip_data = df_req[df_req['ip'] == selected_ip]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Requests over Time for {selected_ip}**")
            ip_time = ip_data.set_index('timestamp').resample('10min').size()
            st.line_chart(ip_time)
            
        with col2:
            st.markdown(f"**Top Paths for {selected_ip}**")
            top_paths = ip_data['path'].value_counts().head(10)
            st.bar_chart(top_paths)

def ua_intelligence_page(df_req, df_ip, suspicious_ips):
    """Displays the User-Agent Intelligence page."""
    st.header("User-Agent Intelligence")
    
    suspicious_ip_set = set(suspicious_ips['ip'])
    
    # Categorize UAs
    def categorize_ua(ua):
        if not isinstance(ua, str): return "Unknown"
        ua_lower = ua.lower()
        if any(x in ua_lower for x in ["bot", "crawl", "spider", "monitoring", "googlebot"]):
            return "Known Bot"
        if any(x in ua_lower for x in ["python", "curl", "requests"]):
            return "Suspicious Script"
        return "Likely Browser"

    # We need to aggregate by UA first to avoid re-running categorization on every row if possible, 
    # but for accuracy we should do it on unique UAs
    unique_uas = df_req['user_agent'].unique()
    ua_categories = {ua: categorize_ua(ua) for ua in unique_uas}
    
    df_req['ua_category'] = df_req['user_agent'].map(ua_categories)
    
    # Aggregate stats
    ua_stats = df_req.groupby('user_agent').agg(
        total_requests=('ip', 'count'),
        ua_category=('ua_category', 'first')
    ).reset_index()
    
    # Count requests from anomalous IPs per UA
    # This is a bit heavier, let's do it efficiently
    anomalous_reqs = df_req[df_req['ip'].isin(suspicious_ip_set)]
    anomalous_counts = anomalous_reqs['user_agent'].value_counts().reset_index()
    anomalous_counts.columns = ['user_agent', 'requests_from_anomalous_ips']
    
    ua_stats = pd.merge(ua_stats, anomalous_counts, on='user_agent', how='left').fillna(0)
    ua_stats['requests_from_anomalous_ips'] = ua_stats['requests_from_anomalous_ips'].astype(int)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("User-Agent Categories")
        cat_counts = df_req['ua_category'].value_counts()
        fig = px.pie(values=cat_counts.values, names=cat_counts.index, title="Traffic by UA Category")
        st.plotly_chart(fig)
        
    with col2:
        st.subheader("Top User-Agents")
        st.dataframe(ua_stats.sort_values('total_requests', ascending=False).head(20), hide_index=True)

def path_abuse_page(df_req, suspicious_ips):
    """Displays the Path Abuse Analysis page."""
    st.header("Path Abuse Analysis")
    
    suspicious_ip_set = set(suspicious_ips['ip'])
    suspicious_reqs = df_req[df_req['ip'].isin(suspicious_ip_set)]
    
    if suspicious_reqs.empty:
        st.info("No suspicious traffic found based on current filters.")
        return

    # Aggregate by path
    path_stats = suspicious_reqs.groupby('path').agg(
        total_requests_from_suspicious_ips=('ip', 'count'),
        number_of_unique_suspicious_ips=('ip', 'nunique')
    ).reset_index()
    
    path_stats = path_stats.sort_values('total_requests_from_suspicious_ips', ascending=False).head(20)
    
    # Chart
    fig = px.bar(path_stats, x='path', y='total_requests_from_suspicious_ips', 
                 title="Top Paths Targeted by Suspicious IPs",
                 hover_data=['number_of_unique_suspicious_ips'])
    st.plotly_chart(fig, use_container_width=True)
    
    # Table
    st.dataframe(path_stats, hide_index=True)

def main():
    # Load Data
    df_req, df_ip = load_data()
    if df_req is None or df_ip is None:
        return

    # Sidebar
    st.sidebar.title("Diecast Bot Monitor")
    page = st.sidebar.radio("Navigation", ["Overview", "Top Suspicious IPs", "User-Agent Intelligence", "Path Abuse Analysis"])
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Global Filters")
    
    min_requests = st.sidebar.number_input("Min Total Requests", min_value=1, value=10)
    
    # Anomaly score usually ranges from -1 to 1, or similar. 
    # Lower is more anomalous. Let's find the range.
    min_score = float(df_ip['anomaly_score'].min())
    max_score = float(df_ip['anomaly_score'].max())
    
    anomaly_threshold = st.sidebar.slider("Anomaly Score Threshold (Include IPs below this)", 
                                          min_value=min_score, max_value=max_score, value=0.0, step=0.01)
    
    show_only_anomalies = st.sidebar.checkbox("Show Only Predicted Anomalies (is_anomaly=1)", value=False)

    # Apply Filters to df_ip
    # 1. Min requests
    df_ip_filtered = df_ip[df_ip['total_requests'] >= min_requests]
    
    # 2. Anomaly Threshold (keep IPs with score <= threshold)
    df_ip_filtered = df_ip_filtered[df_ip_filtered['anomaly_score'] <= anomaly_threshold]
    
    # 3. Is Anomaly Flag
    if show_only_anomalies:
        df_ip_filtered = df_ip_filtered[df_ip_filtered['is_anomaly'] == 1]

    # Routing
    if page == "Overview":
        overview_page(df_req, df_ip, df_ip_filtered)
    elif page == "Top Suspicious IPs":
        suspicious_ips_page(df_req, df_ip_filtered)
    elif page == "User-Agent Intelligence":
        ua_intelligence_page(df_req, df_ip, df_ip_filtered)
    elif page == "Path Abuse Analysis":
        path_abuse_page(df_req, df_ip_filtered)

if __name__ == "__main__":
    main()

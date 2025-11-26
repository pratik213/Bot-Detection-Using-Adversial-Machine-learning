import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Configuration
INPUT_FILE = 'parsed_access_log.csv'
OUTPUT_FILE = 'diecast_ip_anomaly_scores.csv'
SUSPICIOUS_KEYWORDS = ["bot", "crawl", "spider", "python", "curl", "monitoring", "googlebot"]

def load_and_preprocess(file_path):
    """Loads the CSV and performs basic preprocessing."""
    print(f"Loading {file_path}...")
    df = pd.read_csv(file_path)
    
    # Convert timestamp to datetime
    print("Converting timestamps...")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Create minute column
    df['minute'] = df['timestamp'].dt.floor('min')
    
    return df

def create_ip_features(df):
    """Aggregates features per IP."""
    print("Aggregating features per IP...")
    
    # Helper for 4xx errors
    df['is_4xx'] = (df['status'] >= 400) & (df['status'] < 500)
    
    # Helper for bot flag
    ua_pattern = '|'.join(SUSPICIOUS_KEYWORDS)
    df['ua_bot_flag'] = df['user_agent'].str.contains(ua_pattern, case=False, na=False).astype(int)

    # Group by IP
    # We need to calculate requests per minute first to get max/avg
    requests_per_min = df.groupby(['ip', 'minute']).size().reset_index(name='req_count')
    rpm_stats = requests_per_min.groupby('ip')['req_count'].agg(['mean', 'max']).rename(columns={'mean': 'avg_requests_per_minute', 'max': 'max_requests_per_minute'})
    
    # Main aggregation
    ip_stats = df.groupby('ip').agg(
        total_requests=('ip', 'count'),
        unique_paths=('path', 'nunique'),
        error_rate_4xx=('is_4xx', 'mean'),
        ua_bot_flag=('ua_bot_flag', 'max')
    )
    
    # Merge RPM stats
    df_ip = ip_stats.join(rpm_stats).fillna(0)
    
    # Calculate repeated_path_ratio
    # Handle divide by zero safely (though total_requests should be >= 1)
    df_ip['repeated_path_ratio'] = 1 - (df_ip['unique_paths'] / df_ip['total_requests'])
    
    return df_ip

def train_isolation_forest(df_ip):
    """Trains Isolation Forest and adds anomaly scores."""
    print("Training Isolation Forest...")
    
    features = [
        "total_requests", 
        "unique_paths", 
        "repeated_path_ratio",
        "avg_requests_per_minute", 
        "max_requests_per_minute",
        "error_rate_4xx"
    ]
    
    X = df_ip[features].fillna(0)
    
    # Scaling (optional but recommended)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    # contamination=0.05 means we expect top 5% to be anomalies
    model = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
    model.fit(X_scaled)
    
    # Predict
    # decision_function: lower is more anomalous
    # predict: -1 for outlier, 1 for inlier. We want 1 for anomaly, 0 for normal usually, or just keep -1/1
    # The request asks for anomaly_label (0 = normal, 1 = anomaly)
    
    df_ip['anomaly_score'] = model.decision_function(X_scaled)
    # Convert predict output: -1 (anomaly) -> 1, 1 (normal) -> 0
    preds = model.predict(X_scaled)
    df_ip['is_anomaly'] = np.where(preds == -1, 1, 0)
    
    return df_ip

def main():
    # 1. Load
    try:
        df = load_and_preprocess(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found.")
        return

    # 2 & 3. Feature Engineering
    df_ip = create_ip_features(df)
    
    # 4 & 5 & 6. Train and Score
    df_ip = train_isolation_forest(df_ip)
    
    # 7. Inspect Results
    print("\n--- Anomaly Detection Results ---")
    
    # Sort by anomaly_score ascending (lower score = more anomalous)
    top_anomalies = df_ip.sort_values('anomaly_score', ascending=True).head(20)
    
    cols_to_show = [
        'total_requests', 
        'max_requests_per_minute', 
        'repeated_path_ratio', 
        'ua_bot_flag', 
        'anomaly_score', 
        'is_anomaly'
    ]
    
    print("\nTop 20 Suspicious IPs:")
    print(top_anomalies[cols_to_show])
    
    # Sanity check
    bot_anomalies = top_anomalies[top_anomalies['ua_bot_flag'] == 1]
    print(f"\nNumber of top 20 anomalies with ua_bot_flag=1: {len(bot_anomalies)}")
    
    # 8. Save
    print(f"\nSaving results to {OUTPUT_FILE}...")
    df_ip.to_csv(OUTPUT_FILE)
    print("Done.")

if __name__ == "__main__":
    main()

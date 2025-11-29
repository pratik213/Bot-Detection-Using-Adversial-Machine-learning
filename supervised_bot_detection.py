import pandas as pd
import numpy as np
from scipy.stats import entropy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
INPUT_FILE = 'parsed_access_log.csv'
OUTPUT_FILE = 'diecast_bot_predictions_supervised.csv'
PLOT_FILE = 'supervised_evaluation.png'
SUSPICIOUS_KEYWORDS = ["bot", "crawl", "spider", "python", "curl", "monitoring", "googlebot"]
LOGIN_API_KEYWORDS = ["/login", "/user", "/signin", "/api", "/search"]

def load_and_preprocess(file_path):
    print(f"Loading {file_path}...")
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['minute'] = df['timestamp'].dt.floor('min')
    return df

def calculate_entropy(series):
    value_counts = series.value_counts()
    return entropy(value_counts)

def create_features(df):
    print("Feature Engineering...")
    
    # Basic helpers
    df['is_4xx'] = (df['status'] >= 400) & (df['status'] < 500)
    df['is_5xx'] = (df['status'] >= 500) & (df['status'] < 600)
    df['is_3xx'] = (df['status'] >= 300) & (df['status'] < 400)
    
    # Target Label: ua_bot_flag
    ua_pattern = '|'.join(SUSPICIOUS_KEYWORDS)
    df['ua_bot_flag'] = df['user_agent'].str.contains(ua_pattern, case=False, na=False).astype(int)
    
    login_pattern = '|'.join(LOGIN_API_KEYWORDS)
    df['is_login_api'] = df['path'].str.contains(login_pattern, case=False, na=False).astype(int)

    # Time Intervals
    df = df.sort_values(['ip', 'timestamp'])
    df['prev_timestamp'] = df.groupby('ip')['timestamp'].shift(1)
    df['interval_sec'] = (df['timestamp'] - df['prev_timestamp']).dt.total_seconds()
    interval_stats = df.groupby('ip')['interval_sec'].agg(['mean', 'std']).rename(columns={'mean': 'mean_interval_sec', 'std': 'std_interval_sec'})
    
    # RPM
    requests_per_min = df.groupby(['ip', 'minute']).size().reset_index(name='req_count')
    rpm_stats = requests_per_min.groupby('ip')['req_count'].agg(['mean', 'max']).rename(columns={'mean': 'avg_requests_per_minute', 'max': 'max_requests_per_minute'})
    
    # Entropy
    path_entropy = df.groupby('ip')['path'].apply(calculate_entropy).rename('path_entropy')
    
    # Aggregation
    ip_stats = df.groupby('ip').agg(
        total_requests=('ip', 'count'),
        unique_paths=('path', 'nunique'),
        error_rate_4xx=('is_4xx', 'mean'),
        rate_5xx=('is_5xx', 'mean'),
        rate_3xx=('is_3xx', 'mean'),
        login_api_ratio=('is_login_api', 'mean'),
        ua_bot_flag=('ua_bot_flag', 'max') # This is our TARGET
    )
    
    df_ip = ip_stats.join([rpm_stats, interval_stats, path_entropy]).fillna(0)
    
    df_ip['repeated_path_ratio'] = 1 - (df_ip['unique_paths'] / df_ip['total_requests'])
    df_ip['burstiness'] = df_ip['max_requests_per_minute'] / (df_ip['avg_requests_per_minute'] + 1e-6)
    
    return df_ip

def train_supervised(df_ip):
    print("\n--- Training Random Forest (Supervised) ---")
    
    feature_cols = [
        "total_requests", "unique_paths", "repeated_path_ratio",
        "avg_requests_per_minute", "max_requests_per_minute",
        "error_rate_4xx", "rate_5xx", "rate_3xx",
        "mean_interval_sec", "std_interval_sec", "burstiness",
        "path_entropy", "login_api_ratio"
    ]
    
    X = df_ip[feature_cols]
    y = df_ip['ua_bot_flag']
    
    # Check if we have enough data
    if y.sum() == 0:
        print("Error: No bots found in ua_bot_flag. Cannot train supervised model.")
        return None
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Train
    # class_weight='balanced' helps if bots are rare
    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    print("\nConfusion Matrix (Test Set):")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Human', 'Bot']))
    
    # Feature Importance
    importances = pd.DataFrame({
        'feature': feature_cols,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 5 Predictive Features:")
    print(importances.head(5))
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # CM Plot
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title('Confusion Matrix')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    
    # ROC Plot
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    ax2.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    ax2.plot([0, 1], [0, 1], 'k--')
    ax2.set_title('ROC Curve')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(PLOT_FILE)
    print(f"Plots saved to {PLOT_FILE}")
    
    # Return full predictions
    df_ip['bot_probability'] = clf.predict_proba(X)[:, 1]
    df_ip['predicted_bot'] = clf.predict(X)
    
    return df_ip

def main():
    try:
        df = load_and_preprocess(INPUT_FILE)
    except FileNotFoundError:
        print(f"File {INPUT_FILE} not found.")
        return

    df_ip = create_features(df)
    
    df_ip = train_supervised(df_ip)
    
    if df_ip is not None:
        print(f"Saving predictions to {OUTPUT_FILE}...")
        df_ip.to_csv(OUTPUT_FILE)

if __name__ == "__main__":
    main()

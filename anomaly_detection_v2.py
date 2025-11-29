import pandas as pd
import numpy as np
from scipy.stats import entropy
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Configuration
INPUT_FILE = 'parsed_access_log.csv'
OUTPUT_FILE = 'diecast_ip_anomaly_scores_v2.csv'
EVAL_PLOT_FILE = 'evaluation_plots_v2.png'
SUSPICIOUS_KEYWORDS = ["bot", "crawl", "spider", "python", "curl", "monitoring", "googlebot"]
LOGIN_API_KEYWORDS = ["/login", "/user", "/signin", "/api", "/search"]

def load_and_preprocess(file_path):
    """Loads the CSV and performs basic preprocessing."""
    print(f"Loading {file_path}...")
    df = pd.read_csv(file_path)
    print("Converting timestamps...")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['minute'] = df['timestamp'].dt.floor('min')
    return df

def calculate_entropy(series):
    """Calculates Shannon entropy of a series."""
    value_counts = series.value_counts()
    return entropy(value_counts)

def create_advanced_features(df):
    """Aggregates advanced features per IP."""
    print("Aggregating advanced features per IP...")
    
    # 1. Pre-aggregation helpers
    df['is_4xx'] = (df['status'] >= 400) & (df['status'] < 500)
    df['is_5xx'] = (df['status'] >= 500) & (df['status'] < 600)
    df['is_3xx'] = (df['status'] >= 300) & (df['status'] < 400)
    
    ua_pattern = '|'.join(SUSPICIOUS_KEYWORDS)
    df['ua_bot_flag'] = df['user_agent'].str.contains(ua_pattern, case=False, na=False).astype(int)
    
    login_pattern = '|'.join(LOGIN_API_KEYWORDS)
    df['is_login_api'] = df['path'].str.contains(login_pattern, case=False, na=False).astype(int)

    # 2. Time-based features (Intervals)
    print("Calculating time intervals...")
    df = df.sort_values(['ip', 'timestamp'])
    df['prev_timestamp'] = df.groupby('ip')['timestamp'].shift(1)
    df['interval_sec'] = (df['timestamp'] - df['prev_timestamp']).dt.total_seconds()
    
    interval_stats = df.groupby('ip')['interval_sec'].agg(['mean', 'std']).rename(columns={'mean': 'mean_interval_sec', 'std': 'std_interval_sec'})
    
    # 3. RPM stats (for burstiness)
    requests_per_min = df.groupby(['ip', 'minute']).size().reset_index(name='req_count')
    rpm_stats = requests_per_min.groupby('ip')['req_count'].agg(['mean', 'max']).rename(columns={'mean': 'avg_requests_per_minute', 'max': 'max_requests_per_minute'})
    
    # 4. Path Entropy
    print("Calculating path entropy...")
    path_entropy = df.groupby('ip')['path'].apply(calculate_entropy).rename('path_entropy')
    
    # 5. Main Aggregation
    ip_stats = df.groupby('ip').agg(
        total_requests=('ip', 'count'),
        unique_paths=('path', 'nunique'),
        error_rate_4xx=('is_4xx', 'mean'),
        rate_5xx=('is_5xx', 'mean'),
        rate_3xx=('is_3xx', 'mean'),
        login_api_ratio=('is_login_api', 'mean'),
        ua_bot_flag=('ua_bot_flag', 'max')
    )
    
    # 6. Merge all
    df_ip = ip_stats.join([rpm_stats, interval_stats, path_entropy]).fillna(0)
    
    # Derived features
    df_ip['repeated_path_ratio'] = 1 - (df_ip['unique_paths'] / df_ip['total_requests'])
    df_ip['burstiness'] = df_ip['max_requests_per_minute'] / (df_ip['avg_requests_per_minute'] + 1e-6)
    
    return df_ip

def train_and_evaluate(df_ip):
    """Trains Isolation Forest with different contamination levels and evaluates."""
    print("Training and evaluating models...")
    
    feature_cols_v2 = [
        "total_requests", "unique_paths", "repeated_path_ratio",
        "avg_requests_per_minute", "max_requests_per_minute",
        "error_rate_4xx", "rate_5xx", "rate_3xx",
        "mean_interval_sec", "std_interval_sec", "burstiness",
        "path_entropy", "login_api_ratio"
    ]
    
    X = df_ip[feature_cols_v2].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    contamination_levels = [0.01, 0.02, 0.05, 0.10]
    results = []
    best_f1 = -1
    best_model_data = None
    
    y_true = df_ip['ua_bot_flag']
    
    for cont in contamination_levels:
        model = IsolationForest(contamination=cont, random_state=42, n_jobs=-1)
        model.fit(X_scaled)
        
        scores = model.decision_function(X_scaled)
        preds = model.predict(X_scaled)
        # Convert: -1 (anomaly) -> 1, 1 (normal) -> 0
        y_pred = np.where(preds == -1, 1, 0)
        
        # Metrics
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_true, -scores)
        except:
            auc = 0
            
        results.append({
            'contamination': cont,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'auc': auc
        })
        
        if f1 > best_f1:
            best_f1 = f1
            best_model_data = {
                'contamination': cont,
                'scores': scores,
                'preds': y_pred,
                'y_true': y_true
            }
            
    # Print Results
    results_df = pd.DataFrame(results)
    print("\n--- Evaluation Results by Contamination Level ---")
    print(results_df)
    
    best_cont = best_model_data['contamination']
    print(f"\nBest Contamination Level: {best_cont} (F1 Score: {best_f1:.4f})")
    
    # Save best results to df
    df_ip['anomaly_score'] = best_model_data['scores']
    df_ip['is_anomaly'] = best_model_data['preds']
    
    # Visualizations for best model
    plot_evaluation(best_model_data['y_true'], best_model_data['preds'], best_model_data['scores'], best_cont)
    
    return df_ip

def plot_evaluation(y_true, y_pred, scores, contamination):
    """Plots Confusion Matrix and ROC Curve."""
    print(f"Generating evaluation plots for contamination={contamination}...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    metrics = ['TP', 'FN', 'FP', 'TN']
    values = [tp, fn, fp, tn]
    
    ax1.bar(metrics, values, color=['green', 'red', 'orange', 'blue'])
    ax1.set_title(f'Confusion Matrix (Contamination={contamination})')
    ax1.set_ylabel('Count')
    
    # ROC Curve
    try:
        fpr, tpr, _ = roc_curve(y_true, -scores)
        auc = roc_auc_score(y_true, -scores)
        ax2.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
        ax2.plot([0, 1], [0, 1], 'k--')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curve')
        ax2.legend()
    except Exception as e:
        print(f"Could not plot ROC: {e}")
        
    plt.tight_layout()
    plt.savefig(EVAL_PLOT_FILE)
    print(f"Plots saved to {EVAL_PLOT_FILE}")

def main():
    # 1. Load
    try:
        df = load_and_preprocess(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found.")
        return

    # 2. Advanced Feature Engineering
    df_ip = create_advanced_features(df)
    
    # 3 & 4. Train, Evaluate, Visualize
    df_ip = train_and_evaluate(df_ip)
    
    # 5. Save
    print(f"\nSaving results to {OUTPUT_FILE}...")
    df_ip.to_csv(OUTPUT_FILE)
    print("Done.")

if __name__ == "__main__":
    main()

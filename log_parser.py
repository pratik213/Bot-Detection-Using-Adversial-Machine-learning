import re
import pandas as pd
import os
from datetime import datetime

# Configuration
LOG_FILE = 'access_ssl_log.processed'
OUTPUT_FILE = 'parsed_access_log.csv'

# Regex pattern provided in requirements
LOG_PATTERN = re.compile(
    r'(?P<ip>\S+) \S+ \S+ \[(?P<datetime>[^\]]+)\] "(?P<method>\S+) (?P<path>\S+) (?P<protocol>[^"]+)" (?P<status>\d+) (?P<size>\S+) "(?P<referrer>[^"]+)" "(?P<user_agent>[^"]+)"'
)

def parse_log_line(line):
    """Parses a single log line and returns a dictionary of fields."""
    match = LOG_PATTERN.match(line)
    if match:
        return match.groupdict()
    return None

def parse_log_file(file_path):
    """Reads the log file and returns a list of parsed entries."""
    data = []
    if not os.path.exists(file_path):
        # Fallback to access_log.processed if the specific one isn't found, or just error out
        if os.path.exists('access_log.processed'):
            file_path = 'access_log.processed'
            print(f"File {LOG_FILE} not found, using {file_path} instead.")
        else:
            print(f"Error: File {file_path} not found.")
            return []

    print(f"Reading {file_path}...")
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parsed = parse_log_line(line)
            if parsed:
                data.append(parsed)
    return data

def main():
    # 1. Load and Parse
    parsed_data = parse_log_file(LOG_FILE)
    
    if not parsed_data:
        print("No data parsed. Exiting.")
        return

    # 2. Convert to DataFrame
    df = pd.DataFrame(parsed_data)
    
    # Rename columns to match requirements if needed (regex groups already match mostly)
    # Requirements: ["ip","timestamp","method","path","protocol","status","bytes_sent","referrer","user_agent"]
    # Regex groups: ip, datetime, method, path, protocol, status, size, referrer, user_agent
    
    df.rename(columns={
        'datetime': 'timestamp',
        'size': 'bytes_sent'
    }, inplace=True)
    
    # Convert timestamp
    # Apache format: 25/Nov/2025:06:31:10 +1100
    # We need to handle the timezone. pandas to_datetime with format should work.
    print("Converting timestamps...")
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d/%b/%Y:%H:%M:%S %z')
    except Exception as e:
        print(f"Warning: Timestamp conversion encountered errors: {e}")
        # Fallback or partial conversion if needed, but let's try standard first

    # Convert numeric columns
    df['status'] = pd.to_numeric(df['status'], errors='coerce')
    df['bytes_sent'] = pd.to_numeric(df['bytes_sent'], errors='coerce').fillna(0).astype(int)

    print(f"Parsed {len(df)} records.")

    # 3. Bot Detection Heuristics
    print("\n--- Bot Detection Report ---")

    # A) Suspicious user agents
    suspicious_keywords = ["bot", "crawl", "spider", "python", "curl", "monitoring", "googlebot"]
    # Create a regex pattern for case-insensitive search
    ua_pattern = '|'.join(suspicious_keywords)
    suspicious_ua_df = df[df['user_agent'].str.contains(ua_pattern, case=False, na=False)]
    
    # B) High-frequency IPs
    top_ips = df['ip'].value_counts().head(20)

    # C) Repeated path scraping
    # Group by IP and path, count, then sort descending
    repeated_paths = df.groupby(['ip', 'path']).size().reset_index(name='count')
    top_repeated_paths = repeated_paths.sort_values('count', ascending=False).head(20)

    # D) Suspicious status codes (Optional but good to have)
    suspicious_status = df[df['status'].isin([401, 403, 404])]
    top_suspicious_status_ips = suspicious_status['ip'].value_counts().head(10)

    # 4. Show Results
    print("\nTop 20 IPs by Request Count:")
    print(top_ips)

    print(f"\nFound {len(suspicious_ua_df)} requests with suspicious User-Agents.")
    print("Sample of Suspicious User-Agents:")
    print(suspicious_ua_df['user_agent'].unique()[:10])

    print("\nTop 20 Repeated (IP, Path) Combinations:")
    print(top_repeated_paths)
    
    print("\nTop IPs with 401/403/404 Errors:")
    print(top_suspicious_status_ips)

    # 5. Save Data
    print(f"\nSaving parsed data to {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE, index=False)
    print("Done.")

if __name__ == "__main__":
    main()

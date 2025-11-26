# Bot Detection & Monitoring System

This project analyzes Apache access logs to detect potential bot activity using both heuristic rules and Machine Learning (Isolation Forest). It includes a Streamlit dashboard for visualizing the results.

## Features

- **Log Parsing**: Parses raw Apache access logs into a structured CSV format.
- **Heuristic Detection**: Identifies bots based on User-Agent strings, high request rates, and repeated path access.
- **Anomaly Detection**: Uses an Isolation Forest model to detect anomalous IP behaviors (outliers).
- **Interactive Dashboard**: A Streamlit app to explore traffic patterns, suspicious IPs, and User-Agent statistics.

## Project Structure

- `log_parser.py`: Script to parse the raw log file (`access_ssl_log.processed`) and save it to `parsed_access_log.csv`.
- `anomaly_detection.py`: Script to train the Isolation Forest model and generate anomaly scores (`diecast_ip_anomaly_scores.csv`).
- `streamlit_app.py`: The dashboard application.
- `requirements.txt`: Python dependencies.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/YOUR_USERNAME/bot-detection-using-adversial-machine-learning.git
    cd bot-detection-using-adversial-machine-learning
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Parse Logs
Run the parser to convert your raw log file into a CSV:
```bash
python log_parser.py
```
*Note: Ensure your log file is named `access_ssl_log.processed` or `access_log.processed` in the project directory.*

### 2. Detect Anomalies
Run the anomaly detection script to score IPs:
```bash
python anomaly_detection.py
```

### 3. Run Dashboard
Launch the Streamlit app to view the results:
```bash
streamlit run streamlit_app.py
```

## Hosting

To share this app with others, you can deploy it for free on **Streamlit Community Cloud**:
1.  Push this code to a GitHub repository.
2.  Sign up at [share.streamlit.io](https://share.streamlit.io/).
3.  Connect your GitHub account and select this repository.
4.  Click "Deploy"!

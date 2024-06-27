import os
import re
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import numpy as np

# Function to parse Darshan log file
def parse_darshan_log(file_path):
    records = []
    
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('# start_time:'):
                start_time = int(line.split()[-1])
            if line.startswith('# end_time:'):
                end_time = int(line.split()[-1])
            if line.startswith('#') or not line.strip():
                continue
            
            parts = re.split(r'\s+', line)
            if parts[0] == 'POSIX':
                try:
                    counter_name = parts[3]
                    counter_value = int(parts[4])
                    
                    if 'READ' in counter_name:
                        operation = 'read'
                    elif 'WRITE' in counter_name:
                        operation = 'write'
                    else:
                        continue
                    
                    records.append((start_time, end_time, operation, counter_value))
                except (ValueError, IndexError):
                    continue
    return records

# Function to calculate intervals
def calculate_intervals(start_time, end_time, interval_size=300):
    intervals = []
    start_interval = start_time // interval_size * interval_size
    end_interval = end_time // interval_size * interval_size
    for i in range(start_interval, end_interval + interval_size, interval_size):
        intervals.append(i)
    return intervals

# Function to distribute I/O size evenly across intervals
def evenly_distribute_io(records, interval_size=300):
    interval_io = {}
    for start_time, end_time, operation, size in records:
        intervals = calculate_intervals(start_time, end_time, interval_size)
        io_per_interval = size / len(intervals)
        for interval in intervals:
            if interval not in interval_io:
                interval_io[interval] = {'read': 0, 'write': 0}
            interval_io[interval][operation] += io_per_interval
    return interval_io

# Convert interval I/O to DataFrame
def interval_io_to_df(interval_io):
    records = []
    for interval, ops in interval_io.items():
        timestamp = datetime.fromtimestamp(interval)
        records.append((timestamp, ops['read'], ops['write']))
    return pd.DataFrame(records, columns=['timestamp', 'read', 'write'])

# Function to group I/O operations into 5-minute intervals
def group_io_operations(df):
    df['time_bin'] = df['timestamp'].dt.floor('5min')
    grouped = df.groupby(['time_bin'])[['read', 'write']].sum().fillna(0)
    return grouped

# Function to analyze I/O patterns
def analyze_io_patterns(grouped):
    hourly = grouped.resample('h').sum()
    daily = grouped.resample('D').sum()
    monthly = grouped.resample('ME').sum()
    return hourly, daily, monthly

# Function to plot I/O patterns
def plot_io_patterns(hourly, daily, monthly):
    fig, axes = plt.subplots(3, 1, figsize=(15, 15))
    
    for df, ax, title in zip([hourly, daily, monthly],
                             [axes[0], axes[1], axes[2]],
                             ['Hourly I/O Rates', 'Daily I/O Rates', 'Monthly I/O Rates']):
        if not df.empty:
            df.plot(ax=ax, title=title)
            ax.set_ylabel('I/O Size (bytes)')
            ax.set_xlabel('Time')
            if ax.get_xlim()[0] == ax.get_xlim()[1]:
                ax.set_xlim(left=ax.get_xlim()[0] - 1, right=ax.get_xlim()[1] + 1)
    
    plt.tight_layout()
    plt.show()

# Function to detect I/O bursts using adaptive thresholds
def detect_io_bursts(grouped, initial_threshold=1.5, adapt_rate=0.1):
    stats = grouped.describe()
    mean_read = stats['read']['mean']
    std_read = stats['read']['std']
    mean_write = stats['write']['mean']
    std_write = stats['write']['std']
    
    burst_threshold_read = mean_read + initial_threshold * std_read
    burst_threshold_write = mean_write + initial_threshold * std_write
    
    bursts_read = []
    bursts_write = []
    
    for index, row in grouped.iterrows():
        read_burst = row['read'] > burst_threshold_read
        write_burst = row['write'] > burst_threshold_write
        
        bursts_read.append(read_burst)
        bursts_write.append(write_burst)
        
        if read_burst:
            burst_threshold_read += adapt_rate * std_read
        else:
            burst_threshold_read -= adapt_rate * std_read
            
        if write_burst:
            burst_threshold_write += adapt_rate * std_write
        else:
            burst_threshold_write -= adapt_rate * std_write
        
        burst_threshold_read = max(burst_threshold_read, mean_read + initial_threshold * std_read)
        burst_threshold_write = max(burst_threshold_write, mean_write + initial_threshold * std_write)
    
    return pd.Series(bursts_read, index=grouped.index), pd.Series(bursts_write, index=grouped.index)

# Function to process all logs in a directory
def process_all_logs(directory):
    all_data = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            records = parse_darshan_log(file_path)
            interval_io = evenly_distribute_io(records)
            df = interval_io_to_df(interval_io)
            if not df.empty:
                all_data.append(df)
    
    if all_data:
        combined_df = pd.concat(all_data)
        grouped = group_io_operations(combined_df)
        return grouped
    else:
        return pd.DataFrame()

# Function to generate features for machine learning model
def generate_advanced_features(grouped, interval=18):
    X = []
    y_burst_read = []
    y_burst_write = []
    timestamps = []
    
    burst_threshold_read = grouped['read'].mean() + 1.5 * grouped['read'].std()
    burst_threshold_write = grouped['write'].mean() + 1.5 * grouped['write'].std()
    
    for i in range(5, len(grouped) - interval):
        window = grouped.iloc[i-5:i]
        
        features = [
            window['read'].mean(), window['read'].std(), window['write'].mean(), window['write'].std(),
            window['read'].max(), window['read'].min(), window['write'].max(), window['write'].min(),
            window['read'].median(), window['write'].median(),
            window['read'].diff().mean(), window['write'].diff().mean()
        ]
        X.append(features)
        timestamps.append(grouped.index[i])
        
        y_burst_read.append(grouped.iloc[i+interval]['read'] > burst_threshold_read)
        y_burst_write.append(grouped.iloc[i+interval]['write'] > burst_threshold_write)
    
    return pd.DataFrame(X), pd.Series(y_burst_read), pd.Series(y_burst_write), timestamps

# Function to train and evaluate XGBoost model for classification
def train_evaluate_xgboost_classifier(X, y):
    if len(y) < 2:
        return None, 0  # Not enough data to split
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if len(y_train) == 0 or len(y_test) == 0:
        return None, 0  # Handle the case where there is insufficient data
    
    model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return model, accuracy

# Main script
if __name__ == "__main__":
    directory = '/uhome/a01431/Sample/'
    grouped = process_all_logs(directory)
    if not grouped.empty:
        hourly, daily, monthly = analyze_io_patterns(grouped)
        plot_io_patterns(hourly, daily, monthly)
        bursts_read, bursts_write = detect_io_bursts(grouped)
        print("I/O bursts detected in read operations:", bursts_read.sum())
        print("I/O bursts detected in write operations:", bursts_write.sum())
        bursts_df = grouped.copy()
        bursts_df['burst_read'] = bursts_read
        bursts_df['burst_write'] = bursts_write
        bursts_df.to_csv('io_bursts_detected.csv')
        
        interval = 18  # Predicting 90 minutes later (18 intervals of 5 minutes each)
        highest_accuracy_read = 0
        highest_accuracy_write = 0
        lowest_accuracy_read = float('inf')
        lowest_accuracy_write = float('inf')
        total_accuracy_read = 0
        total_accuracy_write = 0
        count = 0

        highest_accuracy_read_time = None
        lowest_accuracy_read_time = None
        highest_accuracy_write_time = None
        lowest_accuracy_write_time = None

        accuracies_read = []
        accuracies_write = []
        
        for i in range(0, len(grouped) - interval, 5):
            X, y_burst_read, y_burst_write, timestamps = generate_advanced_features(grouped.iloc[i:], interval)
            
            if len(y_burst_read) == 0 or len(y_burst_write) == 0:
                print(f"No data to process for interval {interval * 5} minutes at step {i}")
                continue
            
            # Train and evaluate burst prediction model for read operations
            _, accuracy_burst_read = train_evaluate_xgboost_classifier(X, y_burst_read)
            if accuracy_burst_read > 0:
                highest_accuracy_read = max(highest_accuracy_read, accuracy_burst_read)
                lowest_accuracy_read = min(lowest_accuracy_read, accuracy_burst_read)
                total_accuracy_read += accuracy_burst_read
                accuracies_read.append(accuracy_burst_read)
                if accuracy_burst_read == highest_accuracy_read:
                    highest_accuracy_read_time = timestamps[0]
                if accuracy_burst_read == lowest_accuracy_read:
                    lowest_accuracy_read_time = timestamps[0]
            
            # Train and evaluate burst prediction model for write operations
            _, accuracy_burst_write = train_evaluate_xgboost_classifier(X, y_burst_write)
            if accuracy_burst_write > 0:
                highest_accuracy_write = max(highest_accuracy_write, accuracy_burst_write)
                lowest_accuracy_write = min(lowest_accuracy_write, accuracy_burst_write)
                total_accuracy_write += accuracy_burst_write
                accuracies_write.append(accuracy_burst_write)
                if accuracy_burst_write == highest_accuracy_write:
                    highest_accuracy_write_time = timestamps[0]
                if accuracy_burst_write == lowest_accuracy_write:
                    lowest_accuracy_write_time = timestamps[0]
            
            count += 1
        
        if count > 0:
            average_accuracy_read = total_accuracy_read / count
            average_accuracy_write = total_accuracy_write / count

            print(f"Highest read burst prediction accuracy: {highest_accuracy_read} at {highest_accuracy_read_time}")
            print(f"Lowest read burst prediction accuracy: {lowest_accuracy_read} at {lowest_accuracy_read_time}")
            print(f"Average read burst prediction accuracy: {average_accuracy_read}")

            print(f"Highest write burst prediction accuracy: {highest_accuracy_write} at {highest_accuracy_write_time}")
            print(f"Lowest write burst prediction accuracy: {lowest_accuracy_write} at {lowest_accuracy_write_time}")
            print(f"Average write burst prediction accuracy: {average_accuracy_write}")
        else:
            print("Insufficient data to compute accuracies.")
    else:
        print("No data to process.")

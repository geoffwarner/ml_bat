# src/data_preparation.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ta
import os
import sys

# Configure plot aesthetics
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 7)

def load_data(file_path):
    """
    Load stock data from a CSV file.

    Parameters:
    - file_path (str): Absolute path to the CSV file.

    Returns:
    - df (pd.DataFrame): Loaded data as a pandas DataFrame.
    """
    print(f"Loading data from {file_path}...")
    if not os.path.exists(file_path):
        print(f"Error: File not found at path: {file_path}")
        sys.exit(1)
    try:
        df = pd.read_csv(file_path)
        print("Data loaded successfully.")
        print("\nColumns in the loaded DataFrame:")
        print(df.columns.tolist())  # Print column names
        print("\nFirst 5 rows of the DataFrame:")
        print(df.head())  # Print first 5 rows
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def preprocess_data(df):
    """
    Preprocess the stock data.
    
    Steps:
    - Standardize column names (strip spaces and title case).
    - Rename 'Time' to 'Date' if necessary.
    - Convert 'Date' to datetime.
    - Set 'Date' as index and sort.
    - Handle missing values.
    """
    print("Preprocessing data...")
    
    # Standardize column names: strip spaces and capitalize first letter
    df.columns = df.columns.str.strip().str.title()
    print("\nColumns after standardization:")
    print(df.columns.tolist())
    
    # Rename 'Time' to 'Date' if 'Time' exists
    if 'Time' in df.columns:
        df.rename(columns={'Time': 'Date'}, inplace=True)
        print("\nRenamed 'Time' column to 'Date'.")
    else:
        print("\nError: 'Time' column not found. Please verify the CSV structure.")
        sys.exit(1)
    
    # Define required columns (excluding 'Date' since it's set as index)
    required_columns = {'Open', 'High', 'Low', 'Close', 'Volume'}
    
    # Check for missing columns
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        print(f"Error: Missing columns in data: {missing}")
        sys.exit(1)
    
    # Convert 'Date' to datetime
    try:
        df['Date'] = pd.to_datetime(df['Date'])
    except Exception as e:
        print(f"Error converting 'Date' to datetime: {e}")
        sys.exit(1)
    
    # Set 'Date' as index and sort
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    
    # Check for missing values
    missing_values = df.isnull().sum()
    print("\nMissing Values Before Cleaning:")
    print(missing_values)
    
    # Drop rows with missing values
    df.dropna(inplace=True)
    
    # Verify no missing values remain
    missing_values_after = df.isnull().sum()
    print("\nMissing Values After Cleaning:")
    print(missing_values_after)
    
    print("Data preprocessing completed.")
    return df

def perform_eda(df, output_dir):
    """
    Perform Exploratory Data Analysis on the stock data.

    Parameters:
    - df (pd.DataFrame): Cleaned stock data.
    - output_dir (str): Absolute path to the outputs directory.
    """
    print("Performing Exploratory Data Analysis (EDA)...")
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving plots to {output_dir}/")

    # Plot Closing Price Over Time
    plt.figure()
    plt.plot(df['Close'], label='Close Price', color='blue')
    plt.title('Closing Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price (R)')
    plt.legend()
    plt.tight_layout()
    closing_price_plot = os.path.join(output_dir, 'closing_price_over_time.png')
    plt.savefig(closing_price_plot)
    plt.close()
    print(f"Saved: {closing_price_plot}")

    # Plot Trading Volume Over Time (Using Line Plot)
    plt.figure()
    plt.plot(df.index, df['Volume'], label='Volume', color='orange')
    plt.title('Trading Volume Over Time')
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.legend()
    plt.tight_layout()
    trading_volume_plot = os.path.join(output_dir, 'trading_volume_over_time.png')
    plt.savefig(trading_volume_plot)
    plt.close()
    print(f"Saved: {trading_volume_plot}")

    # Calculate and Plot Simple Moving Averages
    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)

    plt.figure()
    plt.plot(df['Close'], label='Close Price', color='blue')
    plt.plot(df['SMA_20'], label='20-Day SMA', color='red')
    plt.plot(df['SMA_50'], label='50-Day SMA', color='green')
    plt.title('Closing Price with 20 & 50-Day Simple Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price (R)')
    plt.legend()
    plt.tight_layout()
    closing_sma_plot = os.path.join(output_dir, 'closing_price_with_sma.png')
    plt.savefig(closing_sma_plot)
    plt.close()
    print(f"Saved: {closing_sma_plot}")

    # Calculate RSI
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)

    # Plot RSI
    plt.figure()
    plt.plot(df['RSI'], label='RSI', color='purple')
    plt.title('Relative Strength Index (RSI)')
    plt.xlabel('Date')
    plt.ylabel('RSI')
    plt.legend()
    plt.tight_layout()
    rsi_plot = os.path.join(output_dir, 'rsi.png')
    plt.savefig(rsi_plot)
    plt.close()
    print(f"Saved: {rsi_plot}")

    # Highlight overbought and oversold regions
    plt.figure()
    plt.plot(df['RSI'], label='RSI', color='purple')
    plt.axhline(70, color='red', linestyle='--', label='Overbought')
    plt.axhline(30, color='green', linestyle='--', label='Oversold')
    plt.title('Relative Strength Index (RSI) with Overbought/Oversold Levels')
    plt.xlabel('Date')
    plt.ylabel('RSI')
    plt.legend()
    plt.tight_layout()
    rsi_levels_plot = os.path.join(output_dir, 'rsi_with_levels.png')
    plt.savefig(rsi_levels_plot)
    plt.close()
    print(f"Saved: {rsi_levels_plot}")

    # Calculate MACD
    df['MACD'] = ta.trend.macd_diff(df['Close'])

    # Plot MACD
    plt.figure()
    plt.plot(df['MACD'], label='MACD Diff', color='brown')
    plt.title('Moving Average Convergence Divergence (MACD)')
    plt.xlabel('Date')
    plt.ylabel('MACD Diff')
    plt.legend()
    plt.tight_layout()
    macd_plot = os.path.join(output_dir, 'macd.png')
    plt.savefig(macd_plot)
    plt.close()
    print(f"Saved: {macd_plot}")

    print("EDA completed successfully.")

def main():
    # Get the absolute path of the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define the absolute path to the data directory
    data_dir = os.path.join(script_dir, '..', 'data')
    
    # Define the absolute path to the outputs directory
    output_dir = os.path.join(script_dir, '..', 'outputs')
    
    # Ensure the data directory exists
    if not os.path.exists(data_dir):
        print(f"Error: The data directory does not exist at {data_dir}")
        sys.exit(1)
    
    # Define the path to your CSV file
    file_path = os.path.join(data_dir, 'bat_daily_data_5000_22-Nov-2024.csv')
    
    # Load data
    df = load_data(file_path)
    
    # Preprocess data
    df = preprocess_data(df)
    
    # Perform EDA
    perform_eda(df, output_dir)
    
    # Define the path to save the processed data
    processed_data_path = os.path.join(data_dir, 'preprocessed_data.csv')
    
    # Save the cleaned data
    try:
        df.to_csv(processed_data_path)
        print(f"\nCleaned data saved to '{processed_data_path}'.")
    except Exception as e:
        print(f"Error saving processed data: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()






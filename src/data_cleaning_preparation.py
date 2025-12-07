import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import logging
import warnings
warnings.filterwarnings('ignore')

class FinancialDataCleaner:
    """
    Comprehensive data cleaning and preparation system for financial data
    """
    
    def __init__(self, data_dir="stock_data", output_dir="cleaned_data"):
        """
        Initialize the Financial Data Cleaner
        
        Args:
            data_dir (str): Directory containing raw CSV files
            output_dir (str): Directory to save cleaned data
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.setup_logging()
        self.create_output_directory()
        self.cleaning_report = {}
        
    def setup_logging(self):
        """Setup comprehensive logging for data cleaning process"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('data_cleaning.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def create_output_directory(self):
        """Create output directory for cleaned data"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            self.logger.info(f"Created output directory: {self.output_dir}")
    
    def load_raw_data(self, filename):
        """
        Load raw data from CSV file with error handling
        
        Args:
            filename (str): Name of the CSV file
            
        Returns:
            pandas.DataFrame: Loaded data or None if error
        """
        try:
            filepath = os.path.join(self.data_dir, filename)
            data = pd.read_csv(filepath)
            self.logger.info(f"Loaded {len(data)} records from {filename}")
            return data
        except Exception as e:
            self.logger.error(f"Error loading {filename}: {e}")
            return None

    
    def handle_missing_values(self, data, strategy='smart'):
        """
        Handle missing values using specified strategy
        
        Args:
            data (pandas.DataFrame): Data with potential missing values
            strategy (str): 'remove', 'interpolate', 'forward_fill', 'smart'
            
        Returns:
            pandas.DataFrame: Data with missing values handled
        """
        self.logger.info(f"Handling missing values using '{strategy}' strategy")
        
        original_count = len(data)
        data_copy = data.copy()
        
        if strategy == 'remove':
            # Remove rows with any missing values
            data_copy = data_copy.dropna()
            removed = original_count - len(data_copy)
            self.logger.info(f"Removed {removed} rows with missing values")
            
        elif strategy == 'interpolate':
            # Interpolate missing price values
            price_columns = ['Open', 'High', 'Low', 'Close']
            for col in price_columns:
                if col in data_copy.columns:
                    missing_before = data_copy[col].isnull().sum()
                    if missing_before > 0:
                        data_copy[col] = data_copy[col].interpolate(method='linear')
                        # Interpolation sometimes cannot fill all missing values
                        missing_after = data_copy[col].isnull().sum()
                        filled = missing_before - missing_after
                        self.logger.info(f"Interpolated {filled} missing values in {col}")
            
            # Handle Volume separately
            if 'Volume' in data_copy.columns:
                volume_missing = data_copy['Volume'].isnull().sum()
                if volume_missing > 0:
                    data_copy['Volume'] = data_copy['Volume'].fillna(0)
                    self.logger.info(f"Filled {volume_missing} missing Volume values with 0")
                    
        elif strategy == 'forward_fill':
            # Forward fill missing values
            data_copy = data_copy.fillna(method='ffill')
            self.logger.info("Applied forward fill to missing values")
            
        elif strategy == 'smart':
            # Smart combination strategy
            # Remove rows where all price data is missing
            price_columns = ['Open', 'High', 'Low', 'Close']
            if all(col in data_copy.columns for col in price_columns):
                all_prices_missing = data_copy[price_columns].isnull().all(axis=1)
                rows_removed = all_prices_missing.sum()
                if rows_removed > 0:
                    data_copy = data_copy[~all_prices_missing]
                    self.logger.info(f"Removed {rows_removed} rows with all missing price data")
            
            # Interpolate remaining missing price values
            for col in price_columns:
                if col in data_copy.columns:
                    missing_count = data_copy[col].isnull().sum()
                    if missing_count > 0:
                        data_copy[col] = data_copy[col].interpolate(method='linear')
                        # Since linear interpolation cannot fill all the missing values sometimes
                        # we employ forward fill and backward fill to handle the remaining gaps:
                        data_copy[col] = data_copy[col].fillna(method='ffill')
                        data_copy[col] = data_copy[col].fillna(method='bfill')
                        self.logger.info(f"Smart filled {missing_count} missing values in {col}")
            
            # Handle Volume with median
            if 'Volume' in data_copy.columns:
                volume_missing = data_copy['Volume'].isnull().sum()
                if volume_missing > 0:
                    median_volume = data_copy['Volume'].median()
                    data_copy['Volume'] = data_copy['Volume'].fillna(median_volume)
                    self.logger.info(f"Filled {volume_missing} Volume values with median")
        
        return data_copy
    
    def remove_duplicates(self, data):
        """
        Remove duplicate records based on Date
        
        Args:
            data (pandas.DataFrame): Data with potential duplicates
            
        Returns:
            pandas.DataFrame: Data with duplicates removed
        """
        self.logger.info("Checking for duplicate records")
        
        original_count = len(data)
        
        if 'Date' in data.columns:
            # Remove duplicates based on Date column
            data_clean = data.drop_duplicates(subset=['Date'], keep='first')
            duplicates_removed = original_count - len(data_clean)
            
            if duplicates_removed > 0:
                self.logger.info(f"Removed {duplicates_removed} duplicate records")
            else:
                self.logger.info("No duplicate records found")
        else:
            # Remove exact duplicates
            data_clean = data.drop_duplicates()
            duplicates_removed = original_count - len(data_clean)
            self.logger.info(f"Removed {duplicates_removed} exact duplicate rows")
        
        return data_clean
    
    
    def ensure_data_consistency(self, data):
        """
        Ensure data consistency across the dataset
        
        Args:
            data (pandas.DataFrame): Data to check consistency
            
        Returns:
            pandas.DataFrame: Consistent data
        """
        self.logger.info("Ensuring data consistency")
        
        data_copy = data.copy()
        issues_fixed = 0
        
        # Fix OHLC relationships
        price_columns = ['Open', 'High', 'Low', 'Close']
        if all(col in data_copy.columns for col in price_columns):
            # Ensure High is the maximum of OHLC
            data_copy['High'] = data_copy[['Open', 'High', 'Low', 'Close']].max(axis=1)
            
            # Ensure Low is the minimum of OHLC
            data_copy['Low'] = data_copy[['Open', 'High', 'Low', 'Close']].min(axis=1)
            
            # Count and log fixes
            invalid_high = (data_copy['High'] < data_copy[['Open', 'Low', 'Close']].max(axis=1)).sum()
            invalid_low = (data_copy['Low'] > data_copy[['Open', 'High', 'Close']].min(axis=1)).sum()
            
            if invalid_high > 0 or invalid_low > 0:
                issues_fixed = invalid_high + invalid_low
                self.logger.info(f"Fixed {issues_fixed} OHLC consistency issues")
        
        # Ensure Volume is non-negative
        if 'Volume' in data_copy.columns:
            negative_volume = (data_copy['Volume'] < 0).sum()
            if negative_volume > 0:
                data_copy['Volume'] = data_copy['Volume'].abs()
                issues_fixed += negative_volume
                self.logger.info(f"Fixed {negative_volume} negative volume values")
        
        return data_copy
    
    def convert_data_types(self, data):
        """
        Convert data types as needed for analysis
        
        Args:
            data (pandas.DataFrame): Data with potentially incorrect types
            
        Returns:
            pandas.DataFrame: Data with correct types
        """
        self.logger.info("Converting data types")
        
        data_copy = data.copy()
        
        # Convert Date column to datetime
        if 'Date' in data_copy.columns:
            try:
                # Convert Series to string first, then clean
                date_strings = data['Date'].astype(str)
                cleaned_strings = date_strings.str.replace(r'[-+]\d{2}:\d{2}$', '', regex=True)
                data_copy['Date'] = pd.to_datetime(cleaned_strings, errors='coerce')
                self.logger.info("Converted Date column to datetime")
            except Exception as e:
                self.logger.warning(f"Could not convert Date column: {e}")
        
        # Ensure price columns are numeric
        price_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for column in price_columns:
            if column in data_copy.columns:
                try:
                    data_copy[column] = pd.to_numeric(data_copy[column], errors='coerce')
                    self.logger.info(f"Converted {column} to numeric")
                except Exception as e:
                    self.logger.warning(f"Could not convert {column} to numeric: {e}")
        
        # Ensure Symbol is string
        if 'Symbol' in data_copy.columns:
            data_copy['Symbol'] = data_copy['Symbol'].astype(str)
            self.logger.info("Ensured Symbol column is string type")
        
        return data_copy
    
    
    def clean_single_stock(self, filename):
        """
        Clean data for a single stock
        
        Args:
            filename (str): Name of the CSV file
            normalization_method (str): Normalization method to use
            
        Returns:
            dict: Cleaning results
        """
        symbol = filename.split('_')[0] if '_' in filename else filename.replace('.csv', '')
        
        self.logger.info(f"Starting data cleaning for {symbol}")
        
        # Load raw data
        raw_data = self.load_raw_data(filename)
                
        # Step-by-step cleaning
        cleaned_data = raw_data.copy()
        
        # 1. Convert data types
        cleaned_data = self.convert_data_types(cleaned_data)
        
        # 2. Handle missing values
        cleaned_data = self.handle_missing_values(cleaned_data, strategy='smart')
        
        # 3. Remove duplicates
        cleaned_data = self.remove_duplicates(cleaned_data)
        
        # 4. Ensure data consistency
        cleaned_data = self.ensure_data_consistency(cleaned_data)
        
        # Save cleaned data
        output_filename = f"{symbol}_cleaned.csv"
        output_path = os.path.join(self.output_dir, output_filename)
        cleaned_data.to_csv(output_path, index=False)
        
        self.logger.info(f"Cleaning complete for {symbol}")
        self.logger.info(f"Saved cleaned data: {output_path}")
        
            
    def clean_all_stock_data(self):
        """
        Clean all stock data files in the data directory
        
        Args:
            normalization_method (str): Normalization method to use
            
        Returns:
            dict: Summary of all cleaning results
        """
        self.logger.info("Starting bulk data cleaning process")
        
        # Find all CSV files
        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        
        if not csv_files:
            self.logger.error("No CSV files found in data directory")
            return None
        
        all_reports = []
        successful_cleanings = 0
        failed_cleanings = 0
        
        for filename in csv_files:
            try:
                report = self.clean_single_stock(filename)
                if report:
                    all_reports.append(report)
                    successful_cleanings += 1
                else:
                    failed_cleanings += 1
            except Exception as e:
                self.logger.error(f" {filename}: {e}")
                failed_cleanings += 1
        
        # Generate summary report
        summary = {
            'total_files_processed': len(csv_files),
            'successful_cleanings': successful_cleanings,
            'failed_cleanings': failed_cleanings,
        }
        
        # Save summary report
        summary_path = os.path.join(self.output_dir, 'cleaning_summary.json')
        import json
        with open(summary_path, 'w') as f:
            summary_json = summary.copy()
            json.dump(summary_json, f, indent=2, default=str)
        
        self.logger.info(f"Bulk cleaning complete!")
        self.logger.info(f"Summary saved: {summary_path}")
        
        return summary

def main():
    """Main function to demonstrate data cleaning capabilities"""
    
    print("FINANCIAL DATA CLEANING AND PREPARATION")
    print("=" * 50)
    
    # Initialize cleaner
    cleaner = FinancialDataCleaner(
        data_dir="stock_data",
        output_dir="cleaned_data"
    )
    
    # Clean all stock data
    cleaner.clean_all_stock_data()



if __name__ == "__main__":
    main()
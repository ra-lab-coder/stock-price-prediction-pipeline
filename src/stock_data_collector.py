import yfinance as yf
import pandas as pd
import os
import time
import logging
import random

class CustomDateStockCollector:
    def __init__(self, data_dir="stock_data"):
        """
        Initialize the Custom Date Stock Data Collector
        
        Args:
            data_dir (str): Directory to save CSV files
        """
        self.data_dir = data_dir
        self.setup_logging()
        self.create_data_directory()
        
    def setup_logging(self):
        """Setup logging for the data collection process"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('custom_date_collection.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def create_data_directory(self):
        """Create directory structure for organized data storage"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            self.logger.info(f"Created data directory: {self.data_dir}")
            
    
    def collect_stock_data_custom_dates(self, symbol, start_date, end_date, interval="1d"):
        self.logger.info(f"Collecting data for {symbol} from {start_date} to {end_date}")
        
        # Create ticker object
        ticker = yf.Ticker(symbol)
        
        # Get historical data for custom date range
        data = ticker.history(
            start=start_date,
            end=end_date,
            auto_adjust=False
        )
        
        if data.empty:
            self.logger.warning(f"No data found for {symbol} in the specified date range")
            return None
            
        # Reset index to make Date a column
        data.reset_index(inplace=True)
        
        # Add symbol column
        data['Symbol'] = symbol
        
        # Reorder columns for better organization
        columns_order = ['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
        data = data[columns_order]
        
        self.logger.info(f"Successfully collected {len(data)} records for {symbol}")
        return data
            
                
    def collect_multiple_stocks_custom_dates(self, symbols, start_date, end_date, delay=5):
        """
        Collect data for multiple stocks with custom date ranges
        
        Args:
            symbols (list): List of stock symbols
            start_date (str): Start date
            end_date (str): End date
            delay (int): Delay between stocks
            
        Returns:
            list: Quality reports for all symbols
        """
        quality_reports = []
        total_symbols = len(symbols)
        
        self.logger.info(f"Starting collection for {total_symbols} symbols from {start_date} to {end_date}")
        
        for i, symbol in enumerate(symbols):
            self.logger.info(f"Processing {symbol} ({i+1}/{total_symbols})")
            
            try:
                # Smart collection based on date range
                data = self.collect_stock_data_custom_dates(symbol, start_date, end_date)
                
                # Verify and save
                quality_report = self.verify_data_quality(data, symbol)
                quality_reports.append(quality_report)
                self.save_to_csv(data, symbol, start_date, end_date)
                
                # Delay between stocks
                if i < total_symbols - 1:
                    time.sleep(delay)
                    
            except Exception as e:
                self.logger.error(f"Error processing {symbol}: {e}")
                quality_reports.append({
                    "symbol": symbol,
                    "status": "Error",
                    "issues": [str(e)]
                })
        
        return quality_reports
    
    def verify_data_quality(self, data, symbol):
        """Verify data quality"""
        if data is None or data.empty:
            return {"symbol": symbol, "status": "No data", "issues": ["No data available"]}
            
        issues = []
        
        missing_data = data.isnull().sum()
        if missing_data.any():
            issues.append(f"Missing values: {missing_data[missing_data > 0].to_dict()}")
            
        price_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']
        negative_prices = data[price_columns] < 0
        if negative_prices.any().any():
            issues.append("Negative prices found")
            
        status = "Good" if not issues else "Issues found"
        
        return {
            "symbol": symbol,
            "records": len(data),
            "date_range": f"{data['Date'].min()} to {data['Date'].max()}",
            "status": status,
            "issues": issues
        }
    
    def save_to_csv(self, data, symbol, start_date, end_date):
        """Save data to CSV with custom date range in filename"""
        if data is None or data.empty:
            self.logger.warning(f"No data to save for {symbol}")
            return
            
        
        filename = f"{symbol}_data_{start_date}_to_{end_date}.csv"
        filepath = os.path.join(self.data_dir, filename)
        
        try:
            data.to_csv(filepath, index=False)
            self.logger.info(f"Data saved to: {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving data for {symbol}: {e}")

# Usage examples
def main(start_date,end_date, symbols ):    
    collector = CustomDateStockCollector()
    reports = collector.collect_multiple_stocks_custom_dates(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        delay=7
    )
    print(reports)
    
if __name__ == "__main__":
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    start_date='2020-01-01'
    end_date='2025-11-30'
    main(start_date, end_date, symbols)
import pandas as pd
import numpy as np
import os
import logging
import src.utils as utils
from src.data_split import DataSplit
from src.data_prep import DataPrep
from src.train import TrainModel
from src.test import test
import warnings
warnings.filterwarnings('ignore')


class StockPredictionML(DataSplit, DataPrep, TrainModel):
    """
    Comprehensive Machine Learning system for stock price prediction
    """
    
    def __init__(self, data_dir="data", output_dir="ml_results"):
        """
        Initialize the ML system
        
        Args:
            data_dir (str): Directory containing cleaned data
            output_dir (str): Directory for ML results and models
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.setup_logging()
        self.create_output_directory()
        
        # ML components
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.target_column = 'Close'
        
        # Results storage
        self.model_results = {}
        self.best_model = None
        self.best_score = float('-inf')
        
    def setup_logging(self):
        """Setup logging for ML process"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('ml_stock_prediction.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def create_output_directory(self):
        """Create output directory for results"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            self.logger.info(f"Created output directory: {self.output_dir}")


def run(symbol='AAPL'):
    """
    Run complete ML pipeline for stock prediction
    
    Args:
        symbol (str): Stock symbol to analyze
    """
    print(f"Running complete ML pipeline for {symbol}")
    
    # Initialize ML system
    ml_system = StockPredictionML()
    
    # Step 1: Load cleaned data from Task 6
    print("Step 1: Loading cleaned data...")
    cleaned_filename = f"{symbol}_cleaned.csv"
    data = ml_system.load_cleaned_data(cleaned_filename)
    
    # Step 2: Split data
    print("\nStep 2: Splitting data...")
    train_df, val_df, test_df = ml_system.split_data(data, symbol)
    
    # Step 3: Feature Engineering
    print("\nStep 3: Engineering features...")
    train_with_features = ml_system.engineer_features(train_df)
    val_with_features = ml_system.engineer_features(val_df)
    test_with_features = ml_system.engineer_features(test_df)
    
    # Step 4: Prepare features and target
    print("\nStep 4: Preparing features and target...")
    X_train, y_train, _ = ml_system.prepare_features_and_target(train_with_features)
    X_val, y_val, _ = ml_system.prepare_features_and_target(val_with_features)
    X_test, y_test, _ = ml_system.prepare_features_and_target(test_with_features)
    
    # Step 5: Scale features
    print("\nStep 5: Scaling features...")
    X_train_scaled, X_val_scaled, X_test_scaled = ml_system.scale_features(
        X_train, X_val, X_test, method='standard'
    )
    
    # Step 6: Initialize and train models, so we can find the best performed model
    print("\nStep 6: Training multiple ML models...")
    ml_system.initialize_models()
    ml_system.train_and_evaluate_on_val(X_train, y_train, X_val, y_val)
    
    # Step 7: Store the model comparison results:
    result_path = os.path.join(ml_system.output_dir, "model_comparison.csv")
    utils.save_model_comparison_results(model_results=ml_system.model_results, output_path=result_path)
    
    # Step 8: Do hyperparameter tuning on the best performed model from step 6
    # but note we're doing the hyperparameter tuning on the original model of the 
    # best performed model, not the trained best model from step 6
    print("\nStep 7: Hyperparameter tuning for best models...")
    tuned_model = ml_system.perform_hyperparameter_tuning(X_train_scaled, y_train, ml_system.best_model)
    
    # Step 9: Retrain this tuned_model on train_set:
    tuned_model.fit(X_train_scaled, y_train)
    final_model = tuned_model
    ml_system.models['final_model'] = final_model
    
    # Step 10: Test the final prediction model
    y_test_pred, test_results = test(X_test_scaled, y_test, final_model, ml_system.best_model)
    print(test_results)
    
    tab_csv_path = os.path.join(ml_system.output_dir, f'ml_report_{symbol}.csv')
    utils.create_tableau_csv(test_df = test_with_features, y_test_pred=y_test_pred, ticker = symbol, csv_path = tab_csv_path)

    return ml_system


def run_multiple_stocks_ml(stocks_list):
    """
    Run ML pipeline for multiple stocks
    """    
    print("Running ML Pipeline for Multiple Stocks")
    print("=" * 45)
    
    for symbol in stocks_list:
        print(f"\n{symbol}:")
        print("-" * 20)
        
        run(symbol)
    

if __name__ == "__main__":
    stocks_list = ["AAPL", "GOOGL", "MSFT"]
    ml_system = run_multiple_stocks_ml(stocks_list)
   
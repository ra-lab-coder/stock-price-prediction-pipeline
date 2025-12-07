import os
import pandas as pd
import numpy as np

def create_model_comparison_report(self):
        """
        Create comprehensive model comparison report
        
        Returns:
            pandas.DataFrame: Model comparison results
        """
        self.logger.info("Creating model comparison report")
        
        comparison_data = []
        
        for model_name, results in self.model_results.items():
            comparison_data.append({
                'Model': model_name,
                'Train_R2': results['train_r2'],
                'Val_R2': results['val_r2'],
                'Train_RMSE': results['train_rmse'],
                'Val_RMSE': results['val_rmse'],
                'Test_RMSE': results['test_rmse'],
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Val_R2', ascending=False)
        
        # Save to CSV
        output_path = os.path.join(self.output_dir, 'model_comparison.csv')
        comparison_df.to_csv(output_path, index=False)
        
        self.logger.info(f"Model comparison saved to: {output_path}")
        return comparison_df
    

def save_predictions(y_test_pred, y_test, csv_path, index, index_name="Date"):
    df_results = pd.DataFrame({
        "Actual_Price": y_test,
        "Predicted_Price": y_test_pred,
    })
    
    if index is not None:
        df_results[index_name] = index
        cols = [index_name, "Actual_Price", "Predicted_Price"]
        df_results = df_results[cols]

    df_results.to_csv(csv_path, index=False)
    return df_results


def create_tableau_csv(test_df, y_test_pred, ticker, csv_path):
    # Ensure predictions are 1D numpy array
    y_pred = np.asarray(y_test_pred).ravel()
    
    # Align lengths: keep only the last len(y_pred) rows of test_df
    if len(test_df) > len(y_pred):
        test_df = test_df.iloc[-len(y_pred):]
    elif len(test_df) < len(y_pred):
        raise ValueError("y_test_pred is longer than test_df, check your splits.")
    
    # Ensure Date is in YYYY-MM-DD string format
    date_col = pd.to_datetime(test_df["Date"]).dt.strftime("%Y-%m-%d")
     
    # Get actual close price
    actual_close = test_df["Close"].values
    
    # Optional columns: Volume and Moving_Average_50
    volume = test_df["Volume"].values if "Volume" in test_df.columns else np.nan
    ma5 = (
        test_df["MA_5"].values
        if "MA_5" in test_df.columns
        else np.nan
    )
    ma20 = (
        test_df["MA_20"].values
        if "MA_20" in test_df.columns
        else np.nan
    )
    
    # Build final DataFrame
    df_out = pd.DataFrame({
        "Date": date_col,
        "Ticker": ticker,
        "Actual_Close": actual_close,
        "Predicted_Close": y_pred,
        "Prediction_Error": y_pred - actual_close,
        "Volume": volume,
        "Moving_Average_5": ma5,
        "Moving_Average_20": ma20
    })

    # Save to CSV
    df_out.to_csv(csv_path, index=False)
    return df_out
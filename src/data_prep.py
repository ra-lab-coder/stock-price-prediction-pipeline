import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class DataPrep:
    def engineer_features(self, data):
        """
        Create advanced features for ML models
        
        Args:
            data (pandas.DataFrame): Input data
            
        Returns:
            pandas.DataFrame: Data with engineered features
        """    
        data_copy = data.copy()
        
        # Price-based features
        if all(col in data_copy.columns for col in ['Open', 'High', 'Low', 'Close']):
            # Price ratios
            data_copy['High_Low_Ratio'] = data_copy['High'].shift(2) / data_copy['Low'].shift(2)
            data_copy['Close_Open_Ratio'] = data_copy['Close'].shift(2) / data_copy['Open'].shift(2)
            
            # Price position within daily range
            data_copy['Price_Position'] = ((data_copy['Close'].shift(2) - data_copy['Low'].shift(2)) / 
                                (data_copy['High'].shift(2) - data_copy['Low'].shift(2)))
            
            # Lagged features (previous day values)
            for lag in [1, 2, 3, 5]:
                data_copy[f'Close_Lag_{lag}'] = data_copy['Close'].shift(lag)
                data_copy[f'Volume_Lag_{lag}'] = data_copy['Volume'].shift(lag) if 'Volume' in data_copy.columns else 0
            
            # Moving averages of different periods
            for window in [5, 10, 20, 50]:
                data_copy[f'MA_{window}'] = data_copy['Close'].shift(2).rolling(window=window, min_periods=1).mean()
                data_copy[f'MA_{window}_Ratio'] = data_copy['Close'].shift(2) / data_copy[f'MA_{window}']
            
            # Volatility features
            data_copy['Price_Volatility_5'] = data_copy['Close'].shift(2).rolling(window=5, min_periods=1).std()
            data_copy['Price_Volatility_20'] = data_copy['Close'].shift(2).rolling(window=20, min_periods=1).std()
            
            # RSI (Relative Strength Index) approximation
            delta = data_copy['Close'].shift(1).diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / loss
            data_copy['RSI'] = 100 - (100 / (1 + rs))
            
        # Volume-based features
        if 'Volume' in data_copy.columns:
            # Volume moving averages
            data_copy['Volume_MA_5'] = data_copy['Volume'].shift(2).rolling(window=5, min_periods=1).mean()
            data_copy['Volume_MA_20'] = data_copy['Volume'].shift(2).rolling(window=20, min_periods=1).mean()
            
            # Volume ratios
            data_copy['Volume_Ratio_5'] = data_copy['Volume'].shift(2) / data_copy['Volume_MA_5']
            data_copy['Volume_Ratio_20'] = data_copy['Volume'].shift(2) / data_copy['Volume_MA_20']
            
            # Price-Volume features
            data_copy['Price_Volume'] = data_copy['Close'].shift(2) * data_copy['Volume'].shift(2)
            
        # Trend features
        if 'Close' in data_copy.columns:
            # Price trends (comparing current to previous periods)
            data_copy['Trend_5'] = (data_copy['Close'].shift(2) - data_copy['Close'].shift(5)) / data_copy['Close'].shift(5)
            data_copy['Trend_10'] = (data_copy['Close'].shift(2) - data_copy['Close'].shift(10)) / data_copy['Close'].shift(10)
            
        # Time-based features (if not already present)
        if 'Date' in data_copy.columns:
            data_copy['DayOfWeek'] = data_copy['Date'].dt.dayofweek
            data_copy['Month'] = data_copy['Date'].dt.month
            data_copy['Quarter'] = data_copy['Date'].dt.quarter
            
            # Days since start (for trend analysis)
            data_copy['Days_Since_Start'] = (data_copy['Date'] - data_copy['Date'].min()).dt.days
        
        # Drop rows with NaN values created by feature engineering
        initial_rows = len(data_copy)
        data_copy = data_copy.dropna()
        final_rows = len(data_copy)
        
        if initial_rows != final_rows:
            self.logger.info(f"Removed {initial_rows - final_rows} rows due to feature engineering NaN values")
        
        return data_copy
        
    def prepare_features_and_target(self, data, target_column='Adj Close'):
        """
        Prepare feature matrix and target variable
        
        Args:
            data (pandas.DataFrame): Processed data
            target_column (str): Name of target column
            
        Returns:
            tuple: (X, y, feature_names)
        """
        
        # Define which columns to exclude from features
        exclude_columns = [
            'Date', 'Symbol', target_column,
            # Exclude future-looking or identical features
            'Adj Close' if target_column == 'Close' else None
        ]
        exclude_columns = [col for col in exclude_columns if col is not None]
        
        # Select feature columns
        feature_columns = [col for col in data.columns if col not in exclude_columns]
        
        # Ensure we have numeric features only
        numeric_features = []
        for col in feature_columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                numeric_features.append(col)
        
        self.feature_columns = numeric_features
        
        # Create feature matrix and target vector
        X = data[self.feature_columns].values
        y = data[target_column].values
            
        return X, y, self.feature_columns
        
        
    def scale_features(self, X_train, X_val, X_test, method='standard'):
        """
        Scale features using specified method
        
        Args:
            X_train, X_val, X_test: Feature matrices
            method (str): Scaling method ('standard', 'minmax')
            
        Returns:
            tuple: Scaled feature matrices
        """    
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            return X_train, X_val, X_test
        
        # Fit scaler on training data only
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Store scaler for future use
        self.scalers[method] = scaler
        
        return X_train_scaled, X_val_scaled, X_test_scaled
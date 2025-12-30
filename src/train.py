import numpy as np
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

class TrainModel:
    def initialize_models(self):
        """
        Initialize different ML models for comparison
        """
        self.logger.info("Initializing ML models")
        
        self.models = {
            'Linear_Regression': LinearRegression(),
            
            'Ridge_Regression': Ridge(alpha=1.0),
            
            'Lasso_Regression': Lasso(alpha=0.1),
            
            'Decision_Tree': DecisionTreeRegressor(
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            
            'Random_Forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            
            'Gradient_Boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            
            'Neural_Network': MLPRegressor(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            )
        }
            

    def train_and_evaluate_on_val(self, X_train, y_train, X_val, y_val):
        self.model_results = {}
        self.best_model = None
        self.best_score = -np.inf

        for model_name, model in self.models.items():
            self.logger.info(f"Training {model_name}...")
            
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)

            results = {
                'model': model,
                'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
                'train_mae': mean_absolute_error(y_train, y_train_pred),
                'train_r2': r2_score(y_train, y_train_pred),
                
                'val_rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
                'val_mae': mean_absolute_error(y_val, y_val_pred),
                'val_r2': r2_score(y_val, y_val_pred),
                
                'y_train_pred': y_train_pred,
                'y_val_pred': y_val_pred
            }

            self.model_results[model_name] = results

            if results['val_r2'] > self.best_score:
                self.best_score = results['val_r2']
                self.best_model = model_name
                
            self.logger.info(f"{model_name} - Val R²: {results['val_r2']:.4f}, "
                                f"Val RMSE: {results['val_rmse']:.4f}")
            

    def perform_hyperparameter_tuning(self, X_train, y_train, model_name=None):
        """
        Perform hyperparameter tuning for specified model
        
        Args:
            X_train, y_train: Training data
            model_name (str): Model to tune (if None, tune best model)
        """
        if model_name is None:
            model_name = self.best_model
        
        self.logger.info(f"Performing hyperparameter tuning for {model_name}")
        
        # Define parameter grids
        param_grids = {
            'Random_Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            
            'Gradient_Boosting': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'min_samples_split': [2, 5, 10]
            },
            
            'Ridge_Regression': {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            },
            
            'Lasso_Regression': {
                'alpha': [0.01, 0.1, 1.0, 10.0]
            }
        }
        
        if model_name not in param_grids:
            self.logger.info(f"No parameter grid defined for {model_name}")
            return
        
        try:
            # Get base model
            base_model = self.models[model_name]
            param_grid = param_grids[model_name]
            
            # Perform grid search with cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=tscv,  
                scoring='r2',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            # Update model with best parameters
            self.models[f'{model_name}_Tuned'] = grid_search.best_estimator_
            
            self.logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
            self.logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
            
            # Return the tuned model
            return grid_search.best_estimator_
            
        except Exception as e:
            self.logger.error(f"Error in hyperparameter tuning for {model_name}: {e}")
            return None
        
"""
Air Quality Prediction Model Training Script
Trains multiple regression models to predict air pollutants and saves the best model for dashboard use.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Model imports
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb

# Model persistence
import joblib
import json

def load_and_prepare_data(filepath):
    """
    Load and preprocess the air quality dataset.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
        
    Returns:
    --------
    pandas.DataFrame
        Preprocessed DataFrame with DateTime index
    """
    print("Step 1: Loading and preparing data...")
    
    # Load the dataset
    df = pd.read_csv(filepath)
    print(f"Loaded dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Drop rows where Date is missing (critical for time-based analysis)
    initial_rows = len(df)
    df = df.dropna(subset=['Date'])
    print(f"Dropped {initial_rows - len(df)} rows with missing Date")
    
    # Convert Date and Time to datetime - FIXED: Use dayfirst=True for European format
    # The data uses DD/MM/YYYY format, so we need to specify dayfirst=True
    try:
        # First attempt with dayfirst=True
        df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)
    except Exception as e:
        print(f"Warning: Error parsing dates with dayfirst=True: {e}")
        # Try mixed format as fallback
        df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='mixed', dayfirst=True)
    
    # Set DateTime as index for time series analysis
    df.set_index('DateTime', inplace=True)
    
    # Drop the original Date and Time columns as they're now redundant
    df.drop(['Date', 'Time'], axis=1, inplace=True, errors='ignore')
    
    # Ensure numeric features are correctly typed
    # Identify numeric columns (excluding the target columns for now)
    numeric_cols = ['PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 
                    'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH',
                    'CO(GT)', 'NOx(GT)', 'NO2(GT)', 'C6H6(GT)']
    
    # Convert to numeric, coercing errors to NaN
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with NaN values in target variables (cannot train on missing targets)
    initial_rows = len(df)
    df = df.dropna(subset=['CO(GT)', 'NOx(GT)', 'NO2(GT)', 'C6H6(GT)'])
    print(f"Dropped {initial_rows - len(df)} rows with missing target values")
    
    # Fill remaining feature NaN values with forward fill (common in time series)
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    print(f"Final dataset shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    return df

def prepare_features_targets(df):
    """
    Prepare features (X) and targets (y) for model training.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Preprocessed DataFrame
        
    Returns:
    --------
    tuple
        (X, y, feature_names, target_names)
    """
    print("\nStep 2: Preparing features and targets...")
    
    # Define features (sensor readings and environmental measurements)
    feature_cols = [
        'PT08.S1(CO)',    # Tin oxide sensor for CO
        'PT08.S2(NMHC)',  # Titanium dioxide sensor for NMHC
        'PT08.S3(NOx)',   # Tungsten oxide sensor for NOx
        'PT08.S4(NO2)',   # Tungsten oxide sensor for NO2
        'PT08.S5(O3)',    # Indium oxide sensor for O3
        'T',              # Temperature (°C)
        'RH',             # Relative Humidity (%)
        'AH'              # Absolute Humidity
    ]
    
    # Define targets (ground truth pollutant concentrations)
    target_cols = [
        'CO(GT)',   # Carbon Monoxide (mg/m³)
        'NOx(GT)',  # Nitrogen Oxides (ppb)
        'NO2(GT)',  # Nitrogen Dioxide (μg/m³)
        'C6H6(GT)'  # Benzene (μg/m³)
    ]
    
    # Verify all columns exist
    missing_features = [col for col in feature_cols if col not in df.columns]
    missing_targets = [col for col in target_cols if col not in df.columns]
    
    if missing_features:
        print(f"Warning: Missing features: {missing_features}")
    if missing_targets:
        print(f"Warning: Missing targets: {missing_targets}")
    
    # Extract features and targets
    X = df[feature_cols].copy()
    y = df[target_cols].copy()
    
    print(f"Features shape: {X.shape}")
    print(f"Targets shape: {y.shape}")
    print(f"Features: {feature_cols}")
    print(f"Targets: {target_cols}")
    
    return X, y, feature_cols, target_cols

def split_and_scale_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into train/test sets and scale features.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Feature matrix
    y : pandas.DataFrame
        Target matrix
    test_size : float
        Proportion of data for testing
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        (X_train_scaled, X_test_scaled, y_train, y_test, scaler)
    """
    print("\nStep 3: Splitting and scaling data...")
    
    # Split into training and testing sets
    # Stratify is not typically used for regression, but we maintain temporal order
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=False
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    # Scale features using StandardScaler (zero mean, unit variance)
    # Important for models sensitive to feature scales (NN, SVM, etc.)
    scaler = StandardScaler()
    
    # Fit scaler on training data only to prevent data leakage
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrames for better readability
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)
    
    print("Features scaled using StandardScaler")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_models(X_train, y_train):
    """
    Train multiple regression models for multi-output prediction.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame or numpy.array
        Scaled training features
    y_train : pandas.DataFrame
        Training targets
        
    Returns:
    --------
    dict
        Dictionary of trained models
    """
    print("\nStep 4: Training regression models...")
    
    models = {}
    
    # 1. Linear Regression (Baseline model)
    print("Training Linear Regression...")
    lr_model = MultiOutputRegressor(LinearRegression())
    lr_model.fit(X_train, y_train)
    models['LinearRegression'] = lr_model
    
    # 2. Random Forest Regressor (Ensemble method, robust to outliers)
    print("Training Random Forest Regressor...")
    rf_model = MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1  # Use all available cores
        )
    )
    rf_model.fit(X_train, y_train)
    models['RandomForest'] = rf_model
    
    # 3. XGBoost Regressor (Gradient boosting, often high performance)
    print("Training XGBoost Regressor...")
    xgb_model = MultiOutputRegressor(
        xgb.XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
    )
    xgb_model.fit(X_train, y_train)
    models['XGBoost'] = xgb_model
    
    # 4. Neural Network (MLP Regressor)
    print("Training Neural Network (MLPRegressor)...")
    nn_model = MLPRegressor(
        hidden_layer_sizes=(100, 50),  # Two hidden layers
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    nn_model.fit(X_train, y_train)
    models['NeuralNetwork'] = nn_model
    
    print(f"Trained {len(models)} models successfully")
    
    return models

def evaluate_models(models, X_test, y_test, target_names):
    """
    Evaluate all models and calculate performance metrics.
    
    Parameters:
    -----------
    models : dict
        Dictionary of trained models
    X_test : pandas.DataFrame or numpy.array
        Test features
    y_test : pandas.DataFrame
        True test targets
    target_names : list
        List of target variable names
        
    Returns:
    --------
    tuple
        (metrics_df, predictions_dict)
    """
    print("\nStep 5: Evaluating models...")
    
    metrics = []
    predictions_dict = {}
    
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Store predictions for analysis
        predictions_dict[model_name] = y_pred
        
        # Calculate metrics for each target
        for i, target in enumerate(target_names):
            # Extract true and predicted values for this target
            y_true_target = y_test.iloc[:, i]
            y_pred_target = y_pred[:, i]
            
            # Calculate RMSE (Root Mean Square Error)
            rmse = np.sqrt(mean_squared_error(y_true_target, y_pred_target))
            
            # Calculate R² (Coefficient of Determination)
            r2 = r2_score(y_true_target, y_pred_target)
            
            # Store metrics
            metrics.append({
                'model': model_name,
                'target': target,
                'rmse': rmse,
                'r2': r2,
                'rmse_normalized': rmse / y_true_target.mean() if y_true_target.mean() != 0 else rmse
            })
            
            print(f"  {target}: RMSE = {rmse:.4f}, R² = {r2:.4f}")
    
    # Convert to DataFrame for easier analysis
    metrics_df = pd.DataFrame(metrics)
    
    # Calculate average metrics per model
    avg_metrics = metrics_df.groupby('model')[['rmse', 'r2']].mean()
    print("\nAverage performance per model:")
    print(avg_metrics)
    
    return metrics_df, predictions_dict

def select_best_model(metrics_df):
    """
    Select the best model based on average RMSE across all targets.
    
    Parameters:
    -----------
    metrics_df : pandas.DataFrame
        DataFrame containing model performance metrics
        
    Returns:
    --------
    str
        Name of the best model
    """
    print("\nStep 6: Selecting best model...")
    
    # Calculate average RMSE for each model
    model_avg_rmse = metrics_df.groupby('model')['rmse'].mean()
    
    # Find model with lowest average RMSE
    best_model_name = model_avg_rmse.idxmin()
    best_rmse = model_avg_rmse.min()
    
    # Also check R²
    model_avg_r2 = metrics_df.groupby('model')['r2'].mean()
    best_r2 = model_avg_r2.max()
    
    print(f"Best model selected: {best_model_name}")
    print(f"Average RMSE: {best_rmse:.4f}")
    print(f"Average R²: {best_r2:.4f}")
    
    return best_model_name

def save_artifacts(best_model_name, models, scaler, metrics_df, feature_names, target_names):
    """
    Save model, scaler, and metrics for dashboard use.
    
    Parameters:
    -----------
    best_model_name : str
        Name of the best model
    models : dict
        Dictionary of all trained models
    scaler : sklearn.preprocessing.StandardScaler
        Fitted scaler
    metrics_df : pandas.DataFrame
        Performance metrics
    feature_names : list
        List of feature names
    target_names : list
        List of target names
    """
    print("\nStep 7: Saving artifacts for dashboard...")
    
    # 1. Save the best model
    best_model = models[best_model_name]
    model_filename = f'best_model_{best_model_name}.joblib'
    joblib.dump(best_model, model_filename)
    print(f"Saved best model to: {model_filename}")
    
    # 2. Save the scaler
    scaler_filename = 'feature_scaler.joblib'
    joblib.dump(scaler, scaler_filename)
    print(f"Saved scaler to: {scaler_filename}")
    
    # 3. Save metrics to CSV
    metrics_filename = 'model_performance_metrics.csv'
    metrics_df.to_csv(metrics_filename, index=False)
    print(f"Saved metrics to: {metrics_filename}")
    
    # 4. Save model metadata
    metadata = {
        'best_model': best_model_name,
        'features': feature_names,
        'targets': target_names,
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_type': str(type(best_model)),
        'n_features': len(feature_names),
        'n_targets': len(target_names)
    }
    
    metadata_filename = 'model_metadata.json'
    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to: {metadata_filename}")
    
    # 5. Save summary statistics
    summary = metrics_df.groupby('model').agg({
        'rmse': ['mean', 'std', 'min', 'max'],
        'r2': ['mean', 'std', 'min', 'max']
    }).round(4)
    
    summary_filename = 'model_performance_summary.csv'
    summary.to_csv(summary_filename)
    print(f"Saved performance summary to: {summary_filename}")
    
    return {
        'model_file': model_filename,
        'scaler_file': scaler_filename,
        'metrics_file': metrics_filename,
        'metadata_file': metadata_filename,
        'summary_file': summary_filename
    }

def main():
    """
    Main execution function.
    """
    print("="*60)
    print("AIR QUALITY PREDICTION MODEL TRAINING")
    print("="*60)
    
    # File path to the cleaned dataset
    data_file = 'AirQualityUCI_cleaned.csv'
    
    try:
        # Step 1: Load and prepare data
        df = load_and_prepare_data(data_file)
        
        # Step 2: Prepare features and targets
        X, y, feature_names, target_names = prepare_features_targets(df)
        
        # Step 3: Split and scale data
        X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_and_scale_data(X, y)
        
        # Step 4: Train models
        models = train_models(X_train_scaled, y_train)
        
        # Step 5: Evaluate models
        metrics_df, predictions_dict = evaluate_models(models, X_test_scaled, y_test, target_names)
        
        # Step 6: Select best model
        best_model_name = select_best_model(metrics_df)
        
        # Step 7: Save artifacts
        artifacts = save_artifacts(
            best_model_name, models, scaler, metrics_df, 
            feature_names, target_names
        )
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"\nBest model: {best_model_name}")
        print("\nSaved artifacts:")
        for key, value in artifacts.items():
            print(f"  {key}: {value}")
        
        # Display final comparison
        print("\nModel Comparison (Average RMSE - lower is better):")
        comparison = metrics_df.groupby('model')['rmse'].mean().sort_values()
        for model, rmse in comparison.items():
            star = " ★" if model == best_model_name else ""
            print(f"  {model}: {rmse:.4f}{star}")
            
    except FileNotFoundError:
        print(f"Error: File '{data_file}' not found.")
        print("Please ensure the file is in the current directory.")
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
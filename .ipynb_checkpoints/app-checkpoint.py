"""
Air Quality Analysis Dashboard
Senior Data Scientist - Environmental Data Analysis & Streamlit Dashboarding

This script performs two main functions:
1. Machine Learning Pipeline: Train and compare models to predict 4 air pollutants
2. Streamlit Dashboard: Interactive interface for predictions and data visualization

Requirements: pandas, scikit-learn, xgboost, streamlit, plotly
"""

# Import libraries
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
import xgboost as xgb

# Dashboard libraries
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import io

# ==============================================
# PART 1: MACHINE LEARNING PIPELINE
# ==============================================

def load_and_preprocess_data():
    """
    Load and preprocess the air quality dataset.
    Handles missing dates and creates datetime index.
    
    Returns:
        pandas.DataFrame: Preprocessed dataframe with datetime index
    """
    print("Loading and preprocessing data...")
    
    # Load the dataset
    df = pd.read_csv('AirQualityUCI_cleaned.csv')
    
    print(f"Original dataset shape: {df.shape}")
    
    # Drop rows where Date is missing (last ~114 rows as per description)
    df = df.dropna(subset=['Date'])
    
    # Combine Date and Time columns into a proper datetime
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], 
                                    format='%d/%m/%Y %H.%M.%S', errors='coerce')
    
    # Drop any rows with invalid datetime (additional safety)
    df = df.dropna(subset=['DateTime'])
    
    # Set DateTime as index
    df.set_index('DateTime', inplace=True)
    
    # Sort by datetime
    df.sort_index(inplace=True)
    
    print(f"Cleaned dataset shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    return df

def prepare_model_data(df):
    """
    Prepare features and targets for machine learning.
    
    Args:
        df (pandas.DataFrame): Preprocessed dataframe
    
    Returns:
        tuple: X (features), y (targets), feature_names, target_names
    """
    print("\nPreparing model data...")
    
    # Define feature columns (8 sensor readings)
    feature_cols = [
        'PT08.S1(CO)',     # Tin oxide sensor for CO
        'PT08.S2(NMHC)',   # Titania sensor for NMHC
        'PT08.S3(NOx)',    # Tungsten oxide sensor for NOx
        'PT08.S4(NO2)',    # Tungsten oxide sensor for NO2
        'PT08.S5(O3)',     # Indium oxide sensor for O3
        'T',               # Temperature
        'RH',              # Relative Humidity
        'AH'               # Absolute Humidity
    ]
    
    # Define target columns (4 true concentration measurements)
    target_cols = [
        'CO(GT)',    # True Carbon Monoxide (mg/m³)
        'NOx(GT)',   # True Nitrogen Oxides (ppb)
        'NO2(GT)',   # True Nitrogen Dioxide (μg/m³)
        'C6H6(GT)'   # True Benzene (μg/m³)
    ]
    
    # Extract features and targets
    X = df[feature_cols].copy()
    y = df[target_cols].copy()
    
    # Check for any remaining missing values in features
    if X.isnull().any().any():
        print("Warning: Missing values found in features. Filling with column means.")
        X = X.fillna(X.mean())
    
    # Check for any remaining missing values in targets
    if y.isnull().any().any():
        print("Warning: Missing values found in targets. Filling with column means.")
        y = y.fillna(y.mean())
    
    print(f"Features shape: {X.shape}")
    print(f"Targets shape: {y.shape}")
    print(f"Feature columns: {feature_cols}")
    print(f"Target columns: {target_cols}")
    
    return X, y, feature_cols, target_cols

def train_and_evaluate_models(X, y, feature_names, target_names):
    """
    Train multiple regression models and compare their performance.
    
    Args:
        X (numpy.ndarray): Feature matrix
        y (numpy.ndarray): Target matrix
        feature_names (list): Names of features
        target_names (list): Names of targets
    
    Returns:
        dict: Dictionary containing trained models, scalers, and performance metrics
    """
    print("\nTraining and evaluating models...")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False  # No shuffle for time series
    )
    
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize models
    models = {
        'Linear Regression': MultiOutputRegressor(LinearRegression()),
        'Random Forest': MultiOutputRegressor(RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        )),
        'XGBoost': MultiOutputRegressor(xgb.XGBRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        )),
        'Neural Network': MLPRegressor(
            hidden_layer_sizes=(100, 50), 
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42
        )
    }
    
    # Dictionary to store results
    results = {
        'models': {},
        'metrics': {},
        'scaler': scaler,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'feature_names': feature_names,
        'target_names': target_names
    }
    
    # Train and evaluate each model
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        
        try:
            # Fit the model
            if model_name == 'Neural Network':
                # MLPRegressor directly supports multi-output
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                # MultiOutputRegressor wrapper
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred, multioutput='raw_values'))
            r2 = r2_score(y_test, y_pred, multioutput='raw_values')
            
            # Store results
            results['models'][model_name] = model
            results['metrics'][model_name] = {
                'RMSE': dict(zip(target_names, rmse)),
                'R2': dict(zip(target_names, r2)),
                'Overall_RMSE': np.mean(rmse),
                'Overall_R2': np.mean(r2)
            }
            
            print(f"{model_name} - Overall RMSE: {np.mean(rmse):.4f}, R²: {np.mean(r2):.4f}")
            
        except Exception as e:
            print(f"Error training {model_name}: {e}")
    
    # Select best model based on overall RMSE (lower is better)
    best_model_name = min(
        results['metrics'].keys(),
        key=lambda x: results['metrics'][x]['Overall_RMSE']
    )
    
    results['best_model'] = best_model_name
    results['best_model_instance'] = results['models'][best_model_name]
    
    print(f"\n✅ Best model: {best_model_name}")
    print(f"Best model overall RMSE: {results['metrics'][best_model_name]['Overall_RMSE']:.4f}")
    
    return results

def create_prediction_pipeline(results):
    """
    Create a function that can make predictions using the best model.
    
    Args:
        results (dict): Results from train_and_evaluate_models
    
    Returns:
        function: Prediction function
    """
    best_model = results['best_model_instance']
    scaler = results['scaler']
    feature_names = results['feature_names']
    target_names = results['target_names']
    
    def predict(features):
        """
        Predict pollutant concentrations from sensor readings.
        
        Args:
            features (dict or pandas.DataFrame): Input features
        
        Returns:
            pandas.DataFrame: Predicted concentrations
        """
        # Convert input to dataframe
        if isinstance(features, dict):
            features_df = pd.DataFrame([features])
        else:
            features_df = features.copy()
        
        # Ensure all feature columns are present
        for col in feature_names:
            if col not in features_df.columns:
                raise ValueError(f"Missing feature: {col}")
        
        # Reorder columns to match training order
        features_df = features_df[feature_names]
        
        # Check if we need to scale (based on model type)
        if results['best_model'] == 'Neural Network':
            features_scaled = scaler.transform(features_df)
            predictions = best_model.predict(features_scaled)
        else:
            predictions = best_model.predict(features_df)
        
        # Convert to dataframe with proper column names
        predictions_df = pd.DataFrame(predictions, columns=target_names)
        
        return predictions_df
    
    return predict

# ==============================================
# PART 2: STREAMLIT DASHBOARD
# ==============================================

def setup_streamlit_ui():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="Air Quality Analysis Dashboard",
        page_icon="🌫️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def create_sidebar_controls(df, feature_names):
    """
    Create sidebar controls for the dashboard.
    
    Args:
        df (pandas.DataFrame): The complete dataset
        feature_names (list): Names of feature columns
    
    Returns:
        dict: Dictionary of user inputs
    """
    st.sidebar.header("📊 Dashboard Controls")
    
    # Prediction mode selector
    mode = st.sidebar.radio(
        "Prediction Mode",
        ["Manual Input", "Historical Analysis"],
        help="Choose between manual prediction or historical analysis"
    )
    
    user_inputs = {'mode': mode}
    
    if mode == "Manual Input":
        st.sidebar.subheader("🎛️ Sensor Readings")
        
        # Get feature ranges from data for slider limits
        feature_ranges = {}
        for feature in feature_names:
            min_val = float(df[feature].min())
            max_val = float(df[feature].max())
            mean_val = float(df[feature].mean())
            feature_ranges[feature] = (min_val, max_val, mean_val)
        
        # Create sliders for each feature
        for feature in feature_names:
            min_val, max_val, mean_val = feature_ranges[feature]
            
            # Create a more informative label
            label_map = {
                'PT08.S1(CO)': 'PT08.S1 (CO Sensor)',
                'PT08.S2(NMHC)': 'PT08.S2 (NMHC Sensor)',
                'PT08.S3(NOx)': 'PT08.S3 (NOx Sensor)',
                'PT08.S4(NO2)': 'PT08.S4 (NO2 Sensor)',
                'PT08.S5(O3)': 'PT08.S5 (O3 Sensor)',
                'T': 'Temperature (°C)',
                'RH': 'Relative Humidity (%)',
                'AH': 'Absolute Humidity'
            }
            
            label = label_map.get(feature, feature)
            
            user_inputs[feature] = st.sidebar.slider(
                label,
                min_value=float(min_val),
                max_value=float(max_val),
                value=float(mean_val),
                step=float((max_val - min_val) / 100),
                help=f"Range: {min_val:.1f} to {max_val:.1f}"
            )
    
    else:  # Historical Analysis mode
        st.sidebar.subheader("📅 Date Range Selection")
        
        # Get date range from data
        min_date = df.index.min().date()
        max_date = df.index.max().date()
        
        # Date range selector
        start_date = st.sidebar.date_input(
            "Start Date",
            value=min_date,
            min_value=min_date,
            max_value=max_date
        )
        
        end_date = st.sidebar.date_input(
            "End Date",
            value=max_date,
            min_value=min_date,
            max_value=max_date
        )
        
        # Ensure start_date <= end_date
        if start_date > end_date:
            st.sidebar.error("Start date must be before end date")
            start_date, end_date = min_date, max_date
        
        user_inputs['start_date'] = start_date
        user_inputs['end_date'] = end_date
        
        # Pollutant selector for charts
        pollutants = ['CO(GT)', 'NOx(GT)', 'NO2(GT)', 'C6H6(GT)']
        selected_pollutants = st.sidebar.multiselect(
            "Pollutants to Display",
            pollutants,
            default=pollutants,
            help="Select which pollutants to show in charts"
        )
        
        user_inputs['selected_pollutants'] = selected_pollutants
    
    return user_inputs

def display_real_time_prediction(predict_fn, user_inputs, feature_names, target_names):
    """
    Display real-time prediction results.
    
    Args:
        predict_fn (function): Prediction function
        user_inputs (dict): User inputs from sidebar
        feature_names (list): Feature column names
        target_names (list): Target column names
    """
    st.header("🌡️ Real-Time Air Quality Prediction")
    
    # Create feature dictionary from user inputs
    features = {feature: user_inputs[feature] for feature in feature_names}
    
    # Make prediction
    try:
        predictions = predict_fn(features)
        
        # Display predictions in columns
        cols = st.columns(4)
        
        # Map pollutant names to units and descriptions
        pollutant_info = {
            'CO(GT)': {
                'unit': 'mg/m³',
                'description': 'Carbon Monoxide',
                'safe_limit': 4.0  # 8-hour average limit (mg/m³)
            },
            'NOx(GT)': {
                'unit': 'ppb',
                'description': 'Nitrogen Oxides',
                'safe_limit': 100.0  # Annual mean (ppb)
            },
            'NO2(GT)': {
                'unit': 'μg/m³',
                'description': 'Nitrogen Dioxide',
                'safe_limit': 40.0  # Annual mean (μg/m³)
            },
            'C6H6(GT)': {
                'unit': 'μg/m³',
                'description': 'Benzene',
                'safe_limit': 5.0  # Annual mean (μg/m³)
            }
        }
        
        for idx, (col, pollutant) in enumerate(zip(cols, target_names)):
            with col:
                value = predictions[pollutant].iloc[0]
                info = pollutant_info[pollutant]
                
                # Create a metric display with color coding based on safety
                if value > info['safe_limit']:
                    st.metric(
                        label=info['description'],
                        value=f"{value:.2f} {info['unit']}",
                        delta="⚠️ Above safe limit",
                        delta_color="inverse"
                    )
                else:
                    st.metric(
                        label=info['description'],
                        value=f"{value:.2f} {info['unit']}",
                        delta="✅ Within safe limits",
                        delta_color="normal"
                    )
                
                # Add gauge chart for visual representation
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=value,
                    title={'text': f"{info['description']} ({info['unit']})"},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [0, info['safe_limit'] * 2]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, info['safe_limit']], 'color': "lightgreen"},
                            {'range': [info['safe_limit'], info['safe_limit'] * 2], 'color': "lightcoral"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': info['safe_limit']
                        }
                    }
                ))
                
                fig.update_layout(height=200, margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Prediction error: {e}")

def display_historical_analysis(df, user_inputs, target_names):
    """
    Display historical data analysis with interactive charts.
    
    Args:
        df (pandas.DataFrame): Complete dataset
        user_inputs (dict): User inputs from sidebar
        target_names (list): Target column names
    """
    st.header("📈 Historical Air Quality Analysis")
    
    # Filter data by date range
    start_date = pd.Timestamp(user_inputs['start_date'])
    end_date = pd.Timestamp(user_inputs['end_date']) + pd.Timedelta(days=1)
    
    mask = (df.index >= start_date) & (df.index <= end_date)
    filtered_df = df.loc[mask]
    
    if len(filtered_df) == 0:
        st.warning("No data available for the selected date range.")
        return
    
    # Display statistics
    st.subheader("📊 Summary Statistics")
    
    # Calculate basic statistics for selected pollutants
    selected_pollutants = user_inputs.get('selected_pollutants', target_names)
    stats_df = filtered_df[selected_pollutants].describe().T[['mean', 'std', 'min', 'max']]
    stats_df.columns = ['Average', 'Std Dev', 'Minimum', 'Maximum']
    
    st.dataframe(stats_df.style.format("{:.2f}").background_gradient(cmap='Blues'))
    
    # Create time series charts
    st.subheader("📉 Time Series Trends")
    
    # Create subplots
    fig = make_subplots(
        rows=len(selected_pollutants), 
        cols=1,
        subplot_titles=selected_pollutants,
        vertical_spacing=0.1
    )
    
    # Map pollutant names to units
    units = {
        'CO(GT)': 'mg/m³',
        'NOx(GT)': 'ppb',
        'NO2(GT)': 'μg/m³',
        'C6H6(GT)': 'μg/m³'
    }
    
    for idx, pollutant in enumerate(selected_pollutants, 1):
        fig.add_trace(
            go.Scatter(
                x=filtered_df.index,
                y=filtered_df[pollutant],
                mode='lines',
                name=pollutant,
                line=dict(width=1),
                hovertemplate=f"{pollutant}: %{{y:.2f}} {units.get(pollutant, '')}<br>Time: %{{x}}<extra></extra>"
            ),
            row=idx, col=1
        )
    
    fig.update_layout(
        height=300 * len(selected_pollutants),
        showlegend=False,
        title_text="Pollutant Concentrations Over Time",
        hovermode='x unified'
    )
    
    # Update y-axis labels with units
    for idx, pollutant in enumerate(selected_pollutants, 1):
        fig.update_yaxes(
            title_text=f"{units.get(pollutant, '')}",
            row=idx, col=1
        )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    st.subheader("🔥 Correlation Matrix")
    
    # Calculate correlation for selected pollutants and features
    corr_cols = selected_pollutants + ['PT08.S1(CO)', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 'T', 'RH']
    corr_matrix = filtered_df[corr_cols].corr()
    
    fig_corr = px.imshow(
        corr_matrix,
        text_auto='.2f',
        aspect='auto',
        color_continuous_scale='RdBu',
        range_color=[-1, 1],
        title="Correlation Between Pollutants and Sensor Readings"
    )
    
    fig_corr.update_layout(height=600)
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Export button for filtered data
    st.subheader("📥 Export Data")
    
    csv = filtered_df.to_csv().encode('utf-8')
    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name=f"air_quality_{start_date.date()}_to_{end_date.date()}.csv",
        mime="text/csv",
        help="Download the filtered historical data for your analysis"
    )

def display_model_performance(results):
    """
    Display model performance metrics in a clean format.
    
    Args:
        results (dict): Results from model training
    """
    st.header("🤖 Model Performance Comparison")
    
    # Create a dataframe for model metrics
    metrics_data = []
    
    for model_name, metrics in results['metrics'].items():
        for pollutant in results['target_names']:
            metrics_data.append({
                'Model': model_name,
                'Pollutant': pollutant,
                'RMSE': metrics['RMSE'][pollutant],
                'R²': metrics['R²'][pollutant]
            })
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Display best model
    best_model = results['best_model']
    st.success(f"**Best Performing Model**: {best_model}")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 RMSE Comparison", "📈 R² Comparison", "📋 Detailed Table"])
    
    with tab1:
        # RMSE bar chart
        fig_rmse = px.bar(
            metrics_df,
            x='Pollutant',
            y='RMSE',
            color='Model',
            barmode='group',
            title="Root Mean Square Error (RMSE) by Pollutant",
            labels={'RMSE': 'RMSE (Lower is Better)'},
            text_auto='.3f'
        )
        fig_rmse.update_layout(height=500)
        st.plotly_chart(fig_rmse, use_container_width=True)
    
    with tab2:
        # R² bar chart
        fig_r2 = px.bar(
            metrics_df,
            x='Pollutant',
            y='R²',
            color='Model',
            barmode='group',
            title="R² Score by Pollutant",
            labels={'R²': 'R² Score (Higher is Better)'},
            text_auto='.3f',
            range_y=[0, 1] if metrics_df['R²'].max() <= 1 else None
        )
        fig_r2.update_layout(height=500)
        st.plotly_chart(fig_r2, use_container_width=True)
    
    with tab3:
        # Pivot table view
        pivot_rmse = metrics_df.pivot(index='Model', columns='Pollutant', values='RMSE')
        pivot_r2 = metrics_df.pivot(index='Model', columns='Pollutant', values='R²')
        
        st.subheader("RMSE Scores")
        st.dataframe(pivot_rmse.style.format("{:.4f}").background_gradient(cmap='RdYlGn_r'))
        
        st.subheader("R² Scores")
        st.dataframe(pivot_r2.style.format("{:.4f}").background_gradient(cmap='RdYlGn'))

def main():
    """
    Main function to run the complete application.
    """
    # Setup Streamlit UI
    setup_streamlit_ui()
    
    # Title and description
    st.title("🌍 Air Quality Analysis Dashboard")
    st.markdown("""
    This dashboard analyzes air quality sensor data and predicts pollutant concentrations 
    using machine learning models. The system can:
    - Predict real-time pollutant levels from sensor readings
    - Visualize historical air quality trends
    - Compare performance of different ML models
    """)
    
    # Load and preprocess data
    with st.spinner("Loading and preprocessing data..."):
        try:
            df = load_and_preprocess_data()
            X, y, feature_names, target_names = prepare_model_data(df)
        except FileNotFoundError:
            st.error("❌ File 'AirQualityUCI_cleaned.csv' not found. Please ensure the file is in the correct directory.")
            st.stop()
        except Exception as e:
            st.error(f"❌ Error loading data: {e}")
            st.stop()
    
    # Train models (cache this to avoid retraining on every interaction)
    @st.cache_resource
    def train_models():
        with st.spinner("Training machine learning models (this may take a minute)..."):
            results = train_and_evaluate_models(X, y, feature_names, target_names)
            predict_fn = create_prediction_pipeline(results)
            return results, predict_fn
    
    try:
        results, predict_fn = train_models()
    except Exception as e:
        st.error(f"❌ Error training models: {e}")
        st.stop()
    
    # Create sidebar controls
    user_inputs = create_sidebar_controls(df, feature_names)
    
    # Main content based on mode
    if user_inputs['mode'] == "Manual Input":
        # Display real-time prediction
        display_real_time_prediction(predict_fn, user_inputs, feature_names, target_names)
        
        # Add feature importance if available
        st.subheader("🔍 Feature Importance")
        
        if results['best_model'] == 'Random Forest':
            try:
                # Extract feature importance from Random Forest
                rf_model = results['models']['Random Forest'].estimators_[0]
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': rf_model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig_importance = px.bar(
                    importance_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title="Feature Importance (Random Forest)",
                    labels={'Importance': 'Relative Importance'}
                )
                st.plotly_chart(fig_importance, use_container_width=True)
            except:
                st.info("Feature importance visualization is available for tree-based models like Random Forest.")
    
    else:  # Historical Analysis mode
        display_historical_analysis(df, user_inputs, target_names)
    
    # Always display model performance
    display_model_performance(results)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Dashboard Information:**
    - Data Source: UCI Air Quality Dataset
    - Models: Linear Regression, Random Forest, XGBoost, Neural Network
    - Metrics: RMSE (Root Mean Square Error), R² (Coefficient of Determination)
    - Built with: Streamlit, Plotly, Scikit-learn, XGBoost
    """)

# ==============================================
# EXECUTION
# ==============================================

if __name__ == "__main__":
    main()
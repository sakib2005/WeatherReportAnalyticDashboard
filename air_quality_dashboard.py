"""
Streamlit Dashboard for Air Quality Prediction
Interactive dashboard to predict pollutant concentrations using pre-trained models.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
from datetime import datetime, timedelta
import io

# Set page configuration
st.set_page_config(
    page_title="Air Quality Predictor",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data(filepath='AirQualityUCI_cleaned.csv'):
    """
    Load and preprocess the air quality dataset.
    Uses caching to avoid reloading on every interaction.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
        
    Returns:
    --------
    pandas.DataFrame
        Preprocessed DataFrame
    """
    try:
        df = pd.read_csv(filepath)
        
        # Convert Date and Time to datetime - FIXED for European format (DD/MM/YYYY)
        try:
            # Use dayfirst=True for DD/MM/YYYY format
            df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)
        except Exception as e:
            # Fallback with mixed format
            st.warning(f"Date parsing issue: {e}. Using mixed format.")
            df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='mixed', dayfirst=True)
        
        df.set_index('DateTime', inplace=True)
        
        # Ensure numeric types
        numeric_cols = ['PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 
                        'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH',
                        'CO(GT)', 'NOx(GT)', 'NO2(GT)', 'C6H6(GT)']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows with missing targets
        df = df.dropna(subset=['CO(GT)', 'NOx(GT)', 'NO2(GT)', 'C6H6(GT)'])
        
        # Fill remaining NaNs with forward fill (time series)
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df
    
    except FileNotFoundError:
        st.error(f"Data file '{filepath}' not found. Please ensure it's in the correct directory.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_resource
def load_model_and_scaler():
    """
    Load the pre-trained model and scaler.
    Uses caching to avoid reloading on every interaction.
    
    Returns:
    --------
    tuple
        (model, scaler, metadata)
    """
    try:
        # Try to load metadata first to know which model file to load
        with open('model_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        best_model_name = metadata.get('best_model', 'RandomForest')
        
        # Construct expected filenames
        model_file = f'best_model_{best_model_name}.joblib'
        scaler_file = 'feature_scaler.joblib'
        
        # Load the model and scaler
        model = joblib.load(model_file)
        scaler = joblib.load(scaler_file)
        
        # Load performance metrics
        try:
            metrics_df = pd.read_csv('model_performance_metrics.csv')
            summary_df = pd.read_csv('model_performance_summary.csv', header=[0, 1])
            metadata['metrics'] = metrics_df
            metadata['summary'] = summary_df
        except:
            st.warning("Could not load performance metrics files.")
            metadata['metrics'] = None
            metadata['summary'] = None
        
        return model, scaler, metadata
    
    except FileNotFoundError as e:
        st.error(f"Model file not found: {str(e)}")
        st.info("Please run the model training script first to generate the required files.")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

def create_prediction_inputs():
    """
    Create sidebar sliders for manual prediction input.
    
    Returns:
    --------
    dict
        Dictionary of feature values
    """
    st.sidebar.header("📊 Manual Prediction Input")
    st.sidebar.markdown("Adjust sliders to predict pollutant concentrations")
    
    # Define feature ranges based on typical values
    # These could be made dynamic based on actual data ranges
    features = {
        'PT08.S1(CO)': {'min': 600, 'max': 2000, 'default': 1200, 'desc': 'Tin oxide sensor for CO'},
        'PT08.S2(NMHC)': {'min': 600, 'max': 2000, 'default': 1200, 'desc': 'Titanium dioxide sensor for NMHC'},
        'PT08.S3(NOx)': {'min': 100, 'max': 1500, 'default': 800, 'desc': 'Tungsten oxide sensor for NOx'},
        'PT08.S4(NO2)': {'min': 100, 'max': 2500, 'default': 1300, 'desc': 'Tungsten oxide sensor for NO2'},
        'PT08.S5(O3)': {'min': 200, 'max': 2500, 'default': 1200, 'desc': 'Indium oxide sensor for O3'},
        'T': {'min': -10, 'max': 40, 'default': 18, 'desc': 'Temperature (°C)'},
        'RH': {'min': 10, 'max': 100, 'default': 50, 'desc': 'Relative Humidity (%)'},
        'AH': {'min': 0.2, 'max': 2.5, 'default': 1.2, 'desc': 'Absolute Humidity'}
    }
    
    user_inputs = {}
    
    for feature, params in features.items():
        # Create slider with description
        value = st.sidebar.slider(
            label=f"{feature}",
            min_value=float(params['min']),
            max_value=float(params['max']),
            value=float(params['default']),
            step=0.1,
            help=params['desc']
        )
        user_inputs[feature] = value
    
    return user_inputs

def make_prediction(model, scaler, input_features, feature_names):
    """
    Make predictions using the trained model.
    
    Parameters:
    -----------
    model : trained model
        Pre-trained regression model
    scaler : StandardScaler
        Fitted scaler for feature normalization
    input_features : dict
        Dictionary of feature values
    feature_names : list
        List of feature names in correct order
        
    Returns:
    --------
    dict
        Dictionary of predictions
    """
    try:
        # Convert input to DataFrame in correct feature order
        input_df = pd.DataFrame([input_features])[feature_names]
        
        # Scale the input features
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        predictions = model.predict(input_scaled)
        
        # Map predictions to target names
        target_names = ['CO(GT)', 'NOx(GT)', 'NO2(GT)', 'C6H6(GT)']
        result = {target: predictions[0][i] for i, target in enumerate(target_names)}
        
        return result
    
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

def create_historical_date_filter(df):
    """
    Create date range selector for historical data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Historical data with DateTime index
        
    Returns:
    --------
    tuple
        (start_date, end_date)
    """
    st.sidebar.header("📅 Historical Data Filter")
    st.sidebar.markdown("Select date range for historical analysis")
    
    # Get min and max dates from data
    min_date = df.index.min().date()
    max_date = df.index.max().date()
    
    # Create date inputs with sensible defaults
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=min_date,
            min_value=min_date,
            max_value=max_date
        )
    
    with col2:
        end_date = st.date_input(
            "End Date",
            value=max_date,
            min_value=min_date,
            max_value=max_date
        )
    
    # Ensure start_date <= end_date
    if start_date > end_date:
        st.sidebar.warning("Start date must be before end date. Adjusting...")
        start_date, end_date = min(start_date, end_date), max(start_date, end_date)
    
    return start_date, end_date

def filter_historical_data(df, start_date, end_date):
    """
    Filter data based on selected date range.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Historical data
    start_date : datetime.date
        Start date
    end_date : datetime.date
        End date
        
    Returns:
    --------
    pandas.DataFrame
        Filtered data
    """
    # Convert dates to datetime for comparison
    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=1)  # Include entire end date
    
    # Filter data
    mask = (df.index >= start_dt) & (df.index <= end_dt)
    filtered_df = df.loc[mask].copy()
    
    return filtered_df

def create_visualizations(df, predictions=None):
    """
    Create interactive visualizations for historical data and predictions.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Historical data
    predictions : dict, optional
        Current prediction results
    """
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["📈 Pollutant Trends", "🌡️ Environmental Factors", "📊 Model Performance"])
    
    with tab1:
        # Pollutant concentration trends
        st.subheader("Pollutant Concentrations Over Time")
        
        # Select pollutants to display
        pollutant_cols = ['CO(GT)', 'NOx(GT)', 'NO2(GT)', 'C6H6(GT)']
        available_pollutants = [col for col in pollutant_cols if col in df.columns]
        
        if available_pollutants:
            # Create subplots
            fig = make_subplots(
                rows=len(available_pollutants), 
                cols=1,
                subplot_titles=available_pollutants,
                vertical_spacing=0.1
            )
            
            for i, pollutant in enumerate(available_pollutants, 1):
                # Add historical data trace
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df[pollutant],
                        mode='lines',
                        name=f'Historical {pollutant}',
                        line=dict(width=2),
                        showlegend=True if i == 1 else False
                    ),
                    row=i, col=1
                )
                
                # Add prediction marker if available
                if predictions and pollutant in predictions:
                    # Find the last date in the data
                    last_date = df.index.max()
                    fig.add_trace(
                        go.Scatter(
                            x=[last_date],
                            y=[predictions[pollutant]],
                            mode='markers',
                            name=f'Predicted {pollutant}',
                            marker=dict(size=12, color='red', symbol='star'),
                            showlegend=True if i == 1 else False
                        ),
                        row=i, col=1
                    )
            
            fig.update_layout(
                height=300 * len(available_pollutants),
                showlegend=True,
                hovermode='x unified'
            )
            
            # Update y-axis labels
            for i in range(1, len(available_pollutants) + 1):
                fig.update_yaxes(title_text="Concentration", row=i, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No pollutant data available for the selected period.")
    
    with tab2:
        # Environmental factors
        st.subheader("Environmental Factors Over Time")
        
        # Select environmental factors to display
        env_cols = ['T', 'RH', 'AH', 'PT08.S1(CO)', 'PT08.S2(NMHC)', 
                   'PT08.S3(NOx)', 'PT08.S4(NO2)', 'PT08.S5(O3)']
        available_env = [col for col in env_cols if col in df.columns]
        
        if available_env:
            # Let user select which factors to display
            selected_factors = st.multiselect(
                "Select environmental factors to display:",
                available_env,
                default=available_env[:3]  # Show first 3 by default
            )
            
            if selected_factors:
                fig = go.Figure()
                
                for factor in selected_factors:
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df[factor],
                            mode='lines',
                            name=factor,
                            line=dict(width=2)
                        )
                    )
                
                fig.update_layout(
                    height=500,
                    xaxis_title="Date",
                    yaxis_title="Value",
                    hovermode='x unified',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No environmental data available for the selected period.")
    
    with tab3:
        # Model performance
        st.subheader("Model Performance Comparison")
        
        try:
            # Load performance metrics
            metrics_df = pd.read_csv('model_performance_metrics.csv')
            
            # Display metrics table
            st.markdown("### Detailed Performance Metrics")
            
            # Pivot table for better visualization
            pivot_rmse = metrics_df.pivot_table(
                index='model', 
                columns='target', 
                values='rmse',
                aggfunc='mean'
            ).round(4)
            
            pivot_r2 = metrics_df.pivot_table(
                index='model', 
                columns='target', 
                values='r2',
                aggfunc='mean'
            ).round(4)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**RMSE Scores (Lower is Better)**")
                st.dataframe(pivot_rmse, use_container_width=True)
            
            with col2:
                st.markdown("**R² Scores (Higher is Better)**")
                st.dataframe(pivot_r2, use_container_width=True)
            
            # Create bar chart comparing models
            st.markdown("### Model Comparison Chart")
            
            # Calculate average metrics
            avg_metrics = metrics_df.groupby('model').agg({
                'rmse': 'mean',
                'r2': 'mean'
            }).reset_index()
            
            # Create bar chart
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Average RMSE", "Average R²"),
                specs=[[{"type": "bar"}, {"type": "bar"}]]
            )
            
            # RMSE bars (lower is better)
            fig.add_trace(
                go.Bar(
                    x=avg_metrics['model'],
                    y=avg_metrics['rmse'],
                    name='RMSE',
                    marker_color='lightcoral',
                    text=avg_metrics['rmse'].round(4),
                    textposition='auto'
                ),
                row=1, col=1
            )
            
            # R² bars (higher is better)
            fig.add_trace(
                go.Bar(
                    x=avg_metrics['model'],
                    y=avg_metrics['r2'],
                    name='R²',
                    marker_color='lightblue',
                    text=avg_metrics['r2'].round(4),
                    textposition='auto'
                ),
                row=1, col=2
            )
            
            fig.update_layout(
                height=400,
                showlegend=False
            )
            
            fig.update_yaxes(title_text="RMSE", row=1, col=1)
            fig.update_yaxes(title_text="R²", row=1, col=2)
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.warning(f"Could not load performance metrics: {str(e)}")
            st.info("Please run the model training script to generate performance metrics.")

def create_data_export(df, predictions=None):
    """
    Create data export functionality.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Data to export
    predictions : dict, optional
        Prediction results to include
    """
    st.sidebar.header("📥 Data Export")
    
    # Export filtered historical data
    if not df.empty:
        csv = df.to_csv()
        st.sidebar.download_button(
            label="📊 Download Filtered Data (CSV)",
            data=csv,
            file_name=f"air_quality_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            help="Download the currently filtered historical data"
        )
    
    # Export prediction results
    if predictions:
        # Create DataFrame from predictions
        pred_df = pd.DataFrame([predictions])
        
        # Add input features if available
        if 'input_features' in st.session_state:
            input_df = pd.DataFrame([st.session_state.input_features])
            pred_df = pd.concat([input_df, pred_df], axis=1)
        
        pred_csv = pred_df.to_csv(index=False)
        st.sidebar.download_button(
            label="🔮 Download Prediction (CSV)",
            data=pred_csv,
            file_name=f"air_quality_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            help="Download the current prediction results"
        )

def display_prediction_results(predictions, metadata):
    """
    Display prediction results in an organized manner.
    
    Parameters:
    -----------
    predictions : dict
        Prediction results
    metadata : dict
        Model metadata
    """
    st.subheader("🎯 Real-Time Prediction Results")
    
    if predictions:
        # Create columns for better layout
        cols = st.columns(4)
        
        # Define pollutant units and descriptions
        pollutant_info = {
            'CO(GT)': {'unit': 'mg/m³', 'desc': 'Carbon Monoxide'},
            'NOx(GT)': {'unit': 'ppb', 'desc': 'Nitrogen Oxides'},
            'NO2(GT)': {'unit': 'μg/m³', 'desc': 'Nitrogen Dioxide'},
            'C6H6(GT)': {'unit': 'μg/m³', 'desc': 'Benzene'}
        }
        
        # Display each prediction in a metric card
        for i, (pollutant, value) in enumerate(predictions.items()):
            with cols[i % 4]:
                info = pollutant_info.get(pollutant, {'unit': '', 'desc': pollutant})
                st.metric(
                    label=info['desc'],
                    value=f"{value:.3f}",
                    help=f"{pollutant} concentration in {info['unit']}"
                )
        
        # Display model information
        with st.expander("Model Information"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Best Model:**")
                st.code(metadata.get('best_model', 'Unknown'))
            
            with col2:
                st.write("**Features:**")
                st.write(f"{metadata.get('n_features', 0)} features")
            
            with col3:
                st.write("**Training Date:**")
                st.write(metadata.get('training_date', 'Unknown'))
        
        # Add prediction timestamp
        st.caption(f"Prediction made at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    else:
        st.warning("No predictions available. Adjust the sliders to generate predictions.")

def main():
    """
    Main function to run the Streamlit dashboard.
    """
    # Header
    st.title("🌫️ Air Quality Prediction Dashboard")
    st.markdown("""
    Predict pollutant concentrations using sensor data and visualize historical trends.
    Adjust the sliders in the sidebar to make real-time predictions.
    """)
    
    # Initialize session state for predictions
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None
    if 'input_features' not in st.session_state:
        st.session_state.input_features = None
    
    # Load data and model
    with st.spinner("Loading data and model..."):
        df = load_data()
        model, scaler, metadata = load_model_and_scaler()
    
    if df is None or model is None or scaler is None:
        st.error("Failed to load required data or model. Please check the setup.")
        return
    
    # Sidebar controls
    st.sidebar.title("Dashboard Controls")
    
    # Create prediction inputs
    user_inputs = create_prediction_inputs()
    
    # Store input features in session state
    st.session_state.input_features = user_inputs
    
    # Create prediction button
    if st.sidebar.button("🚀 Make Prediction", type="primary", use_container_width=True):
        with st.spinner("Making prediction..."):
            # Make prediction
            feature_names = metadata.get('features', list(user_inputs.keys()))
            predictions = make_prediction(model, scaler, user_inputs, feature_names)
            st.session_state.predictions = predictions
    
    # Historical data filter
    if df is not None and not df.empty:
        start_date, end_date = create_historical_date_filter(df)
        
        # Filter historical data
        filtered_df = filter_historical_data(df, start_date, end_date)
        
        # Display date range info
        st.sidebar.markdown(f"**Date Range:** {start_date} to {end_date}")
        st.sidebar.markdown(f"**Samples:** {len(filtered_df):,} records")
    else:
        filtered_df = pd.DataFrame()
        start_date, end_date = None, None
    
    # Data export
    create_data_export(filtered_df, st.session_state.predictions)
    
    # Main content area
    if df is not None and not df.empty:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Display data summary
            st.subheader("📊 Data Summary")
            
            if not filtered_df.empty:
                # Display basic statistics
                stats_cols = st.columns(4)
                
                with stats_cols[0]:
                    st.metric("Total Records", len(filtered_df))
                
                with stats_cols[1]:
                    st.metric("Date Range", f"{(end_date - start_date).days} days")
                
                with stats_cols[2]:
                    avg_temp = filtered_df['T'].mean() if 'T' in filtered_df.columns else 0
                    st.metric("Avg Temp", f"{avg_temp:.1f}°C")
                
                with stats_cols[3]:
                    avg_rh = filtered_df['RH'].mean() if 'RH' in filtered_df.columns else 0
                    st.metric("Avg Humidity", f"{avg_rh:.1f}%")
                
                # Quick data preview
                with st.expander("Preview Filtered Data"):
                    st.dataframe(
                        filtered_df.head(10),
                        use_container_width=True,
                        column_config={
                            "DateTime": st.column_config.DatetimeColumn("Date Time")
                        }
                    )
            else:
                st.warning("No data available for the selected date range.")
        
        with col2:
            # Display current input values
            st.subheader("⚙️ Current Input Values")
            input_df = pd.DataFrame([user_inputs]).T.reset_index()
            input_df.columns = ['Feature', 'Value']
            st.dataframe(input_df, use_container_width=True, hide_index=True)
    
    # Display prediction results
    if metadata:
        display_prediction_results(st.session_state.predictions, metadata)
    
    # Create visualizations
    if df is not None and not df.empty:
        create_visualizations(filtered_df, st.session_state.predictions)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **About this dashboard:**  
    - Uses pre-trained machine learning models to predict air pollutant concentrations  
    - Visualizes historical air quality data  
    - Model trained on the AirQualityUCI dataset  
    - Built with Streamlit and Plotly
    """)

if __name__ == "__main__":
    main()
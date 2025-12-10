import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="NovaMart Marketing Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# Data loading with caching
@st.cache_data
def load_data():
    """Load all CSV files into memory"""
    data_folder = Path("data")
    
    data = {
        'campaign': pd.read_csv(data_folder / "campaign_performance.csv"),
        'customer': pd.read_csv(data_folder / "customer_data.csv"),
        'product': pd.read_csv(data_folder / "product_sales.csv"),
        'lead_scoring': pd.read_csv(data_folder / "lead_scoring_results.csv"),
        'feature_importance': pd.read_csv(data_folder / "feature_importance.csv"),
        'learning_curve': pd.read_csv(data_folder / "learning_curve.csv"),
        'geographic': pd.read_csv(data_folder / "geographic_data.csv"),
        'attribution': pd.read_csv(data_folder / "channel_attribution.csv"),
        'funnel': pd.read_csv(data_folder / "funnel_data.csv"),
        'journey': pd.read_csv(data_folder / "customer_journey.csv"),
        'correlation': pd.read_csv(data_folder / "correlation_matrix.csv")
    }
    
    # Convert date columns
    data['campaign']['date'] = pd.to_datetime(data['campaign']['date'])
    
    return data

# Load data
try:
    data = load_data()
    
    # Sidebar navigation
    st.sidebar.markdown("## 🎯 NovaMart Analytics")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigate to:",
        [
            "📈 Executive Overview",
            "📊 Campaign Analytics",
            "👥 Customer Insights",
            "🛍️ Product Performance",
            "🗺️ Geographic Analysis",
            "🔄 Attribution & Funnel",
            "🤖 ML Model Evaluation"
        ]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "**NovaMart Marketing Dashboard**\n\n"
        "Comprehensive analytics platform for monitoring marketing performance, "
        "customer behavior, and ML model effectiveness."
    )
    
    # Route to appropriate page
    if page == "📈 Executive Overview":
        from pages import executive_overview
        executive_overview.render(data)
    
    elif page == "📊 Campaign Analytics":
        from pages import campaign_analytics
        campaign_analytics.render(data)
    
    elif page == "👥 Customer Insights":
        from pages import customer_insights
        customer_insights.render(data)
    
    elif page == "🛍️ Product Performance":
        from pages import product_performance
        product_performance.render(data)
    
    elif page == "🗺️ Geographic Analysis":
        from pages import geographic_analysis
        geographic_analysis.render(data)
    
    elif page == "🔄 Attribution & Funnel":
        from pages import attribution_funnel
        attribution_funnel.render(data)
    
    elif page == "🤖 ML Model Evaluation":
        from pages import ml_evaluation
        ml_evaluation.render(data)

except FileNotFoundError:
    st.error(
        "⚠️ Data files not found! Please ensure all CSV files are in the 'data' folder.\n\n"
        "Required files:\n"
        "- campaign_performance.csv\n"
        "- customer_data.csv\n"
        "- product_sales.csv\n"
        "- lead_scoring_results.csv\n"
        "- feature_importance.csv\n"
        "- learning_curve.csv\n"
        "- geographic_data.csv\n"
        "- channel_attribution.csv\n"
        "- funnel_data.csv\n"
        "- customer_journey.csv\n"
        "- correlation_matrix.csv"
    )

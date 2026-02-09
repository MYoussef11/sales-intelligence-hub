import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
import time
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_settings

settings = get_settings()

# Config
API_URL = f"http://{settings.POSTGRES_SERVER}:8000" if os.environ.get("DOCKER_ENV") else "http://localhost:8000"
st.set_page_config(page_title=settings.APP_NAME, layout="wide")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard Overview", "Forecasting", "Lead Scoring", "Dealer Segments", "AI Assistant"])

# 1. Dashboard Overview
if page == "Dashboard Overview":
    st.title("Sales Intelligence & Automation Hub")
    st.markdown("### Executive Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Dealers", "124", "+2")
    col2.metric("Monthly Revenue", "€4.2M", "+12%")
    col3.metric("Avg Conversion", "3.4%", "-0.2%")
    col4.metric("Active Leads", "1,240", "+50")
    
    st.info("System Status: All ML Services are Online.")

# 2. Forecasting
elif page == "Forecasting":
    st.title("Sales Forecasting")
    st.markdown("Predictive revenue analytics per dealer.")
    
    dealer_id = st.number_input("Enter Dealer ID", min_value=1, value=1)
    
    if st.button("Generate Forecast"):
        with st.spinner("Running XGBoost Model..."):
            try:
                response = requests.get(f"{API_URL}/forecast/{dealer_id}")
                if response.status_code == 200:
                    data = pd.DataFrame(response.json())
                    data['date'] = pd.to_datetime(data['date'])
                    
                    fig = px.line(data, x='date', y='forecast', title=f"30-Day Revenue Forecast for Dealer {dealer_id}")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.dataframe(data)
                else:
                    st.error(f"Error: {response.text}")
            except Exception as e:
                st.error(f"Connection Error: {e}")

# 3. Lead Scoring
elif page == "Lead Scoring":
    st.title("Lead Scoring & Prioritization")
    
    col1, col2 = st.columns(2)
    with col1:
        source = st.selectbox("Lead Source", ["website", "referral", "email"])
        response_time = st.slider("Response Time (minutes)", 0, 120, 15)
        
    if st.button("Score Lead"):
        try:
            payload = {"source": source, "response_time_minutes": response_time}
            response = requests.post(f"{API_URL}/score_lead", json=payload)
            if response.status_code == 200:
                result = response.json()
                prob = result['conversion_probability']
                st.metric("Conversion Probability", f"{prob*100:.1f}%")
                
                if prob > 0.7:
                    st.success("High Priority Lead! Assign to Senior Rep.")
                elif prob > 0.3:
                    st.warning("Medium Priority.")
                else:
                    st.error("Low Priority. Automate follow-up.")
            else:
                st.error("API Error")
        except Exception as e:
            st.error(f"Connection Error: {e}")

# 4. Dealer Segments
elif page == "Dealer Segments":
    st.title("Dealer Segmentation")
    
    if st.button("Refresh Segments"):
        with st.spinner("Clustering Dealers..."):
            try:
                response = requests.get(f"{API_URL}/segments")
                if response.status_code == 200:
                    data = pd.DataFrame(response.json())
                    
                    fig = px.scatter(data, x='dealer_id', y='cluster', color='cluster', 
                                     title="Dealer Clusters (0: Standard, 1: High Value, 2: At Risk)")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("API Error")
            except Exception as e:
                st.error(f"Connection Error: {e}")

# 5. AI Assistant
elif page == "AI Assistant":
    st.title("Internal Sales Assistant (RAG)")
    
    question = st.text_input("Ask a question about policies or inventory:")
    
    if st.button("Ask Agent"):
        try:
            payload = {"question": question}
            response = requests.post(f"{API_URL}/agent/query", json=payload)
            if response.status_code == 200:
                answer = response.json()['answer']
                st.markdown(f"**Agent:** {answer}")
            else:
                st.error("API Error")
        except Exception as e:
            st.error(f"Connection Error: {e}")

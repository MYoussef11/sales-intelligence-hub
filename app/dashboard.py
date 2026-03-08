import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_settings

settings = get_settings()

# --- Sentry Error Tracking ---
import sentry_sdk
if settings.SENTRY_DSN:
    sentry_sdk.init(
        dsn=settings.SENTRY_DSN,
        traces_sample_rate=1.0,
        send_default_pii=False,
        environment=os.getenv("SENTRY_ENVIRONMENT", "development"),
    )

# Config
API_URL = f"http://{settings.API_SERVER}:8000"
st.set_page_config(
    page_title=settings.APP_NAME,
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .stMetric { background: #1e293b; padding: 16px; border-radius: 12px; border: 1px solid #334155; }
    .stMetric label { color: #94a3b8 !important; }
    .agent-badge-sql { background: #1e40af; color: #93c5fd; padding: 4px 12px; border-radius: 16px; font-size: 13px; font-weight: 600; }
    .agent-badge-rag { background: #065f46; color: #6ee7b7; padding: 4px 12px; border-radius: 16px; font-size: 13px; font-weight: 600; }
    div[data-testid="stSidebar"] { background: #0f172a; }
</style>
""", unsafe_allow_html=True)

# --- Plotly Theme ---
CHART_COLORS = ["#3b82f6", "#8b5cf6", "#06b6d4", "#f59e0b", "#ef4444", "#22c55e"]
CHART_TEMPLATE = "plotly_dark"

# --- Sidebar ---
st.sidebar.title("Sales Intelligence Hub")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["📊 Overview", "📈 Forecasting", "🎯 Lead Scoring", "🏢 Dealer Segments", "🤖 AI Assistant"],
    label_visibility="collapsed"
)

# Sidebar Filters
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Filters")

# Dealer filter
try:
    dealers_resp = requests.get(f"{API_URL}/dealers", timeout=3)
    dealers = dealers_resp.json() if dealers_resp.status_code == 200 else []
except:
    dealers = []

dealer_options = {d["dealer_name"]: d["dealer_id"] for d in dealers} if dealers else {}
selected_dealer = st.sidebar.selectbox(
    "Dealer",
    ["All Dealers"] + list(dealer_options.keys())
)
selected_dealer_id = dealer_options.get(selected_dealer) if selected_dealer != "All Dealers" else None

# ===================================
# PAGE 1: Overview
# ===================================
if page == "📊 Overview":
    st.title("Executive Dashboard")
    
    # Fetch live KPIs
    try:
        kpi_resp = requests.get(f"{API_URL}/analytics/summary", timeout=5)
        if kpi_resp.status_code == 200:
            kpis = kpi_resp.json()
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Dealers", f"{kpis['total_dealers']:,}")
            col2.metric("Monthly Revenue", f"${kpis['total_revenue']:,.0f}", f"{kpis['revenue_change_pct']:+.1f}%")
            col3.metric("Active Deals", f"{kpis['active_deals']:,}")
            col4.metric("Total Leads", f"{kpis['total_leads']:,}")
            
            st.markdown("---")
            
            # Charts row
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.subheader("Revenue Comparison")
                rev_data = pd.DataFrame({
                    "Period": ["Previous 30 Days", "Current 30 Days"],
                    "Revenue": [kpis["prev_revenue"], kpis["total_revenue"]]
                })
                fig = px.bar(rev_data, x="Period", y="Revenue", color="Period",
                           color_discrete_sequence=[CHART_COLORS[4], CHART_COLORS[0]],
                           template=CHART_TEMPLATE)
                fig.update_layout(showlegend=False, height=350)
                st.plotly_chart(fig, use_container_width=True)
            
            with col_right:
                st.subheader("Lead Pipeline")
                lead_data = pd.DataFrame({
                    "Status": ["Converted", "Unconverted"],
                    "Count": [kpis["total_leads"] - kpis["low_score_leads"], kpis["low_score_leads"]]
                })
                fig = px.pie(lead_data, names="Status", values="Count",
                           color_discrete_sequence=[CHART_COLORS[5], CHART_COLORS[4]],
                           template=CHART_TEMPLATE, hole=0.4)
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Failed to load KPIs. Is the backend running?")
    except Exception as e:
        st.error(f"Connection Error: {e}")
        st.info("Make sure the backend is running at " + API_URL)

# ===================================
# PAGE 2: Forecasting
# ===================================
elif page == "📈 Forecasting":
    st.title("Revenue Forecasting")
    st.markdown("XGBoost-based 30-day revenue prediction with time-series features.")
    
    col_input, col_info = st.columns([1, 2])
    
    with col_input:
        dealer_id = st.number_input("Dealer ID", min_value=1, value=selected_dealer_id or 1)
        run_forecast = st.button("🚀 Generate Forecast", use_container_width=True)
    
    if run_forecast:
        with st.spinner("Running XGBoost forecast model..."):
            try:
                response = requests.get(f"{API_URL}/forecast/{dealer_id}", timeout=30)
                if response.status_code == 200:
                    data = pd.DataFrame(response.json())
                    data['date'] = pd.to_datetime(data['date'])
                    
                    # Interactive forecast chart
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=data['date'], y=data['forecast'],
                        mode='lines+markers',
                        name='Forecast',
                        line=dict(color=CHART_COLORS[0], width=3),
                        marker=dict(size=5),
                        hovertemplate='%{x|%b %d}<br>Revenue: $%{y:,.0f}<extra></extra>'
                    ))
                    
                    fig.update_layout(
                        title=f"30-Day Revenue Forecast — Dealer #{dealer_id}",
                        xaxis_title="Date",
                        yaxis_title="Predicted Revenue ($)",
                        template=CHART_TEMPLATE,
                        height=450,
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Summary metrics
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total 30-Day Forecast", f"${data['forecast'].sum():,.0f}")
                    col2.metric("Peak Day", f"${data['forecast'].max():,.0f}")
                    col3.metric("Avg Daily Revenue", f"${data['forecast'].mean():,.0f}")
                    
                    with st.expander("📋 View Raw Data"):
                        st.dataframe(data.style.format({"forecast": "${:,.2f}"}), use_container_width=True)
                else:
                    st.error(f"Error: {response.text}")
            except Exception as e:
                st.error(f"Connection Error: {e}")

# ===================================
# PAGE 3: Lead Scoring
# ===================================
elif page == "🎯 Lead Scoring":
    st.title("Lead Scoring & Prioritization")
    st.markdown("Random Forest model predicting lead conversion probability.")
    
    col_form, col_result = st.columns([1, 1])
    
    with col_form:
        st.subheader("Score a Lead")
        source = st.selectbox("Lead Source", ["website", "referral", "email", "trade_show", "cold_call"])
        response_time = st.slider("Response Time (minutes)", 0, 120, 15)
        score_btn = st.button("⚡ Score Lead", use_container_width=True)
    
    if score_btn:
        try:
            payload = {"source": source, "response_time_minutes": response_time}
            response = requests.post(f"{API_URL}/score_lead", json=payload, timeout=10)
            if response.status_code == 200:
                result = response.json()
                prob = result['conversion_probability']
                
                with col_result:
                    st.subheader("Result")
                    st.metric("Conversion Probability", f"{prob*100:.1f}%")
                    
                    # Score gauge
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prob * 100,
                        number={"suffix": "%"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": CHART_COLORS[0]},
                            "steps": [
                                {"range": [0, 30], "color": "#7f1d1d"},
                                {"range": [30, 70], "color": "#78350f"},
                                {"range": [70, 100], "color": "#14532d"}
                            ],
                        }
                    ))
                    fig.update_layout(height=250, template=CHART_TEMPLATE)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    if prob > 0.7:
                        st.success("🔥 **High Priority Lead!** Assign to Senior Rep immediately.")
                    elif prob > 0.3:
                        st.warning("⚠️ **Medium Priority.** Schedule follow-up within 48 hours.")
                    else:
                        st.error("📉 **Low Priority.** Add to automated nurture sequence.")
            else:
                st.error("API Error")
        except Exception as e:
            st.error(f"Connection Error: {e}")

# ===================================
# PAGE 4: Dealer Segments
# ===================================
elif page == "🏢 Dealer Segments":
    st.title("Dealer Segmentation")
    st.markdown("K-Means clustering to categorize dealers by value and risk.")
    
    if st.button("🔄 Refresh Segments", use_container_width=False):
        with st.spinner("Running K-Means clustering..."):
            try:
                response = requests.get(f"{API_URL}/segments", timeout=15)
                if response.status_code == 200:
                    data = pd.DataFrame(response.json())
                    
                    cluster_labels = {0: "Standard", 1: "High Value", 2: "At Risk"}
                    data['segment'] = data['cluster'].map(cluster_labels)
                    
                    col_chart, col_dist = st.columns([2, 1])
                    
                    with col_chart:
                        fig = px.scatter(
                            data, x='dealer_id', y='cluster', color='segment',
                            color_discrete_map={"Standard": CHART_COLORS[0], "High Value": CHART_COLORS[5], "At Risk": CHART_COLORS[4]},
                            title="Dealer Cluster Assignment",
                            template=CHART_TEMPLATE,
                            height=400
                        )
                        fig.update_layout(yaxis_title="Cluster")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col_dist:
                        st.subheader("Distribution")
                        dist = data['segment'].value_counts()
                        for seg, count in dist.items():
                            pct = count / len(data) * 100
                            st.metric(seg, f"{count} dealers", f"{pct:.0f}%")
                else:
                    st.error("API Error")
            except Exception as e:
                st.error(f"Connection Error: {e}")

# ===================================
# PAGE 5: AI Assistant (Chat UI)
# ===================================
elif page == "🤖 AI Assistant":
    st.title("AI Sales Assistant")
    st.markdown("Ask questions about policies (RAG) or data (SQL). The system automatically routes to the right agent.")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                agent = msg.get("agent_type", "unknown")
                if agent == "sql":
                    st.markdown('<span class="agent-badge-sql">🧠 SQL Agent</span>', unsafe_allow_html=True)
                elif agent == "rag":
                    st.markdown('<span class="agent-badge-rag">📄 RAG Agent</span>', unsafe_allow_html=True)
            st.markdown(msg["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about policies, inventory, revenue, dealers..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = requests.post(
                        f"{API_URL}/agent/query",
                        json={"question": prompt},
                        timeout=30
                    )
                    if response.status_code == 200:
                        result = response.json()
                        answer = result.get("answer", "No answer generated.")
                        agent_type = result.get("agent_type", "unknown")
                        
                        # Show agent badge
                        if agent_type == "sql":
                            st.markdown('<span class="agent-badge-sql">🧠 SQL Agent</span>', unsafe_allow_html=True)
                        elif agent_type == "rag":
                            st.markdown('<span class="agent-badge-rag">📄 RAG Agent</span>', unsafe_allow_html=True)
                        
                        st.markdown(answer)
                        
                        # Save to history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "agent_type": agent_type
                        })
                    else:
                        st.error("API Error — couldn't reach the agent.")
                except Exception as e:
                    st.error(f"Connection Error: {e}")

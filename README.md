# Sales Intelligence & Automation Hub

A modular, scalable platform for B2B car dealership sales intelligence, featuring AI-driven forecasting, lead scoring, dealer segmentation, and a multi-agent system powered by LangChain and LangGraph.

## 🚀 Features

### Data & Analytics
-   **PostgreSQL Data Layer**: Normalized schema for Dealers, Inventory, Transactions, and Leads with realistic synthetic data generation (3 years of historical records).
-   **Revenue Forecasting**: XGBoost-based 30-day revenue prediction per dealer with time-series features (rolling means, quarterly patterns).
-   **Lead Scoring**: Random Forest model assigning conversion probabilities to leads, with detailed evaluation metrics (precision, recall, F1, classification report).
-   **Dealer Segmentation**: K-Means clustering to categorize dealers (High Value, Standard, At Risk), with silhouette score and cluster distribution analysis.

### Multi-Agent AI System
-   **Orchestrator**: LangGraph-based router that classifies queries and delegates to specialized agents, returning agent-type badges for UI display.
-   **RAG Agent**: Retrieval-Augmented Generation agent for answering policy, compliance, and warranty questions from an internal knowledge base.
-   **SQL Agent**: Secure natural language-to-SQL interface with read-only guardrails and query result limits.

### Application
-   **FastAPI Backend**: RESTful API serving all ML models, agent queries, live analytics, and lead management.
-   **Streamlit Dashboard**: Multi-page interactive UI (Overview, Forecasting, Lead Scoring, Dealer Segments, AI Assistant) with Plotly charts, dark theme, and agent-type badges.
-   **Lead Landing Page**: Customer-facing lead intake form served by FastAPI at `/landing`, with auto-assigned dealer routing.
-   **n8n Workflow Automation**: Event-driven workflows for lead scoring notifications and scheduled KPI alerts via Telegram.
-   **Dockerized Infrastructure**: Full-stack deployment with Docker Compose on a unified network.

## 🛠️ Getting Started

### Prerequisites
-   Docker & Docker Compose
-   Python 3.10+ (for local development)
-   OpenAI API Key

### Quick Start (Docker)
Run the entire stack with one command:
```bash
docker-compose up --build
```
Access the services:
-   **Dashboard**: http://localhost:8501
-   **API Docs**: http://localhost:8000/docs
-   **Lead Intake**: http://localhost:8000/landing
-   **n8n Workflows**: http://localhost:5678

### Local Development
1.  **Setup Environment**:
    ```bash
    python -m venv .venv
    .\.venv\Scripts\activate
    pip install -r requirements.txt
    ```

2.  **Configuration**:
    -   Copy `.env.example` to `.env`:
        ```bash
        copy .env.example .env
        ```
    -   Update `.env` with your credentials and API keys.

3.  **Start Database**:
    ```bash
    docker-compose up -d db
    ```

4.  **Generate Data & Train Models**:
    ```bash
    python scripts/generate_data.py
    python scripts/train_models.py
    ```

5.  **Run Services Locally**:
    ```bash
    # Backend API
    uvicorn app.main:app --reload

    # Dashboard (in a separate terminal)
    streamlit run app/dashboard.py
    ```

6.  **Test Landing Page**:
    Open http://localhost:8000/landing and submit a test lead.

## 📂 Project Structure
```
├── app/
│   ├── main.py             # FastAPI backend (API + lead management)
│   ├── dashboard.py        # Streamlit multi-page dashboard
│   └── templates/
│       └── landing.html    # Customer-facing lead intake form
├── ml_services/
│   ├── orchestrator.py     # LangGraph multi-agent router
│   ├── rag_agent.py        # RAG agent (FAISS + OpenAI)
│   ├── sql_agent.py        # Secure NL-to-SQL agent
│   ├── forecasting.py      # Revenue forecasting (XGBoost)
│   ├── lead_scoring.py     # Lead scoring model
│   └── segmentation.py     # Dealer segmentation (K-Means)
├── scripts/                # Data generation & model training
├── data/docs/              # Knowledge base for RAG agent
├── n8n_data/               # n8n persistent data (gitignored)
├── AGENTS.md               # AI agent instructions for codebase
├── config.py               # Centralized Pydantic configuration
├── docker-compose.yml      # Container orchestration
└── requirements.txt        # Python dependencies
```

## 🔒 Security
-   **SQL Guardrails**: Regex-based blocking of destructive SQL commands + read-only prompt engineering.
-   **Environment Variables**: All secrets managed via `.env` (excluded from version control).
-   **Sensitive Data**: Model files, CSVs, database dumps, and n8n data are excluded via `.gitignore`.

## ⚙️ Architecture
-   **Config**: Centralized in `config.py` using Pydantic Settings.
-   **Logging**: Structured logging across all services.
-   **Models**: Trained via `scripts/train_models.py` with evaluation metrics, persisted to `models/`.
-   **Docker Networking**: All services communicate on a unified `sales-net` bridge network.
-   **n8n Integration**: Backend triggers n8n webhooks for lead processing automation.

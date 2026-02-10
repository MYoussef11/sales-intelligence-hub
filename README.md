# Sales Intelligence & Automation Hub

A modular, scalable platform for B2B car dealership sales intelligence — featuring AI-driven forecasting, lead scoring, dealer segmentation, and a multi-agent system powered by LangChain and LangGraph.

## 🚀 Features

### Data & Analytics
-   **PostgreSQL Data Layer**: Normalized schema for Dealers, Inventory, Transactions, and Leads with realistic synthetic data generation (3–5 years of historical records).
-   **Revenue Forecasting**: XGBoost-based 30-day revenue prediction per dealer.
-   **Lead Scoring**: Random Forest model assigning conversion probabilities to leads.
-   **Dealer Segmentation**: K-Means clustering to categorize dealers (High Value, Standard, At Risk).

### Multi-Agent AI System
-   **Orchestrator**: LangGraph-based router that classifies queries and delegates to specialized agents.
-   **RAG Agent**: Retrieval-Augmented Generation agent for answering policy, compliance, and warranty questions from an internal knowledge base.
-   **SQL Agent**: Secure natural language-to-SQL interface with read-only guardrails and query result limits.

### Application
-   **FastAPI Backend**: RESTful API serving all ML models and agent queries.
-   **Streamlit Dashboard**: Interactive UI for visualizing forecasts, scores, segments, and chatting with the AI assistant.
-   **Dockerized Infrastructure**: Full-stack deployment with Docker Compose.

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

## 📂 Project Structure
```
├── app/                    # FastAPI backend & Streamlit dashboard
├── ml_services/            # ML models & AI agents
│   ├── orchestrator.py     # LangGraph multi-agent router
│   ├── rag_agent.py        # RAG agent (FAISS + OpenAI)
│   ├── sql_agent.py        # Secure NL-to-SQL agent
│   ├── forecasting.py      # Revenue forecasting
│   ├── lead_scoring.py     # Lead scoring model
│   └── segmentation.py     # Dealer segmentation
├── scripts/                # Data generation, training, export
├── data/docs/              # Knowledge base for RAG agent
├── config.py               # Centralized configuration
├── docker-compose.yml      # Container orchestration
└── requirements.txt        # Python dependencies
```

## 🔒 Security
-   **SQL Guardrails**: Regex-based blocking of destructive SQL commands + read-only prompt engineering.
-   **Environment Variables**: All secrets managed via `.env` (excluded from version control).
-   **Sensitive Data**: Model files, CSVs, and database dumps are excluded via `.gitignore`.

## ⚙️ Architecture
-   **Config**: Centralized in `config.py` using Pydantic Settings.
-   **Logging**: Structured logging across all services.
-   **Models**: Trained via `scripts/train_models.py`, persisted to `models/` for efficient loading.
-   **Docker Networking**: Services communicate via Docker's internal DNS (`db`, `backend`).

# Sales Intelligence & Automation Hub (POC)

A modular, production-ready platform for B2B car dealership sales, featuring AI forecasting, lead scoring, segmentation, and RAG-based assistance.

## 🚀 Features
-   **Data Layer**: PostgreSQL schema with synthetic data generation (3-5 years).
-   **ML Services**:
    -   **Forecasting**: XGBoost revenue prediction.
    -   **Lead Scoring**: Random Forest conversion probability.
    -   **Segmentation**: K-Means dealer clustering.
    -   **RAG Agent**: Policy helper (LangChain).
-   **Application**: FastAPI Backend + Streamlit Dashboard.
-   **Infrastructure**: Dockerized services.

## 🛠️ Getting Started

### Prerequisites
-   Docker & Docker Compose
-   Python 3.10+ (for local development)

### Quick Start (Docker)
Run the entire stack with one command:
```bash
docker-compose up --build
```
Access the services:
-   **Dashboard**: http://localhost:8501
-   **API Docs**: http://localhost:8000/docs
-   **n8n**: http://localhost:5678

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
    -   Update `.env` with your API keys (e.g., OpenAI).

3.  **Data Generation & Training**:
    -   Generate synthetic data:
        ```bash
        python scripts/generate_data.py
        ```
    -   Train ML models:
        ```bash
        python scripts/train_models.py
        ```
    -   Export data to CSV (optional):
        ```bash
        python scripts/export_data.py
        ```

4.  **Run Services**:
    -   Backend: `uvicorn app.main:app --reload`
    -   Dashboard: `streamlit run app/dashboard.py`

## 📂 Project Structure
-   `app/`: FastAPI backend and Streamlit dashboard.
-   `ml_services/`: Machine learning model logic.
-   `scripts/`: Utilities for data generation, training, and export.
-   `config.py`: Centralized configuration.
-   `docker-compose.yml`: Container orchestration.

## 🔧 Refactoring Notes
-   **Config**: All settings are in `config.py` and `.env`.
-   **Logging**: Centralized logging logic used throughout.
-   **Models**: Models are trained via `scripts/train_models.py` and saved to `models/` for efficient loading.

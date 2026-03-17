# AI-Powered Sales Intelligence Platform

A complete hackathon-ready prototype for SMEs to generate AI-driven sales predictions, actionable business insights, and interact with an AI Sales Assistant.

## Features
- **CSV Data Upload**: Seamless ingestion of historical sales data.
- **Multi-Model ML Pipeline**: Automatically trains Linear Regression, Random Forest, and XGBoost models, selecting the best performer based on R² score and MAE.
- **Dynamic Forecasting**: Predicts future sales for the next 30 days.
- **AI Business Insights**: Interprets the ML predictions and feature importance into plain English, actionable advice (e.g., "Sales Drop Predicted - Run Promotions").
- **AI Chat Assistant**: Interactive chatbot to ask questions about the data and predictions.
- **Explainability**: Visualizes which features (like 'Day of Week' or 'Historical Trends') had the biggest impact on the AI's decision.
- **Modern Dashboard**: A clean, responsive, dark-mode glassmorphic UI.

## Tech Stack
- **Backend**: Python, FastAPI, Pandas, Scikit-Learn, XGBoost
- **Frontend**: Vanilla HTML5, CSS3, JavaScript, Chart.js

## Project Structure
```text
ai_sales_platform/
├── backend/
│   ├── app.py             # FastAPI entrypoint
│   ├── ml_pipeline.py     # Data processing & multi-model training
│   └── ai_insights.py     # Insight generation & chat logic
├── data/
│   ├── generate_data.py   # Script to generate sample data
│   └── sample_sales.csv   # Example historical sales dataset
├── frontend/
│   ├── css/
│   │   └── style.css      # Dashboard styling
│   ├── js/
│   │   └── app.js         # API integration and Chart rendering
│   └── index.html         # Main UI
├── requirements.txt
└── README.md
```

## How to Run Locally

### 1. Requirements
Ensure you have Python 3.9+ installed.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate Sample Data (Optional)
If you need sample data to test the platform:
```bash
python data/generate_data.py
```
This will create `data/sample_sales.csv`.

### 4. Start the Server
Navigate to the `backend/` directory and run:
```bash
cd backend
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

### 5. Access the Dashboard
Open your web browser and go to:
[http://127.0.0.1:8000](http://127.0.0.1:8000)

Upload the `sample_sales.csv` using the sidebar widget to see the magic happen!

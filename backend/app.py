import os
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from ml_pipeline import SalesPredictor
from ai_insights import AIInsightsGenerator, ChatAssistant

app = FastAPI(title="AI Sales Platform")

# CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = SalesPredictor()
chat_assistant = ChatAssistant()

# Store latest pipeline results in memory for chat context
latest_results = None
latest_insights = None

class ChatRequest(BaseModel):
    message: str

@app.post("/api/upload")
async def upload_csv(file: UploadFile = File(...)):
    global latest_results, latest_insights
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed.")
        
    try:
        df = pd.read_csv(file.file)
        results = predictor.run_pipeline(df, forecast_days=30)
        insights = AIInsightsGenerator.generate_insights(results)
        
        # Update global state for chat
        latest_results = results
        latest_insights = insights
        chat_assistant.update_context(results, insights)
        
        return {
            "status": "success",
            "results": results,
            "insights": insights
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat(request: ChatRequest):
    if latest_results is None:
        return {"response": "Please upload a data file first so I can analyze your sales!"}
        
    reply = chat_assistant.reply(request.message)
    return {"response": reply}

# Serve static files from frontend
frontend_dir = os.path.join(os.path.dirname(__file__), '..', 'frontend')
os.makedirs(frontend_dir, exist_ok=True)
app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

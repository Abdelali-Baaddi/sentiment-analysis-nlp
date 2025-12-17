"""
FastAPI REST API for Sentiment Analysis
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from predict import SentimentPredictor

# Initialize FastAPI app
app = FastAPI(
    title="Sentiment Analysis API",
    description="API for sentiment analysis using machine learning",
    version="1.0.0"
)

# Load predictor
predictor = None

@app.on_event("startup")
async def load_model():
    """Load model on startup"""
    global predictor
    try:
        predictor = SentimentPredictor(
            model_path='../models/best_model.pkl',
            vectorizer_path='../models/tfidf_vectorizer.pkl'
        )
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")

# Request/Response models
class TextRequest(BaseModel):
    text: str
    
    class Config:
        schema_extra = {
            "example": {
                "text": "This movie was absolutely amazing!"
            }
        }

class BatchTextRequest(BaseModel):
    texts: List[str]
    
    class Config:
        schema_extra = {
            "example": {
                "texts": [
                    "This movie was great!",
                    "I hated this film.",
                    "It was okay, nothing special."
                ]
            }
        }

class PredictionResponse(BaseModel):
    text: str
    sentiment: str
    label: int
    confidence: float = None

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total: int

# API endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Sentiment Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "batch_predict": "/batch-predict",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model": "loaded"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: TextRequest):
    """
    Predict sentiment for a single text
    
    - **text**: The text to analyze
    
    Returns sentiment (positive/negative), label (0/1), and confidence score
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not request.text or request.text.strip() == "":
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        result = predictor.predict(request.text)
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/batch-predict", response_model=BatchPredictionResponse)
async def batch_predict(request: BatchTextRequest):
    """
    Predict sentiment for multiple texts
    
    - **texts**: List of texts to analyze
    
    Returns predictions for all texts
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not request.texts or len(request.texts) == 0:
        raise HTTPException(status_code=400, detail="Texts list cannot be empty")
    
    # Check for empty texts
    if any(not text or text.strip() == "" for text in request.texts):
        raise HTTPException(status_code=400, detail="All texts must be non-empty")
    
    try:
        results = predictor.predict_batch(request.texts)
        predictions = [PredictionResponse(**r) for r in results]
        return BatchPredictionResponse(
            predictions=predictions,
            total=len(predictions)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# Run with: uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
# backend/main.py

import pandas as pd
import re
import os
import traceback
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
from pydantic import BaseModel, Field, validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from logger import setup_logger

# --- Logging Setup ---
logger = setup_logger()

# --- Global variables ---
sentiment_analyzer = None
df = None


def load_model():
    """Load the sentiment analysis model (lazy loading)."""
    global sentiment_analyzer
    if sentiment_analyzer is None:
        model_path = 'sentiment_model.joblib'
        vectorizer_path = 'tfidf_vectorizer.joblib'
        
        # Check if custom model exists
        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            logger.info(f"Loading custom model from {model_path}...")
            try:
                import joblib
                model = joblib.load(model_path)
                vectorizer = joblib.load(vectorizer_path)
                
                # Create a wrapper to match the pipeline interface
                class CustomAnalyzer:
                    def __init__(self, model, vectorizer):
                        self.model = model
                        self.vectorizer = vectorizer
                    
                    def __call__(self, text):
                        # Handle pipeline input format (remove [CLS] etc if present)
                        if isinstance(text, str):
                            clean_text = text.replace('[CLS] ', '').replace(' [SEP]', '').split(' [SEP] ')[0]
                            # Preprocess using the same logic as training would be ideal, 
                            # but for now we rely on the vectorizer's preprocessing
                            features = self.vectorizer.transform([clean_text])
                            prediction = self.model.predict(features)[0]
                            return [{'label': prediction}]
                            
                sentiment_analyzer = CustomAnalyzer(model, vectorizer)
                logger.info("Custom model loaded successfully.")
                return sentiment_analyzer
            except Exception as e:
                logger.error(f"Failed to load custom model: {e}. Falling back to Hugging Face model.")
        
        # Fallback to Hugging Face model
        logger.info("Loading Aspect-Based Sentiment Analysis model (Hugging Face)...")
        # Use slow tokenizer to avoid tiktoken compatibility issues
        sentiment_analyzer = pipeline(
            "text-classification", 
            model="yangheng/deberta-v3-base-absa-v1.1",
            tokenizer="yangheng/deberta-v3-base-absa-v1.1",
            use_fast=False
        )
        logger.info("Hugging Face model loaded successfully.")
    return sentiment_analyzer


def load_data():
    """Load the tweets dataset (lazy loading)."""
    global df
    if df is None:
        try:
            df = pd.read_csv('Tweets.csv', usecols=['name', 'text'])
            logger.info(f"Loaded {len(df)} tweets from Tweets.csv")
        except FileNotFoundError:
            logger.error("Tweets.csv not found!")
            raise RuntimeError("Data file not found. Please ensure Tweets.csv exists in the backend directory.")
    return df


# --- FastAPI App Setup ---
app = FastAPI(
    title="Sentiment Analysis API",
    description="Aspect-based sentiment analysis for tweets",
    version="2.0.0"
)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS
origins = ["http://localhost:5173", "http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=[" *"],
)


# --- Pydantic Models ---
class AnalysisRequest(BaseModel):
    """Request model for sentiment analysis."""
    query: str = Field(..., min_length=1, max_length=100, description="Search query/keyword")
    
    @validator('query')
    def validate_query(cls, v):
        """Validate and sanitize query input."""
        # Remove leading/trailing whitespace
        v = v.strip()
        
        # Check not empty after strip
        if not v:
            raise ValueError('Query cannot be empty')
        
        # Basic SQL injection prevention
        dangerous_patterns = [';', '--', '/*', '*/', 'DROP', 'DELETE', 'INSERT', 'UPDATE']
        v_upper = v.upper()
        if any(pattern in v_upper for pattern in dangerous_patterns):
            raise ValueError('Invalid characters detected in query')
        
        return v

class ReviewRequest(BaseModel):
    """Request model for single CRM review sentiment analysis."""
    text: str = Field(..., min_length=1, description="The feedback review text to analyze")

class ReviewResponse(BaseModel):
    """Response model returning a strict sentiment label."""
    sentiment: str


class AnalysisResponse(BaseModel):
    """Response model for sentiment analysis."""
    query: str
    total_tweets_found: int
    results: dict
    tweets: list


# --- Helper Functions ---
def analyze_sentiment_for_aspect(text: str, aspect: str) -> str:
    """
    Analyze sentiment of text regarding a specific aspect.
    
    Uses DeBERTa-v3 model fine-tuned for aspect-based sentiment analysis.
    
    Args:
        text: Tweet text to analyze
        aspect: The aspect/keyword to analyze sentiment about
        
    Returns:
        Sentiment label ('positive', 'negative', or 'neutral')
        
    Example:
        >>> analyze_sentiment_for_aspect("Great flight!", "flight")
        'positive'
    """
    analyzer = load_model()
    result = analyzer(f"[CLS] {text} [SEP] {aspect} [SEP]")[0]
    return result['label'].lower()


# --- API Endpoints ---
@app.on_event("startup")
async def startup_event():
    """Log application startup and load resources."""
    logger.info("Application starting...")
    # Load model and data on startup
    load_model()
    load_data()
    logger.info("Application started successfully")


@app.get("/health")
def health_check():
    """
    Health check endpoint for monitoring.
    
    Returns:
        dict: Service health status
    """
    data = load_data()
    model_type = "custom" if hasattr(sentiment_analyzer, 'vectorizer') else "huggingface" if sentiment_analyzer else "none"
    
    return {
        "status": "healthy",
        "model_loaded": sentiment_analyzer is not None,
        "model_type": model_type,
        "data_loaded": data is not None,
        "total_tweets": len(data) if data is not None else 0
    }


@app.get("/analyze", response_model=AnalysisResponse)
@limiter.limit("60/minute")
def analyze_tweets(request: Request, query: str):
    """
    Analyze sentiment of tweets containing the specified query.
    
    Args:
        request: FastAPI request object (for rate limiting)
        query: Search keyword/phrase
        
    Returns:
        AnalysisResponse: Sentiment analysis results
        
    Raises:
        HTTPException: 400 for invalid input, 500 for server errors
    """
    try:
        # Validate input
        request_obj = AnalysisRequest(query=query)
        validated_query = request_obj.query
        
        logger.info(f"Analysis request received: query='{validated_query}'")
        
        # Load data
        data = load_data()
        
        # Filter tweets containing the query
        filtered_tweets = data[
            data['text'].fillna('').str.contains(
                rf'\b{re.escape(validated_query)}\b', 
                case=False, 
                regex=True
            )
        ]
        
        total_found = len(filtered_tweets)
        
        # Handle no results
        if total_found == 0:
            logger.warning(f"No tweets found for query: '{validated_query}'")
            return AnalysisResponse(
                query=validated_query,
                total_tweets_found=0,
                results={"positive": 0, "negative": 0, "neutral": 0},
                tweets=[]
            )
        
        # Sample tweets (max 200)
        tweet_sample = filtered_tweets.head(200)
        tweet_objects = tweet_sample.to_dict('records')
        
        # Analyze sentiment for each tweet
        sentiments_for_count = []
        for tweet in tweet_objects:
            try:
                sentiment = analyze_sentiment_for_aspect(tweet['text'], validated_query)
                tweet['sentiment'] = sentiment
                sentiments_for_count.append(sentiment)
            except Exception as e:
                logger.error(f"Error analyzing individual tweet: {e}")
                # Default to neutral on error
                tweet['sentiment'] = 'neutral'
                sentiments_for_count.append('neutral')
        
        # Count sentiments
        sentiment_counts = {
            "positive": sentiments_for_count.count('positive'),
            "negative": sentiments_for_count.count('negative'),
            "neutral": sentiments_for_count.count('neutral')
        }
        
        logger.info(
            f"Analysis complete: query='{validated_query}', "
            f"found={total_found}, analyzed={len(tweet_objects)}, "
            f"results={sentiment_counts}"
        )
        
        return AnalysisResponse(
            query=validated_query,
            total_tweets_found=total_found,
            results=sentiment_counts,
            tweets=tweet_objects
        )
        
    except ValueError as e:
        # Validation errors (400 Bad Request)
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        # Unexpected errors (500 Internal Server Error)
        logger.error(f"Unexpected error: {e}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500, 
            detail="An internal error occurred. Please try again later."
        )

@app.post("/predict_review", response_model=ReviewResponse)
def predict_review(request_data: ReviewRequest):
    """
    Evaluate a single review text and return its sentiment label.
    Designed for external CRM inter-service communication.
    """
    try:
        logger.info("CRM prediction request received.")
        
        # 1. Load the model (returns CustomAnalyzer wrapper or HF Pipeline)
        analyzer = load_model()
        
        # 2. Pass the text directly into the prediction flow
        # Note: If the HF ABSA model complains about missing aspect tokens in the future, 
        # you can format it here like: f"[CLS] {request_data.text} [SEP] general [SEP]"
        raw_result = analyzer(request_data.text)
        
        # 3. Extract the label from the returned array structure (e.g., [{'label': 'positive'}])
        raw_label = raw_result[0]['label']
        
        # 4. Enforce strict capitalization ("Positive", "Negative", "Neutral")
        sentiment_label = raw_label.strip().capitalize()
        
        # Safety enforcement
        valid_labels = ["Positive", "Negative", "Neutral"]
        if sentiment_label not in valid_labels:
            logger.warning(f"Unexpected label '{sentiment_label}' mapped to Neutral.")
            sentiment_label = "Neutral"
            
        return ReviewResponse(sentiment=sentiment_label)
        
    except Exception as e:
        logger.error(f"Error predicting CRM review sentiment: {e}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500, 
            detail="An internal error occurred during prediction."
        )
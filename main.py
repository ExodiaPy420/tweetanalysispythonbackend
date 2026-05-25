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
from deep_translator import GoogleTranslator

# --- Logging Setup ---
logger = setup_logger()

# --- Global variables ---
sentiment_analyzer = None
df = None
hf_fallback_cache = None


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
    """Response model returning a strict sentiment label and observability metrics."""
    sentiment: str
    model_used: str = Field(..., description="The name of the ML model that generated this prediction")
    confidence_score: float = Field(..., description="The probability/confidence score of the prediction (0.0 to 1.0)")


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
    Evaluate a single review text with Confidence-Based Routing.
    If the custom model is uncertain (< 0.70), routes to DeBERTa-v3 fallback.
    Returns the sentiment alongside observability metrics.
    """
    global hf_fallback_cache
    
    try:
        logger.info("CRM prediction request received.")
        
        try:
            # Automatically detect Spanish (or any language) and translate to English
            processed_text = GoogleTranslator(source='auto', target='en').translate(request_data.text)
            logger.info(f"Translated input: '{request_data.text}' -> '{processed_text}'")
        except Exception as e:
            logger.warning(f"Translation failed: {e}. Proceeding with original text.")
            processed_text = request_data.text
        
        # 1. Load the primary model (often CustomAnalyzer)
        analyzer = load_model()
        
        # Initialize default observability metrics
        model_used = "Unknown"
        confidence_score = 0.0
        
        # Determine if we are wrapping the local Scikit-Learn model
        is_custom_model = hasattr(analyzer, 'model') and hasattr(analyzer, 'vectorizer')
        
        if is_custom_model:
            # 1. Immediately calculate the features vector
            clean_text = processed_text.replace('[CLS] ', '').replace(' [SEP]', '').split(' [SEP] ')[0]
            features = analyzer.vectorizer.transform([clean_text])
            
            # Flag to determine if we escalate to Hugging Face
            route_to_fallback = False
            
            # 2. OOV Check First
            if features.nnz == 0:
                logger.warning("Out-of-Vocabulary (OOV) detected: features.nnz == 0. Routing to DeBERTa fallback.")
                route_to_fallback = True
            else:
                # 3. Features exist. Check for probability metrics.
                if hasattr(analyzer.model, 'predict_proba'):
                    # Extract the highest probability
                    probabilities = analyzer.model.predict_proba(features)[0]
                    confidence = max(probabilities)
                    
                    logger.info(f"Custom model confidence scored at: {confidence:.4f}")
                    
                    # Threshold Logic
                    if confidence >= 0.70:
                        raw_result = analyzer(processed_text)
                        sentiment_label = raw_result[0]['label']
                        model_used = "Custom_ScikitLearn"
                        confidence_score = float(confidence)
                        logger.info("Confidence optimal. Proceeding with Custom Model prediction.")
                    else:
                        logger.warning(f"Low confidence ({confidence:.4f}) detected, routing to DeBERTa fallback.")
                        route_to_fallback = True
                else:
                    # Model lacks predict_proba (Standard SVM), but we know nnz > 0
                    logger.info("Custom model lacks predict_proba, but recognized vocabulary. Proceeding.")
                    raw_result = analyzer(processed_text)
                    sentiment_label = raw_result[0]['label']
                    model_used = "Custom_ScikitLearn"
                    confidence_score = 1.0  # Fallback assumption for models lacking proba with recognized inputs
            
            # 4. Handle Fallback Execution if flagged
            if route_to_fallback:
                # Instantiate / call Hugging Face Pipeline lazily
                if hf_fallback_cache is None:
                    logger.info("Initializing Hugging Face Fallback into cache...")
                    hf_fallback_cache = pipeline(
                        "text-classification", 
                        model="yangheng/deberta-v3-base-absa-v1.1",
                        tokenizer="yangheng/deberta-v3-base-absa-v1.1",
                        use_fast=False
                    )
                
                # HF ABSA expects an aspect token.
                fallback_result = hf_fallback_cache(f"[CLS] {processed_text} [SEP] general [SEP]")
                sentiment_label = fallback_result[0]['label']
                model_used = "DeBERTa_Fallback"
                confidence_score = float(fallback_result[0].get('score', 0.0))
        else:
            # We are already defaulted to Hugging Face natively because local .joblib files were missing
            fallback_result = analyzer(f"[CLS] {request_data.text} [SEP] general [SEP]")
            sentiment_label = fallback_result[0]['label']
            model_used = "DeBERTa_Fallback"
            confidence_score = float(fallback_result[0].get('score', 0.0))
            
        # --- Contract Enforcement ---
        final_label = str(sentiment_label).strip().capitalize()
        
        valid_labels = ["Positive", "Negative", "Neutral"]
        if final_label not in valid_labels:
            logger.warning(f"Unexpected label format '{final_label}' mapped to Neutral.")
            final_label = "Neutral"
            
        return ReviewResponse(
            sentiment=final_label,
            model_used=model_used,
            confidence_score=confidence_score
        )
        
    except Exception as e:
        logger.error(f"Error predicting CRM review sentiment: {e}")
        raise HTTPException(
            status_code=500, 
            detail="An internal error occurred during review prediction."
        )
# backend/main.py

import pandas as pd
import re
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline

# --- Model Loading ---
# This loads a model specifically fine-tuned for Aspect-Based Sentiment Analysis.
# The first time you run this, it will download the model (it may take a few minutes).
print("Loading Aspect-Based Sentiment Analysis model...")
sentiment_analyzer = pipeline("text-classification", model="yangheng/deberta-v3-base-absa-v1.1")
print("Model loaded successfully.")


# --- FastAPI App Setup ---
app = FastAPI()

origins = ["http://localhost:5173"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Data Loading ---
try:
    df = pd.read_csv('Tweets.csv', usecols=['name', 'text'])
except FileNotFoundError:
    print("Error: Tweets.csv not found.")
    exit()


def analyze_sentiment_for_aspect(text, aspect):
    result = sentiment_analyzer(f"[CLS] {text} [SEP] {aspect} [SEP]")[0]
    # The model returns 'positive', 'negative', 'neutral'. We'll map it to our format.
    return result['label'].lower()


# --- API Endpoint ---
@app.get("/analyze")
def analyze_tweets(query: str):
    if not query:
        return {"error": "Query parameter cannot be empty."}

    # regex for whole-word matching to find relevant tweets
    filtered_tweets = df[df['text'].fillna('').str.contains(rf'\b{re.escape(query)}\b', case=False, regex=True)]
    
    tweet_sample = filtered_tweets.head(200)
    tweet_objects = tweet_sample.to_dict('records')
    
    sentiments_for_count = []
    for tweet in tweet_objects:
        # For each tweet, analyze the sentiment ABOUT our search query
        sentiment = analyze_sentiment_for_aspect(tweet['text'], query)
        tweet['sentiment'] = sentiment
        sentiments_for_count.append(sentiment)

    if not tweet_objects:
        return {
            "query": query, "total_tweets_found": 0,
            "results": {"positive": 0, "negative": 0, "neutral": 0},
            "tweets": []
        }

    sentiment_counts = {
        "positive": sentiments_for_count.count('positive'),
        "negative": sentiments_for_count.count('negative'),
        "neutral": sentiments_for_count.count('neutral')
    }

    return {
        "query": query,
        "total_tweets_found": len(filtered_tweets),
        "results": sentiment_counts,
        "tweets": tweet_objects 
    }
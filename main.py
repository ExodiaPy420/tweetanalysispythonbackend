# main.py

# --- Part 1: Imports ---
# We import the necessary libraries.
# FastAPI is for creating the web server.
# Pandas is for loading and managing our data.
# NLTK is for the sentiment analysis.
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# --- Part 2: Initial Setup ---
# This line creates our main web application instance.
app = FastAPI()

origins = ["http://localhost:5173"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allow origins listed above
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# This initializes the VADER sentiment analyzer.
# The first time you use it, you might need to download the lexicon.
# We did this with the download_nltk.py script.
analyzer = SentimentIntensityAnalyzer()

# --- Part 3: Data Loading ---
# We load our dataset. It's important to handle potential errors,
# like if the file doesn't exist.
try:
    # We only load the 'text' column since that's all we need.
    df = pd.read_csv('Tweets.csv', usecols=['name','text'])
except FileNotFoundError:
    print("Error: Tweets.csv not found. Make sure it's in the 'backend' directory.")
    # If the file isn't found, we exit the program.
    exit()

# --- Part 4: The Core Sentiment Analysis Function ---
def analyze_sentiment(text: str) -> str:
    """
    Analyzes the sentiment of a given text using VADER.
    Returns 'positive', 'negative', or 'neutral'.
    """
    # The polarity_scores() method returns a dictionary with scores.
    scores = analyzer.polarity_scores(text)
    
    # The 'compound' score is a single value that summarizes the sentiment.
    # > 0.05 is generally positive.
    # < -0.05 is generally negative.
    # Between -0.05 and 0.05 is neutral.
    compound_score = scores['compound']
    
    if compound_score >= 0.05:
        return 'positive'
    elif compound_score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

# --- Part 5: The API Endpoints ---
# This is a "decorator" that tells FastAPI that the function below it
# should handle GET requests to the root URL ("/").
@app.get("/")
def read_root():
    return {"message": "Welcome to the Sentiment Analysis API. Use the /analyze endpoint."}

# This decorator handles GET requests to the "/analyze" URL.
# In backend/main.py, find and REPLACE the entire analyze_tweets function

@app.get("/analyze")
def analyze_tweets(query: str):
    """
    This endpoint now returns a list of tweet objects,
    each containing the author's name and the tweet text.
    """
    if not query:
        return {"error": "Query parameter cannot be empty."}

    # The 'name' column is now loaded, so we can use it.
    filtered_tweets = df[df['text'].fillna('').str.contains(query, case=False)]

    tweet_sample = filtered_tweets.head(200)

    # This is the key change: Convert the pandas DataFrame slice
    # into a list of dictionaries (records).
    tweet_objects = tweet_sample.to_dict('records')

    # We need the texts for analysis
    tweet_texts = [tweet['text'] for tweet in tweet_objects]

    if not tweet_texts:
        return {
            "query": query,
            "total_tweets_found": 0,
            "results": {"positive": 0, "negative": 0, "neutral": 0},
            "tweets": []
        }

    sentiments = [analyze_sentiment(tweet) for tweet in tweet_texts]

    sentiment_counts = {
        "positive": sentiments.count('positive'),
        "negative": sentiments.count('negative'),
        "neutral": sentiments.count('neutral')
    }

    # The 'tweets' key now contains our list of objects.
    return {
        "query": query,
        "total_tweets_found": len(filtered_tweets),
        "results": sentiment_counts,
        "tweets": tweet_objects 
    }
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
import re 

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

# Add this new helper function above your analyze_tweets function
def get_sentence_with_keyword(text, keyword):
    # A simple way to split text into sentences
    sentences = re.split(r'[.!?]', text)
    for sentence in sentences:
        # Find the sentence that contains the keyword (case-insensitive)
        if re.search(rf'\b{re.escape(keyword)}\b', sentence, re.IGNORECASE):
            return sentence
    # If no specific sentence is found (e.g., tweet is one sentence), return the whole text
    return text

# Now, replace your existing analyze_tweets function with this final, most intelligent version
@app.get("/analyze")
def analyze_tweets(query: str):
    """
    Final version: Uses whole-word matching for search and
    sentence-level analysis for more accurate sentiment.
    """
    if not query:
        return {"error": "Query parameter cannot be empty."}

    # 1. Use regex for whole-word matching
    filtered_tweets = df[df['text'].fillna('').str.contains(rf'\b{re.escape(query)}\b', case=False, regex=True)]
    
    tweet_sample = filtered_tweets.head(200)
    tweet_objects = tweet_sample.to_dict('records')
    
    sentiments_for_count = []
    for tweet in tweet_objects:
        # 2. Analyze sentiment on the relevant sentence, not the whole tweet
        relevant_text = get_sentence_with_keyword(tweet['text'], query)
        sentiment = analyze_sentiment(relevant_text)
        
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
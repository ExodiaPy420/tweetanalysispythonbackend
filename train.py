# backend/train.py

import pandas as pd
import re
import joblib
import nltk
from nltk.corpus import stopwords
import csv
import os
from datetime import datetime

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier # <-- NEW MODEL IMPORT
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE


def preprocess_text(text):
    try:
        stop_words = set(stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))
    text = re.sub(r'@[A-Za-z0-9_]+|https?:\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)


def log_experiment(run_id, model_type, params, metrics):
    log_file = 'training_log.csv'
    if not os.path.exists(log_file):
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['run_id', 'timestamp', 'model_type', 'model_params', 'accuracy', 'positive_precision', 'positive_recall', 'positive_f1'])
    
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            run_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            model_type, params, metrics['accuracy'],
            metrics['positive']['precision'], metrics['positive']['recall'],
            metrics['positive']['f1-score']
        ])
    print(f"Results for run {run_id} logged to {log_file}")


def train_model():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"RUN ID: {run_id}")
    print("Starting model training process...")

    
    df = pd.read_csv('Tweets.csv')
    df = df[['text', 'airline_sentiment']]
    df = df[df['airline_sentiment'] != 'neutral']
    df['clean_text'] = df['text'].apply(preprocess_text)
    X = df['clean_text']
    y = df['airline_sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_tfidf, y_train)

    
    print("Step 5: Training and tuning the RandomForestClassifier model...")
    
    
    model_type = "RandomForest"
    rfc = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [100, 200], # Number of trees in the forest
        'max_depth': [10, 20, None] # Maximum depth of the trees
    }
    
    # GridSearchCV 
    grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train_smote, y_train_smote)
    
    best_model = grid_search.best_estimator_
    print(f"Best RandomForest parameters: {grid_search.best_params_}")

    
    print("\nStep 6: Evaluating the best model...")
    y_pred = best_model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    
    print(f"\n--- {model_type} Performance ---")
    print(f"Accuracy on Test Set: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nStep 7: Logging the experiment...")
    metrics_to_log = { 'accuracy': report_dict['accuracy'], 'positive': report_dict['positive'] }
    log_experiment(run_id, model_type, grid_search.best_params_, metrics_to_log)

    print("\nStep 8: Saving the final artifacts...")
    joblib.dump(vectorizer, f'tfidf_vectorizer_{run_id}.joblib')
    joblib.dump(best_model, f'sentiment_model_{run_id}.joblib')
    print("Training complete. Artifacts saved with unique ID.")

if __name__ == '__main__':
    train_model()
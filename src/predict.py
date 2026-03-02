import pickle
from pathlib import Path

base_dir = Path(__file__).parent.parent
models_dir = base_dir / "models"
dev_data_dir = base_dir / "data" / "development"

with open(models_dir / "sentiment_model.pkl", 'rb') as f:
    sentiment_model = pickle.load(f)

with open(models_dir / "intent_keywords.pkl", 'rb') as f:
    intent_keywords = pickle.load(f)

with open(models_dir / "nmf_model.pkl", 'rb') as f:
    nmf_model = pickle.load(f)

with open(dev_data_dir / "tfidf_vectorizer.pkl", 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

def classify_intent(review):
    text = review.lower()
    scores = {intent: 0 for intent in intent_keywords}
    for intent, keywords in intent_keywords.items():
        scores[intent] = sum(text.count(keyword) for keyword in keywords)
    return max(scores, key=scores.get)

def predict_review(review):
    X_review = tfidf_vectorizer.transform([review])
    sentiment = sentiment_model.predict(X_review)[0]
    
    intent = classify_intent(review)
    
    nmf_features = nmf_model.transform(X_review)
    topic = nmf_features.argmax(axis=1)[0]
    
    return (sentiment, intent, topic)

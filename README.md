# Sentiment Analysis & Intent Classification System

A comprehensive Natural Language Processing (NLP) system for analyzing customer reviews to extract sentiment, intent, and topics. This project uses machine learning and rule-based approaches to classify reviews with a user-friendly Gradio interface.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Methods & Algorithms](#methods--algorithms)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Model Training](#model-training)
- [Results](#results)

## 🎯 Project Overview

This project analyzes customer reviews to provide three predictions:

1. **Sentiment Classification** (Positive/Negative/Neutral)
2. **Intent Detection** (Complaint/Refund/Delivery/General)
3. **Topic Modeling** (Topic1 Topic2 Topic3 Topic4 Topic5)

### Key Objectives

- Preprocess and clean text data
- Train and evaluate multiple ML models
- Develop an interactive web interface
- Provide production-ready predictions

## 📂 Project Structure

```
sentiment-analysis/
├── data/
│   ├── raw/                          # Original dataset
│   │   └── reviews.csv
│   ├── processed/                    # Cleaned and preprocessed data
│   │   └── processed.csv
│   └── development/                  # Train/test splits and vectorizers
│       ├── X_train_bow.npz           # Bag of Words training features
│       ├── X_test_bow.npz            # Bag of Words testing features
│       ├── X_train_tfidf.npz         # TF-IDF training features
│       ├── X_test_tfidf.npz          # TF-IDF testing features
│       ├── y_train.csv               # Training labels
│       ├── y_test.csv                # Testing labels
│       ├── bow_vectorizer.pkl        # Saved BoW vectorizer
│       └── tfidf_vectorizer.pkl      # Saved TF-IDF vectorizer
├── models/
│   ├── sentiment_model.pkl           # Trained Logistic Regression (TF-IDF)
│   ├── intent_keywords.pkl           # Intent classification keywords
│   ├── nmf_model.pkl                 # NMF topic model
│   ├── nb_bow.pkl                    # Naive Bayes (BoW) - baseline
│   ├── nb_tfidf.pkl                  # Naive Bayes (TF-IDF) - baseline
│   └── lr_bow.pkl                    # Logistic Regression (BoW)
├── notebooks/
│   ├── preprocessing.ipynb           # Data cleaning and preprocessing
│   ├── models.ipynb                  # Model training and evaluation
│   └── evaluation.ipynb              # Comprehensive model evaluation
├── src/
│   ├── download.py                   # Download data from Kaggle
│   ├── preprocess.py                 # Text preprocessing utilities
│   └── predict.py                    # Prediction functions
├── interface/
│   └── app.py                        # Gradio web interface
├── reports/
│   └── *.png, *.csv                  # Evaluation reports and visualizations
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## 🔧 Methods & Algorithms

### Text Preprocessing Pipeline

The preprocessing pipeline in [`src/preprocess.py`](src/preprocess.py) includes:

1. **Text Cleaning**: Remove special characters and extra whitespace
2. **Normalization**: Convert text to lowercase
3. **Tokenization**: Split text into individual words using NLTK
4. **Lemmatization**: Reduce words to their base form using WordNetLemmatizer
5. **Stopword Removal**: Remove common English stopwords (the, is, etc.)

```python
from preprocess import preprocess_text
cleaned = preprocess_text("The product is amazing!")
# Output: "product amazing"
```

### Feature Extraction

Two vectorization methods are used:

- **Bag of Words (BoW)**: CountVectorizer - counts word frequencies
- **TF-IDF**: TfidfVectorizer - weights words by importance

### Classification Models

| Model | Features | Accuracy | Purpose |
|-------|----------|----------|---------|
| Logistic Regression (TF-IDF) | TF-IDF | ~0.88-0.92 | **Primary sentiment classifier** |
| Logistic Regression (BoW) | BoW | ~0.85-0.88 | Baseline comparison |
| Naive Bayes (TF-IDF) | TF-IDF | ~0.82-0.86 | Baseline comparison |
| Naive Bayes (BoW) | BoW | ~0.80-0.84 | Baseline comparison |
| TextBlob (Rule-based) | Polarity scores | ~0.60-0.70 | Baseline comparison |

### Intent Classification

Rule-based keyword matching:

```python
intent_keywords = {
    'complaint': ['problem', 'issue', 'broken', 'defect', ...],
    'refund': ['refund', 'return', 'money', ...],
    'delivery': ['deliver', 'shipping', 'delayed', ...],
    'general': ['love', 'great', 'excellent', ...]
}
```

### Topic Modeling

Non-negative Matrix Factorization (NMF) with TF-IDF features extracts latent topics from reviews.

## 💻 Installation

### Prerequisites

- Python 3.8+
- pip or conda

### Step 1: Clone the Repository

```bash
git clone https://github.com/Muhammad-Huzzaifa/sentiment-analysis.git
cd sentiment-analysis
```

### Step 2: Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download Pre-trained Models

The models are pre-trained and included in the `models/` directory. If you need to retrain:

```bash
python notebooks/preprocessing.ipynb   # Preprocess data
python notebooks/models.ipynb           # Train models
```

## 🚀 Usage

### Option 1: Interactive Web Interface (Recommended)

```bash
python interface/app.py
```

Then open your browser to `http://localhost:7860`

**Features:**
- Paste or type a review
- Click "Analyze Review" or press Enter
- See sentiment, intent, and topic predictions instantly
- Pre-loaded example reviews for testing

### Option 2: Python API

```python
from src.predict import predict_review
from src.preprocess import preprocess_text

# Preprocess the review
review = "This product arrived late but I love it anyway!"
clean_review = preprocess_text(review)

# Get predictions
sentiment, intent, topic = predict_review(clean_review)

print(f"Sentiment: {sentiment}")      # positive
print(f"Intent: {intent}")             # general
print(f"Topic: {topic}")               # 0
```

### Option 3: Command Line

```bash
python src/predict.py "Your review text here"
```

## ✨ Features

### Sentiment Analysis
- **Classes**: Positive, Negative, Neutral
- **Method**: Logistic Regression with TF-IDF
- **Accuracy**: ~90%

### Intent Detection
- **Classes**: Complaint, Refund, Delivery, General
- **Method**: Rule-based keyword matching
- **Use Case**: Route reviews to appropriate departments

### Topic Modeling
- **Method**: Non-negative Matrix Factorization (NMF)
- **Topics**: Automatically extracted latent topics
- **Use Case**: Understand main themes in reviews

## 📊 Model Training

### Data Pipeline

```
Raw Data (reviews.csv)
         ↓
  Preprocessing
         ↓
  Train/Test Split (80/20)
         ↓
  Feature Extraction (BoW & TF-IDF)
         ↓
  Model Training
         ↓
  Evaluation & Visualization
```

### Training Steps

1. **Run preprocessing notebook**:
   ```bash
   jupyter notebook notebooks/preprocessing.ipynb
   ```

2. **Train models**:
   ```bash
   jupyter notebook notebooks/models.ipynb
   ```

3. **Evaluate results**:
   ```bash
   jupyter notebook notebooks/evaluation.ipynb
   ```

## 📈 Results

### Model Performance Summary

Metrics evaluated on test set:
- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1-Score (weighted)

### Visualizations Generated

- **Confusion Matrices**: For each model
- **Metrics Comparison**: Bar charts comparing all models
- **Classification Reports**: Detailed per-class metrics

All results saved in `reports/` directory as PNG images and CSV files.

## 📚 Dependencies

Key libraries used:

| Library | Purpose |
|---------|---------|
| pandas | Data manipulation |
| numpy | Numerical computations |
| scikit-learn | ML models & preprocessing |
| nltk | Natural language processing |
| scipy | Sparse matrix handling |
| matplotlib | Plotting and visualization |
| seaborn | Statistical visualizations |
| textblob | Sentiment baseline |
| gradio | Web interface |

Full list in [`requirements.txt`](requirements.txt)

## 🔄 Workflow

```
User Input (Review)
      ↓
Preprocessing (clean, tokenize, lemmatize)
      ↓
Feature Extraction (TF-IDF vectorization)
      ↓
Sentiment Prediction (Logistic Regression)
Intent Detection (Keyword matching)
Topic Extraction (NMF)
      ↓
Output (3 predictions)
```

## 📝 Example Usage

**Input Review:**
```
"The package arrived three days late and the product is damaged. 
I want a refund immediately!"
```

**Output:**
```
Sentiment: NEGATIVE
Intent: REFUND
Topic: 2
```

## 🐛 Troubleshooting

### Issue: Model files not found
**Solution**: Ensure you're running commands from the project root directory.

### Issue: Port 7860 already in use
**Solution**: Change port in `interface/app.py`:
```python
demo.launch(server_name="0.0.0.0", server_port=7861)
```

### Issue: NLTK data not found
**Solution**: Download required NLTK data:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```

## 📧 Contact & Attribution

**Author**: Muhammad Huzaifa  
**Assignment**: Sentiment Analysis Project

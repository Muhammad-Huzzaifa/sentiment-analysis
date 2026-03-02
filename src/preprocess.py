import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

def clean_text(text):
    if not text:
        return ""

    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def normalize_text(text):
    if not text:
        return ""

    return text.lower()

def tokenize_text(text):
    if not text:
        return []

    tokens = word_tokenize(text)
    return tokens

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    if tag.startswith('V'):
        return wordnet.VERB
    if tag.startswith('N'):
        return wordnet.NOUN
    if tag.startswith('R'):
        return wordnet.ADV
    return wordnet.NOUN

def lemmatize_tokens(tokens):
    if not tokens:
        return []

    lemmatizer = WordNetLemmatizer()
    pos_tags = nltk.pos_tag(tokens)
    lemmatized_tokens = [lemmatizer.lemmatize(token, get_wordnet_pos(pos)) for token, pos in pos_tags]
    return lemmatized_tokens

def stopword_removal(tokens):
    if not tokens:
        return []

    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return filtered_tokens

def preprocess_text(text):
    if not text:
        return ""

    cleaned_text = clean_text(text)
    normalized_text = normalize_text(cleaned_text)
    tokens = tokenize_text(normalized_text)
    lemmatized_tokens = lemmatize_tokens(tokens)
    filtered_tokens = stopword_removal(lemmatized_tokens)

    return ' '.join(filtered_tokens)
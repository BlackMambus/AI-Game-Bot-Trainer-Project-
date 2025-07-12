import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

nltk.download('stopwords')
from nltk.corpus import stopwords
import string

# Sample dataset
data = {
    "email": [
        "Meeting at 10am with the client",
        "Win a free iPhone now!",
        "Dinner plans tonight?",
        "Your Amazon order has shipped",
        "Urgent: Update your bank account info",
        "Team project deadline extended",
        "50% off on all shoes this weekend!"
    ],
    "label": [
        "Work",
        "Spam",
        "Personal",
        "Promotions",
        "Spam",
        "Work",
        "Promotions"
    ]
}

df = pd.DataFrame(data)

# Preprocessing function
def clean_text(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

df['cleaned'] = df['email'].apply(clean_text)

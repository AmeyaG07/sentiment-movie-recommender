import pandas as pd
import numpy as np
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, classification_report

# Download stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# -------------------------------
# 1. Load and Clean Data
# -------------------------------
df = pd.read_csv('data/IMDB_Dataset.csv')  # Make sure this path exists

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)            # remove HTML tags
    text = re.sub(r'[^a-zA-Z]', ' ', text)       # remove non-letter characters
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

df['clean_review'] = df['review'].apply(clean_text)

# -------------------------------
# 2. Encode Sentiment
# -------------------------------
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# -------------------------------
# 3. Feature Extraction with TF-IDF
# -------------------------------
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_review'])
y = df['sentiment']

# -------------------------------
# 4. Train/Test Split & Classifier
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

# -------------------------------
# 5. Evaluation
# -------------------------------
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# -------------------------------
# 6. Recommend Positively Rated Movies
# -------------------------------
def recommend_movies(query, top_n=5):
    query_clean = clean_text(query)
    query_vec = vectorizer.transform([query_clean])
    sentiment = model.predict(query_vec)[0]

    if sentiment == 0:
        print("Negative sentiment detected. No recommendations.")
        return []

    # Use cosine similarity on the TF-IDF vectors
    positive_df = df[df['sentiment'] == 1].copy()
    positive_vectors = vectorizer.transform(positive_df['clean_review'])
    similarities = cosine_similarity(query_vec, positive_vectors).flatten()
    
    top_indices = similarities.argsort()[-top_n:][::-1]
    recommendations = positive_df.iloc[top_indices][['review', 'sentiment']]
    
    print(f"\n✅ Recommendations based on your positive review:\n")
    for i, row in recommendations.iterrows():
        print(f"👉 {row['review'][:150]}...\n")
    
    return recommendations

# -------------------------------
# 7. Try a Sample Input
# -------------------------------
sample_input = "A heartwarming and beautifully acted movie with a powerful message."
recommend_movies(sample_input)

# mental_health_detector.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load data
df = pd.read_csv("mental_health_posts.csv")

# Check for nulls
df.dropna(subset=["text", "label"], inplace=True)

# Check if DataFrame is empty after filtering
if df.empty:
    print("No data available after filtering. Please check your CSV file for valid 'neutral' and 'depression' labels.")
    exit()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# Function to predict a new post
def predict_post(post):
    vec = vectorizer.transform([post])
    pred = model.predict(vec)[0]
    print(f"\n📝 Post: {post}\nPrediction: {'Depression' if pred == 1 else 'Neutral'}")

# Example usage
predict_post("I can't sleep. I feel so anxious and alone.")
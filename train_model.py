
import pandas as pd
import numpy as np
import re
import nltk
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

print("🟢 Starting Fake News Detector Trainer...\n")
try:
    data = pd.read_csv('train.csv')
except Exception as e:
    raise FileNotFoundError("❌ Could not find 'train.csv'. Please place it in the same folder.") from e

print("✅ Dataset loaded successfully!")
print("Columns in dataset:", list(data.columns))
print("Sample data:\n", data.head(3), "\n")

if not {'news', 'label'}.issubset(data.columns):
    raise KeyError("❌ Your dataset must contain 'news' and 'label' columns.")


# Convert to numeric if necessary
data['label'] = data['label'].replace({'FAKE': 0, 'REAL': 1, 'fake': 0, 'real': 1})
data['label'] = data['label'].astype(int)

def clean_text(text):
    """Lowercase, remove URLs, punctuation, and stopwords"""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    return " ".join(tokens)

print("🧹 Cleaning text (this may take a moment)...")
data['news'] = data['news'].fillna('').apply(clean_text)
print("✅ Text cleaning complete!\n")

X = data['news']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"📊 Training samples: {len(X_train)}")
print(f"📊 Testing samples:  {len(X_test)}\n")


print("🔢 Converting text to numerical features (TF-IDF)...")
vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
print("✅ Vectorization complete!\n")


print("🤖 Training Logistic Regression model...")
model = LogisticRegression(max_iter=2000, class_weight='balanced')
model.fit(X_train_vec, y_train)
print("✅ Model trained successfully!\n")


print("📈 Evaluating model...")
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Accuracy: {accuracy * 100:.2f}%\n")
print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))

with open('fake_news_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("\n💾 Model and vectorizer saved successfully!")
print("Files created: 'fake_news_model.pkl' and 'vectorizer.pkl'\n")

sample_text = ["The Indian Space Research Organisation announced a new mission to Mars."]
sample_vec = vectorizer.transform(sample_text)
pred = model.predict(sample_vec)[0]
print("🧠 Test Prediction:", "Real News ✅" if pred == 1 else "Fake News 🚫")

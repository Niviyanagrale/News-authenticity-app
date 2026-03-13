import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Load datasets properly
true = pd.read_csv("dataset/True.csv", encoding="ISO-8859-1", engine="python")
fake = pd.read_csv("dataset/Fake.csv", encoding="ISO-8859-1", engine="python")

# Label them correctly
true["label"] = "REAL"
fake["label"] = "FAKE"

# Take only rows where text is NOT empty
true = true[true["text"].notna() & (true["text"].str.strip() != "")]
fake = fake[fake["text"].notna() & (fake["text"].str.strip() != "")]

# Balance dataset
min_len = min(len(true), len(fake))
true = true.sample(min_len, random_state=42)
fake = fake.sample(min_len, random_state=42)

df = pd.concat([true, fake], ignore_index=True)
df = df.sample(frac=1, random_state=42)

X = df["text"].astype(str)
y = df["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# TF-IDF
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.75)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Logistic Regression
model = LogisticRegression(max_iter=2000, class_weight="balanced")
model.fit(X_train_vec, y_train)

# Save model
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

# TEST on known real news
sample = true.iloc[0]["text"]
sample_vec = vectorizer.transform([sample])
print("Prediction for real news:", model.predict(sample_vec)[0])

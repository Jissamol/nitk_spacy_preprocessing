from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Step 1: Define the dataset
texts = [
    'I love this movie',
    'This movie is terrible',
    'I really enjoyed this film',
    'This film is awful',
    'What a fantastic experience',
    'I hated this film',
    'This was a great movie',
    'The film was not good',
    'I am very happy with this movie',
    'I am disappointed with this film'
]

labels = [
    'positive',  # 1
    'negative',  # 0
    'positive',
    'negative',
    'positive',
    'negative',
    'positive',
    'negative',
    'positive',
    'negative'
]

# Step 2: Encode labels
y = [1 if label == 'positive' else 0 for label in labels]  # 1=positive, 0=negative

# Step 3: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    texts, y, test_size=0.3, random_state=42
)

# Step 4: Vectorize using TF-IDF
tfidf = TfidfVectorizer(stop_words='english', max_features=100)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Step 5: Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Step 6: Predict
y_pred = model.predict(X_test_tfidf)

# Step 7: Classification Report
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=["negative", "positive"]))

# Step 8: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\n Confusion Matrix:\n", cm)

# Step 9: Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["negative", "positive"],
            yticklabels=["negative", "positive"])
plt.title("Confusion Matrix - TF-IDF + Naive Bayes")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

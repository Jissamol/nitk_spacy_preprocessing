from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Dataset
texts = [
    'I love this movie',                     # positive
    'This movie is terrible',                # negative
    'I really enjoyed this film',            # positive
    'This film is awful',                    # negative
    'What a fantastic experience',           # positive
    'I hated this film',                     # negative
    'This was a great movie',                # positive
    'The film was not good',                 # negative
    'I am very happy with this movie',       # positive
    'I am disappointed with this film'       # negative
]

labels = [
    'positive', 'negative', 'positive', 'negative', 'positive',
    'negative', 'positive', 'negative', 'positive', 'negative'
]

# 2. Encode labels
y = [1 if label == 'positive' else 0 for label in labels]  # 1 = positive, 0 = negative

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(texts, y, test_size=0.3, random_state=42)

# 4. Vectorize text
vectorizer = CountVectorizer()
X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)

# 5. Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train_bow, y_train)

# 6. Predict
y_pred = model.predict(X_test_bow)

# 7. Classification report
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=["negative", "positive"]))

# 8. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\n📊 Confusion Matrix:\n", cm)

# 9. Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=["negative", "positive"], yticklabels=["negative", "positive"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


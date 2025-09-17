# Required Libraries
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Sample Dataset (Expanded)
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
]

labels = [1, 0, 1, 0, 1, 0, 1, 0, 1]  # Matches 9 texts

# --- Bag of Words (BoW) ---
vectorizer_bow = CountVectorizer()
X_bow = vectorizer_bow.fit_transform(texts)

# Stratified Split
X_train_bow, X_test_bow, y_train_bow, y_test_bow = train_test_split(
    X_bow, labels, test_size=0.3, random_state=42, stratify=labels
)

# Train SVM on BoW
svm_bow = SVC(kernel='linear')
svm_bow.fit(X_train_bow, y_train_bow)

# Predict and Evaluate
y_pred_bow = svm_bow.predict(X_test_bow)
print("=== Bag of Words (BoW) ===")
print("Accuracy:", accuracy_score(y_test_bow, y_pred_bow))
print(classification_report(y_test_bow, y_pred_bow, zero_division=1))


# --- TF-IDF ---
vectorizer_tfidf = TfidfVectorizer()
X_tfidf = vectorizer_tfidf.fit_transform(texts)

# Stratified Split
X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(
    X_tfidf, labels, test_size=0.3, random_state=42, stratify=labels
)

# Train SVM on TF-IDF
svm_tfidf = SVC(kernel='linear')
svm_tfidf.fit(X_train_tfidf, y_train_tfidf)

# Predict and Evaluate
y_pred_tfidf = svm_tfidf.predict(X_test_tfidf)
print("=== TF-IDF ===")
print("Accuracy:", accuracy_score(y_test_tfidf, y_pred_tfidf))
print(classification_report(y_test_tfidf, y_pred_tfidf, zero_division=1))

# === Confusion Matrix for BoW ===
cm_bow = confusion_matrix(y_test_bow, y_pred_bow)
plt.figure(figsize=(5, 4))
sns.heatmap(cm_bow, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title("Confusion Matrix - BoW")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# === Confusion Matrix for TF-IDF ===
cm_tfidf = confusion_matrix(y_test_tfidf, y_pred_tfidf)
plt.figure(figsize=(5, 4))
sns.heatmap(cm_tfidf, annot=True, fmt='d', cmap='Greens', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title("Confusion Matrix - TF-IDF")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

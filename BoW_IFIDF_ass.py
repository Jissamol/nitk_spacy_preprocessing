import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Load sample data (15 reviews)
# -------------------------------
df = pd.read_csv("IMDB Dataset.csv").head(15)
df = df[['review', 'sentiment']]
df.columns = ['text', 'label']
df['label'] = df['label'].map({'positive': 1, 'negative': 0})

# -------------------------------
# Train/Test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42)

# -------------------------------
# Vectorization
# -------------------------------
bow = CountVectorizer(stop_words='english')
tfidf = TfidfVectorizer(stop_words='english')

X_train_bow = bow.fit_transform(X_train)
X_test_bow = bow.transform(X_test)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# -------------------------------
# Train Models & Accuracy
# -------------------------------
model_bow = MultinomialNB().fit(X_train_bow, y_train)
model_tfidf = MultinomialNB().fit(X_train_tfidf, y_train)

acc_bow = accuracy_score(y_test, model_bow.predict(X_test_bow))
acc_tfidf = accuracy_score(y_test, model_tfidf.predict(X_test_tfidf))

# ✅ Explicit Accuracy Print
print("\n================== ACCURACY ==================")
print(f"Bag-of-Words Accuracy : {acc_bow:.2f}")
print(f"TF-IDF Accuracy       : {acc_tfidf:.2f}")
print("================================================\n")

# -------------------------------
# TF-IDF Word Cloud
# -------------------------------
tfidf_vocab = tfidf.get_feature_names_out()
tfidf_scores = X_train_tfidf.toarray().sum(axis=0)
tfidf_freq = dict(zip(tfidf_vocab, tfidf_scores))
tfidf_cloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(tfidf_freq)

plt.figure(figsize=(10, 5))
plt.imshow(tfidf_cloud, interpolation='bilinear')
plt.axis("off")
plt.title("TF-IDF Word Cloud")
plt.tight_layout()
plt.show()

# -------------------------------
# Cosine Similarity Heatmap (TF-IDF)
# -------------------------------
cosine_sim_matrix = cosine_similarity(X_train_tfidf)
doc_labels = [f"Doc{i+1}" for i in range(len(X_train))]

plt.figure(figsize=(8, 6))
sns.heatmap(cosine_sim_matrix, annot=True, fmt=".2f", cmap='coolwarm',
            xticklabels=doc_labels, yticklabels=doc_labels)
plt.title("TF-IDF Cosine Similarity (Train Documents)")
plt.tight_layout()
plt.show()

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from wordcloud import WordCloud # type: ignore
import seaborn as sns
import pandas as pd

# Sample text data
docs = [
    "TextBlob is a simple NLP tool.",
    "TF-IDF helps find important words.",
    "Bag of Words counts word frequency.",
    "NLP tools include BoW, TF-IDF, and Word Embeddings."
]

# ==== Bag of Words ====
bow = CountVectorizer()
bow_matrix = bow.fit_transform(docs)
bow_words = bow.get_feature_names_out()
bow_counts = bow_matrix.toarray().sum(axis=0)
bow_freq = dict(zip(bow_words, bow_counts))

# BoW Word Cloud
bow_cloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(bow_freq)
plt.imshow(bow_cloud, interpolation='bilinear')
plt.axis("off")
plt.title("Bag-of-Words Word Cloud")
plt.show()

# ==== TF-IDF ====
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(docs)
tfidf_words = tfidf.get_feature_names_out()
tfidf_scores = tfidf_matrix.toarray().sum(axis=0)
tfidf_freq = dict(zip(tfidf_words, tfidf_scores))

# TF-IDF Word Cloud
tfidf_cloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(tfidf_freq)
plt.imshow(tfidf_cloud, interpolation='bilinear')
plt.axis("off")
plt.title("TF-IDF Word Cloud")
plt.show()

# ==== Document Similarity ====
similarity = cosine_similarity(tfidf_matrix)
doc_labels = [f"Doc{i+1}" for i in range(len(docs))]

# Similarity Heatmap
sns.heatmap(similarity, annot=True, cmap='coolwarm', xticklabels=doc_labels, yticklabels=doc_labels)
plt.title("Document Similarity (TF-IDF)")
plt.show()

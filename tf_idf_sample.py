from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd

# Sample documents
docs = [
    "sam eats pizza after football",
    "pizza and burgers are delicious",
    "Ravi plays football on sunday",
    "Burgers and pizza after game",
    "she loves pizza and tennis"
]

# TF-IDF Vectorization
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(docs)
tfidf_words = tfidf.get_feature_names_out()
tfidf_array = tfidf_matrix.toarray()

# Plot setup: 1 row, N columns (one for each document)
num_docs = len(docs)
fig, axes = plt.subplots(1, num_docs, figsize=(5 * num_docs, 5))

# If only one document, axes is not a list — fix that
if num_docs == 1:
    axes = [axes]

# Generate word cloud for each document
for i, vector in enumerate(tfidf_array):
    word_scores = {
        word: score for word, score in zip(tfidf_words, vector) if score > 0
    }

    wc = WordCloud(width=600, height=400, background_color='white').generate_from_frequencies(word_scores)

    axes[i].imshow(wc, interpolation='bilinear')
    axes[i].axis("off")
    axes[i].set_title(f"Doc {i+1}", fontsize=14)

plt.tight_layout()
plt.suptitle("TF-IDF Word Clouds (One per Document)", fontsize=16, y=1.05)
plt.show()

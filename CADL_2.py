# -------------------------------
# Feature Extraction in NLP
# Bag-of-Words (BoW) & TF-IDF
# -------------------------------

# Import libraries
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd

# Example dataset (you can replace with tweets or movie reviews dataset)
corpus = [
    "I love this movie, it is fantastic!",
    "This film was terrible and boring.",
    "What a great and wonderful experience.",
    "The movie was bad, I did not enjoy it.",
    "Absolutely loved the storyline and the acting."
]

print("==== Dataset (Corpus) ====")
for i, doc in enumerate(corpus, 1):
    print(f"Doc {i}: {doc}")

# -------------------------------
# 1. Bag-of-Words (BoW)
# -------------------------------
print("\n==== Bag-of-Words Representation ====")

# Initialize CountVectorizer
bow_vectorizer = CountVectorizer()

# Fit and transform corpus
bow_matrix = bow_vectorizer.fit_transform(corpus)

# Convert to DataFrame for better visualization
bow_df = pd.DataFrame(bow_matrix.toarray(), columns=bow_vectorizer.get_feature_names_out())
print(bow_df)

# -------------------------------
# 2. TF-IDF Representation
# -------------------------------
print("\n==== TF-IDF Representation ====")

# Initialize TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform corpus
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

# Convert to DataFrame
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
print(tfidf_df.round(3))  # Round to 3 decimals for readability

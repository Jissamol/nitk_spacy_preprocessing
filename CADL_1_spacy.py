import spacy
from nltk.stem import PorterStemmer  # For stemming
import nltk

# Pretrained model
nlp = spacy.load("en_core_web_sm") # english, standared general, web data,small

# Initialize NLTK stemmer
stemmer = PorterStemmer()

# Download (only once if not yet done)
nltk.download('punkt')

# Input text
text = "An Air India Boeing 787 Dreamliner (VT-ANB), operating from Ahmedabad to London Gatwick airport, crashed shortly after takeoff. 169 Indians, 53 British nationals, seven Portuguese, and one Canadian national were on board."

# Process text
doc = nlp(text)

# Tokens
print("Tokens:")
tokens = [token.text for token in doc]
print(tokens)

# Stop word removal
filtered_tokens = [token for token in doc if not token.is_stop]
print("\nAfter Stopword Removal:")
print([token.text for token in filtered_tokens])

# Lemmatization
print("\nLemmatization:")
print([token.lemma_ for token in filtered_tokens])

# Stemming (use NLTK)
print("\nStemming:")
print([stemmer.stem(token.text) for token in filtered_tokens])

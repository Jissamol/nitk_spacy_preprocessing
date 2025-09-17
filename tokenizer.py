import nltk
nltk.download('punkt')

from nltk.tokenize import word_tokenize

text = "NLTK is working correctly now!"
tokens = word_tokenize(text)

print("Tokens:", tokens)

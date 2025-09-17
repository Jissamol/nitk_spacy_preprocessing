import spacy

# Load the small English model
nlp = spacy.load("en_core_web_sm")

# Your input text
text = "SpaCy is an amazing NLP library!"

# Process the text
doc = nlp(text)

# Print tokens
print("Tokens:")
for token in doc:
    print(token.text)

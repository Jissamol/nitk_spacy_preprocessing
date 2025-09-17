from textblob import TextBlob

# Example text
text = "I love TextBlob! It's wonderfully simple and useful."

# Create a TextBlob object
blob = TextBlob(text)
words = blob.words
print("Words:", words)

# Perform sentiment analysis
sentiment = blob.sentiment
print("Polarity:", sentiment.polarity)
print("Subjectivity:", sentiment.subjectivity)

blob = TextBlob("I hate rainy days.")
print(blob.sentiment)  # negative polarity

blob = TextBlob("The movie was okay, not great but not bad.")
print(blob.sentiment)  # neutral polarity

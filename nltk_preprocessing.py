import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Input text
text = "An Air India Boeing 787 Dreamliner (VT-ANB), operating from Ahmedabad to London Gatwick airport, crashed shortly after takeoff. 169 Indians, 53 British nationals, seven Portuguese, and one Canadian national were on board."

# Tokenize
tokens = word_tokenize(text)
print("Original Tokens:", tokens)

# Remove stopwords
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
print("After Stopword Removal:", filtered_tokens)

# Stemming
stemmer = PorterStemmer()
stemmed = [stemmer.stem(word) for word in filtered_tokens]
print("After Stemming:", stemmed)

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized = [lemmatizer.lemmatize(word) for word in filtered_tokens]
print("After Lemmatization:", lemmatized)

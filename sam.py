import re
import spacy

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Sample text
text = """
Contact us at support@example.com or john.doe123@company.co.in.
You can also reach out to +91-9876543210 or 9123456789.
Our events are scheduled on 12-05-2024 and 01-01-2025.
"""

# Process text with spaCy for tokenization
doc = nlp(text)
print("spaCy Tokens:")
for token in doc:
    print(token.text, end=' ')
print("\n")

# -------- REGEX PATTERNS --------

# 1. Email regex
email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
emails = re.findall(email_pattern, text)

# 2. Indian mobile number regex (with optional +91-)
phone_pattern = r'(?:\+91[-\s]?)?[6-9]\d{9}'
phones = re.findall(phone_pattern, text)

# 3. Date regex (dd-mm-yyyy)
date_pattern = r'\b(0[1-9]|[12][0-9]|3[01])-(0[1-9]|1[0-2])-(19|20)\d{2}\b'
raw_dates = re.findall(date_pattern, text)

# Reformat extracted tuples back to full dates
formatted_dates = ['-'.join(date) for date in raw_dates]

# -------- OUTPUT --------
print("Extracted Emails:", emails)
print("Extracted Phone Numbers:", phones)
print("Extracted Dates:", formatted_dates)

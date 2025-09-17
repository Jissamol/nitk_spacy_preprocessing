import nltk
from nltk import pos_tag, word_tokenize, RegexpParser
from nltk.tree import Tree

sentence = "Apple is looking at buying U.K. startup for $1 billion."
tokens = word_tokenize(sentence)
tags = pos_tag(tokens)

# Define a simple grammar
chunk_grammar = "NP: {<DT>?<JJ>*<NN.*>+}"
chunk_parser = RegexpParser(chunk_grammar)
chunk_tree = chunk_parser.parse(tags)

# Visualize (Jupyter) or print
chunk_tree.pretty_print()

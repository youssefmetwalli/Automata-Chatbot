from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import spacy
import string
import joblib

# Load the spaCy model
nlp = spacy.load('en_core_web_md')

# Function for text preprocessing
def preprocess(text):
    text = text.lower()
    return text.translate(str.maketrans('', '', string.punctuation))

# Sample training data
training_data = [
    ("What is a DFA?", "ask_dfa"),
    ("What is finite automata?", "ask_dfa"),
    ("Can you explain DFA?", "ask_dfa"),
    ("Can you explain deterministic finite automaton", "ask_dfa"),
    ("Tell me about Turing Machines", "ask_turing_machine"),
    ("What is a Turing Machine?", "ask_turing_machine"),
    ("Define Turing Machine", "ask_turing_machine"),
    ("What are regular expressions?", "ask_regex"),
    ("Can you explain regular expressions?", "ask_regex"),
    ("Define regular expression", "ask_regex"),
    ("What is Python?", "unknown"),
    ("What is the weather today?", "unknown"),
    ("How are you?", "unknown")
]

texts, labels = zip(*training_data)
preprocessed_texts = [preprocess(text) for text in texts]

# Build a scikit-learn pipeline
pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', LogisticRegression())
])

# Train the model
pipeline.fit(preprocessed_texts, labels)

# Save the trained model to disk
joblib.dump(pipeline, 'intent_classifier.pkl')
print("Model trained and saved successfully.")

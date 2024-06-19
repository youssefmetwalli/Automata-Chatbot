import json
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

# Function for text preprocessing
def preprocess(text):
    text = text.lower()
    return text.translate(str.maketrans('', '', string.punctuation))

# Load training data from a JSON file
def load_training_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    texts, labels = zip(*data)
    preprocessed_texts = [preprocess(text) for text in texts]
    return preprocessed_texts, labels

# Load the training data
texts, labels = load_training_data('training_data.json')

# Build a scikit-learn pipeline
pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', LogisticRegression())
])

# Train the model
pipeline.fit(texts, labels)

# Save the trained model to disk
joblib.dump(pipeline, 'intent_classifier.pkl')
print("Model trained and saved successfully.")

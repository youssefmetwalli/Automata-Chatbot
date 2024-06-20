from flask import Flask, request, render_template
import joblib
import string
import json
import re

# Load the trained intent classifier model
pipeline = joblib.load('intent_classifier.pkl')

# Flask app initialization
app = Flask(__name__, template_folder='../templates', static_folder='../static')

# Function for text preprocessing
def preprocess(text):
    text = text.lower()
    return text.translate(str.maketrans("", "", string.punctuation))

# Load response templates from a JSON file
def load_responses(file_path):
    with open(file_path, 'r') as file:
        responses = json.load(file)
    return responses

responses = load_responses('responses.json')

# Define keyword-intent mapping
keywords = {
    "dfa": "ask_dfa",
    "deterministic finite automaton": "ask_dfa",
    "turing machine": "ask_turing_machine",
    "regular expression": "ask_regex",
    "regex": "ask_regex"
}

# Tokenize user input and match keywords to intents
def get_intent_from_keywords(user_input):
    for keyword, intent in keywords.items():
        if re.search(r'\b' + re.escape(keyword) + r'\b', user_input):
            return intent
    return "unknown"

# Response function based on intent classification or keyword matching
def get_response(user_input):
    preprocessed_input = preprocess(user_input)
    intent = get_intent_from_keywords(preprocessed_input)

    if intent == "unknown":
        # Fallback to using the classifier if no keyword matched
        proba = pipeline.predict_proba([preprocessed_input])[0]
        predicted_intent_index = proba.argmax()
        confidence = proba[predicted_intent_index]
        predicted_intent = pipeline.classes_[predicted_intent_index]

        # Set a threshold for confidence
        threshold = 0.6
        if confidence < threshold:
            predicted_intent = "unknown"
        intent = predicted_intent
    else:
        confidence = 1.0  # Assuming full confidence for keyword matches

    print(f"Predicted intent: {intent}, confidence: {confidence}")  # Debug statement

    return responses.get(intent, "I don't understand. Can you ask something else?")

# Route for the chatbot
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_input = request.form["user_input"]
        response = get_response(user_input)
        return render_template("index.html", user_input=user_input, response=response)
    return render_template("index.html", user_input=None, response=None)

if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)

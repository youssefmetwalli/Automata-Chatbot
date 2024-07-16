from flask import Flask, request, render_template, jsonify
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

keywords = {
    "dfa": "ask_dfa",
    "Dfa": "ask_dfa",
    "deterministic finite automaton": "ask_dfa",
    "turing machine": "ask_turing_machine",
    "regular expression": "ask_regex",
    "regex": "ask_regex",
    "Ndfa": "ask_ndfa",
    "ndfa": "ask_ndfa"
}

# Define intent priorities (lower number means higher priority)
intent_priorities = {
    "ask_dfa_vs_ndfa": 1,
    "ask_ndfa": 2,
    "ask_dfa": 3,
    "ask_turing_machine": 4,
    "ask_regex": 5,
    "unknown": 6
}

def get_intent_from_keywords(user_input):
    found_intents = set()
    for keyword, intent in keywords.items():
        if re.search(r'\b'+ re.escape(keyword) + r'\b', user_input):
            found_intents.add(intent)
    
    if "ask_dfa" in found_intents and "ask_ndfa" in found_intents:
        return "ask_dfa_vs_ndfa"
    
    if not found_intents:
        return "unknown"
    
    # Find the intent with the highest priority
    highest_priority_intent = min(found_intents, key=lambda intent: intent_priorities[intent])
    return highest_priority_intent

# Response function based on intent classification
def get_response(user_input):
    preprocessed_input = preprocess(user_input)
    
    # Try to get intent from keywords
    predicted_intent = get_intent_from_keywords(preprocessed_input)
    
    if predicted_intent == "unknown":
        # If no keywords matched, use the classifier
        proba = pipeline.predict_proba([preprocessed_input])[0]
        predicted_intent_index = proba.argmax()
        confidence = proba[predicted_intent_index]
        predicted_intent = pipeline.classes_[predicted_intent_index]
        
        # Set a threshold for confidence
        threshold = 0.6
        if confidence < threshold:
            predicted_intent = "unknown"
    
        print(f"Predicted intent: {predicted_intent}, confidence: {confidence}")  # Debug statement

    return responses.get(predicted_intent, "I don't understand. Can you ask something else?")

# Route for the chatbot
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_input = request.form["user_input"]
        response = get_response(user_input)
        return jsonify(response=response)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)

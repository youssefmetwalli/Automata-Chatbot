# src/app.py
from flask import Flask, request, render_template, url_for
import joblib
import string

# Load the trained intent classifier model
pipeline = joblib.load('intent_classifier.pkl')

# Flask app initialization
app = Flask(__name__, template_folder='../templates', static_folder='../static')

# Function for text preprocessing
def preprocess(text):
    text = text.lower()
    return text.translate(str.maketrans("", "", string.punctuation))

# Response function based on intent classification
def get_response(user_input):
    preprocessed_input = preprocess(user_input)
    proba = pipeline.predict_proba([preprocessed_input])[0]
    predicted_intent_index = proba.argmax()
    confidence = proba[predicted_intent_index]
    predicted_intent = pipeline.classes_[predicted_intent_index]

    # Set a threshold for confidence
    threshold = 0.6
    if confidence < threshold:
        predicted_intent = "unknown"
    
    print(f"Predicted intent: {predicted_intent}, confidence: {confidence}")  # Debug statement

    responses = {
        "ask_dfa": "A DFA is a deterministic finite automaton.",
        "ask_turing_machine": "A Turing Machine is a theoretical machine.",
        "ask_regex": "A regular expression defines a search pattern.",
        "unknown": "I don't understand. Can you ask something else?"
    }
    return responses.get(predicted_intent, "I don't understand. Can you ask something else?")

# Route for the chatbot
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_input = request.form["user_input"]
        response = get_response(user_input)
        return render_template("index.html", user_input=user_input, response=response)
    return render_template("index.html", user_input=None, response=None)

if __name__ == "__main__":
    app.run(debug=True)

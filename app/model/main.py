
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load the intents from the JSON file
with open('model/intent.json') as file:
    intents = json.load(file)

# Prepare the training data
X = []  # Input sentences
y = []  # Corresponding intents

for intent in intents['intents']:
    for pattern in intent['patterns']:
        X.append(pattern)
        y.append(intent['tag'])

# Create the TF-IDF vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Create the logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Function to predict the intent of a given question
def predict_intent(question):
    question_vector = vectorizer.transform([question])
    intent = model.predict(question_vector)[0]
    return intent

# Function to get the response for a given intent
def get_response(intent):
    for intent_data in intents['intents']:
        if intent_data['tag'] == intent:
            responses = intent_data['responses']
            return responses[0]  # Return only the first response

# Example usage
while True:
    question = input("Enter your question: ")
    if question.lower() == "close":
        break
    intent = predict_intent(question)
    response = get_response(intent)
    print("Response:", response)

import json
import random
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Load intents
with open("intents.json") as file:
    data = json.load(file)

# Prepare training data
X_train = []
y_train = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        X_train.append(pattern)
        y_train.append(intent["tag"])

# Create and train model
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X_train, y_train)

# Load spaCy model for entity extraction
nlp = spacy.load("en_core_web_sm")

# Chat function
def chatbot():
    print("ChatBot: Hello! Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            print("ChatBot: Goodbye!")
            break

        # Predict intent
        predicted_tag = model.predict([user_input])[0]

        # Get random response for predicted intent
        for intent in data["intents"]:
            if intent["tag"] == predicted_tag:
                response = random.choice(intent["responses"])
                break

        # Entity extraction
        doc = nlp(user_input)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        if entities:
            print(f"ChatBot (Entities): {entities}")

        print(f"ChatBot: {response}")

# Run chatbot
if __name__ == "__main__":
    chatbot()

import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))  # rb -> reading in binary mode
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.keras')


# Clean up the sentence
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


# Convert the sentence into a bag of words (binary 0, 1 representation)
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


# Predict the class of the input sentence
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    # Sort by probability in descending order
    results.sort(key=lambda x: x[1], reverse=True)

    # Return a list of intents with probabilities
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results] or [{"intent": "noanswer", "probability": "0"}]


# Get the appropriate response
def get_response(intents_list, intents_json):
    if not intents_list:
        return "I'm sorry, I don't understand that."

    tag = intents_list[0]['intent']  # Safely access the top intent
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])  # Use random.choice for a single response
            break
    else:
        result = "I'm not sure how to respond to that."

    return result


print("GO! Bot is running!")

# Main loop for chatbot
while True:
    message = input("Type something: ")
    ints = predict_class(message)
    res = get_response(ints, intents)
    print(res)

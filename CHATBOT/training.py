import random  # to create random responses
import json  # for handling intents JSON file
import pickle  # for serialization
import numpy as np  # for numerical computations

import nltk  # Natural Language Toolkit
from nltk.stem import WordNetLemmatizer  # for reducing words to their base form
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents file
intents = json.loads(open('intents.json').read())

# Initialize lists
words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

# Tokenize and preprocess data
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)  # Tokenize the pattern
        words.extend(word_list)  # Add words to the vocabulary
        documents.append((word_list, intent['tag']))  # Append pattern and tag pair
        if intent['tag'] not in classes:
            classes.append(intent['tag'])  # Add tag to classes

# Lemmatize and remove duplicates
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(list(set(words)))
classes = sorted(set(classes))

# Save words and classes for future use
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Prepare training data
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_pattern = document[0]
    word_pattern = [lemmatizer.lemmatize(word.lower()) for word in word_pattern]
    for word in words:
        bag.append(1) if word in word_pattern else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

# Shuffle and convert to NumPy array
random.shuffle(training)
training = np.array(training, dtype=object)

# Split into features and labels
train_x = np.array([item[0] for item in training])  # Feature set
train_y = np.array([item[1] for item in training])  # Labels

# Build the model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile the model
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
hist=model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5)

# Save the model
model.save('chatbot_model.keras',hist)  # Recommended native Keras format

print("Training complete. Model saved as 'chatbot_model.model'")

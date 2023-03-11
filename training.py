import numpy as np
import json
import nltk
import pickle
import random


## nltk.stem is a library for stemming words. 
# Stemming is finding similarities between words with the same root words.
# For example, the root word of "running" is "run".

# The WordNetLemmatizer is a class that can be used to lemmatize words.

from nltk.stem import WordNetLemmatizer

# Keras is a library that can be used to create neural networks.
# Sequential is a class that can be used to create a sequential neural network.

from keras.models import Sequential

# Dense is a class that can be used to create a fully connected layer in a neural network.
# Activation is a class that can be used to create an activation function in a neural network.
# Dropout is a class that can be used to create a dropout layer in a neural network.

from keras.layers import Dense, Activation, Dropout

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

nltk.download('punkt')

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']


# Loop through each sentence in our intents patterns
# Tokenize each word in the sentence
# Add to the words list
# Add to documents in our corpus
for intent in intents['intent']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize, lower each word and remove duplicates
# Lemmatize - groups together the different inflected forms of a word so they can be analysed as a single item
nltk.download('wordnet')

#Here we are lemmatizing each word in the words list and then converting it to lowercase.

words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(list(set(words)))

# Sort classes
classes = sorted(list(set(classes)))

#pickle.dump is a function that can be used to save data to a file
#wb is a parameter that can be used to write bytes to a file.
#Here we are saving the words and classes to a file called words.pkl and classes.pkl

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Create our training data
training = []
output_empty = [0] * len(classes)

# Create an empty array for our output
for document in documents:
    # Initialize our bag of words
    bag = []
    # List of tokenized words for the pattern
    pattern_words = document[0]
    # Lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # Create our bag of words array with 1, if word match found in current pattern
    for word in words:
        bag.append(1) if word in pattern_words else bag.append(0)

    # Output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1

    training.append([bag, output_row])

# Shuffle our features and turn into np.array

random.shuffle(training)
training = np.array(training)

# Create train and test lists. X - patterns, Y - intents
# Here we are creating two lists, train_x and train_y and then we are splitting the training data into two lists.
# train_x contains the patterns and train_y contains the intents.
# We are using the numpy library to split the training data into two lists.
train_x = list(training[:,0])
train_y = list(training[:,1])


# Here we are creating a sequential neural network with three layers.
# The first layer has 128 neurons, the second layer has 64 neurons and the third layer has the number of neurons equal to the number of intents.
# The activation function for the first two layers is relu and the activation function for the third layer is softmax.
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Here we are compiling the model using the adam optimizer and the sparse_categorical_crossentropy loss function.
# the loss function is used to calculate the error between the predicted output and the actual output.
# the optimizer is used to update the weights of the neural network.
# the metrics parameter is used to calculate the accuracy of the model.

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Here we are fitting the model using the training data.
# The number of epochs is 200 and the batch size is 5.
# The epochs parameter is used to specify the number of times the model will be trained on the training data.
# The batch size parameter is used to specify the number of training examples in one forward/backward pass.
# The verbose parameter is used to specify the amount of information that will be displayed during the training process.
# The verbose parameter can be set to 0, 1 or 2.
# If the verbose parameter is set to 0, no information will be displayed during the training process.
# If the verbose parameter is set to 1, the progress bar will be displayed during the training process.
# If the verbose parameter is set to 2, the progress bar and the loss and accuracy will be displayed during the training process.

hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

# Here we are saving the model to a file called chatbot_model.h5
model.save('chatbot_model.h5', hist)
print("Model created!")
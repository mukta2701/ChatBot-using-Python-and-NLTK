## CHATBOT using Python

This is a Python chatbot built using Keras and NLTK libraries. The chatbot is trained on a JSON file containing various intents and responses. The model is built using a neural network architecture and is able to predict the class and probability of the user's input.

It can understand and respond to a variety of user messages based on pre-defined intents.

## REQUIREMENTS

-JSON
-pickle
-Python 3.x
-Keras 
    Keras is a library that can be used to create neural networks.
-Numpy
-NLTK (Natural Language Tool Kit)

##SETUP

-Clone this repository to your local machine.
-Install the required packages listed above using pip.

    pip install -r requirements.txt


-Run the 'chatbot.py' script to start the chatbot.

##USAGE

The chatbot is designed to respond to specific intents related to various topics such as time, date, and weather. The intents and their responses are stored in the intents.json file. The model is trained on this file to predict the class and probability of the user's input.

To use the chatbot, simply enter your message and press enter. The chatbot will predict the class and probability of your input, and then respond with an appropriate response from the intents.json file.

##FILE STRUCTURE

- 'chatbot.py' : Main Python script that runs the chatbot.

- 'intents.json': JSON file that contains the pre-defined intents and responses for the chatbot.

- 'words.pkl' : Pickle file that contains a list of all unique words in the intents.

- 'classes.pkl' : Pickle file that contains a list of all unique classes (intents) in the intents.

- 'chatbot_model.h5' : Keras model that was trained on the intents to classify user messages.

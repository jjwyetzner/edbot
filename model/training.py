import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lem = WordNetLemmatizer()
import random
import pickle
import json
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import gradient_descent_v2
import numpy as np

words = []
classes = []
documents = []
ignoredWords = ['?', '!']
training = open('model/training.json', encoding='utf-8').read()
intents = json.loads(training)

#tokenization
for individualIntent in intents['training']:
    for word in individualIntent['patterns']:
        tokenizedWord = nltk.word_tokenize(word)
        words.extend(tokenizedWord)
        documents.append((tokenizedWord, individualIntent['tag']))

        if individualIntent['tag'] not in classes:
            classes.append(individualIntent['tag'])

words = [lem.lemmatize(tokenizedWord.lower()) for tokenizedWord in words if tokenizedWord not in ignoredWords]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))
pickle.dump(words, open('model/words.pkl', 'wb'))
pickle.dump(classes,open('model/classes.pkl', 'wb'))

#model
training = []
emptyOutput = [0] * len(classes)

for individualDocument in documents:
    bagOfWords = [0]
    patternWords = individualDocument[0]
    patternWords = [lem.lemmatize(individualWord.lower()) for individualWord in patternWords]

    for word in words:
        bagOfWords.append(1) if word in patternWords else bagOfWords.append(0)

    outputRow = list(emptyOutput)
    outputRow[classes.index(individualDocument[1])] = 1
    training.append([bagOfWords, outputRow])

random.shuffle(training)
training = np.array(training)
trainPatterns = list(training[:,0])
trainIntents = list(training[:,1])

model = Sequential()
model.add(Dense(256, input_shape = (len(trainPatterns[0]),), activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(trainIntents[0]), activation = 'softmax'))
sgd = gradient_descent_v2.SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])
hist = model.fit(np.array(trainPatterns), np.array(trainIntents), epochs = 200, batch_size = 5, verbose = 1)
model.save('model/model.h5', hist)
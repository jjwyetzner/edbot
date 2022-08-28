from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
import uvicorn
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
lem = WordNetLemmatizer()
import random
import pickle
import json
import numpy as np
from keras.models import load_model

model = load_model('model/model.h5')
words = pickle.load(open('model/words.pkl', 'rb'))
classes = pickle.load(open('model/classes.pkl', 'rb'))

def cleanUpSentence(sentence):
    
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
import pkg_resources
from symspellpy import SymSpell, Verbosity
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

model = load_model('model/model.h5')
words = pickle.load(open('model/words.pkl', 'rb'))
classes = pickle.load(open('model/classes.pkl', 'rb'))
intents = json.loads(open('app/model/general/training.json', encoding='utf-8').read())
needinfo = ["I'm having trouble understanding. Could you tell me more? For more help, please see the resource pages linked above.", "Could you elaborate more on that?"]

def cleanUpSentence(sentence):
    originalWords = nltk.word_tokenize(sentence)
    sentenceWords = []
    
    for individualWord in originalWords:  
        suggestions = sym_spell.lookup(individualWord, Verbosity.CLOSEST, max_edit_distance=2, include_unknown=True)
        suggestion = str(suggestions[0])
        suggestion = suggestion.split(",", 1)
        sentenceWords.append(suggestion[0])

    sentenceWords = [lem.lemmatize(individualWord.lower()) for individualWord in sentenceWords]
    return [individualWord.lower() for individualWord in sentenceWords]

def bagOfWords(sentence, words):
    sentenceWords = cleanUpSentence(sentence)
    bag = [0]*len(words)

    for individualWord in sentenceWords:
        for x,y in enumerate(words):
            if y == individualWord:
                bag[x] = 1

    return(np.array(bag))

def getResponse(request):
    bag = bagOfWords(request, words)
    res = model.predict(np.array([bag]))[0]
    print(res)
    ERROR_THRESHOLD = 0.75

    results = [[i,r] for i,r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    print(results)

    ints = []
    for individualResult in results:
        ints.append({"intent": classes[individualResult[0]], "probability": str(individualResult[1])})
    if(ints == []):
        return random.choice(needinfo)
    else: 
        tag = ints[0]['intent']
        listOfIntents = intents['training']
        for i in listOfIntents:
            if(i['tag'] == tag):
                return random.choice(i['responses'])

    return random.choice(needinfo)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def main(request: Request):
    return templates.TemplateResponse("main.html", {"request": request})

@app.get("/home", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("main.html", {"request": request})

@app.get("/resources", response_class=HTMLResponse)
def resources(request: Request):
    return templates.TemplateResponse("resources.html", {"request": request})

@app.get("/tips", response_class=HTMLResponse)
def tips(request: Request):
    return templates.TemplateResponse("tips.html", {"request": request})

@app.get("/get")
def getBotResponse(msg: str):
    return str(getResponse(msg))

if __name__ == "__main__":
    uvicorn.run("main:app")
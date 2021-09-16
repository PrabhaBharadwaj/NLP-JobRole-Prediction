# -*- coding: utf-8 -*-
"""
Created on Wed sept 15 2021

@author: Prabhavathi
"""

# 1. Library imports
import uvicorn  #ASGI
from fastapi import FastAPI
import numpy as np
import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import re
import unicodedata
from pydantic import BaseModel
from joblib import load
import bz2



# 2. Class which describes resource_type
class resource_type(BaseModel):
    #id: float 
    job_title: str 
	
	
# 3. Create the app object
app = FastAPI()

# 4. Call the joblib model file
# This for not zipped file
#pipeline = load("text_classification.joblib")

# This for bz2 zipped file
with bz2.BZ2File("text_classification.joblib" + '.bz2', 'rb') as fo:  # doctest: +ELLIPSIS
	pipeline = load(fo)


# 5. Index route, opens automatically on http://127.0.0.1:8000
#@app is fastapi imported object app
@app.get('/')
def index():
    return {'message': 'Hello'}

# 6. Predict the output
@app.post('/predict')
def predict_role(data:resource_type):
	
	data = data.dict()
	# Convert data to Dataframe
	data_df = pd.DataFrame(list(pd.Series(data['job_title'])), columns = ['job_title'])
	

	# 7. Do preprocessing
	
	nlp = spacy.load('en_core_web_md')

	def make_to_base(x):
		x_list = []
		# TOKENIZATION
		doc = nlp(x)
    
		for token in doc:
			lemma = str(token.lemma_)
			if lemma == '-PRON-' or lemma == 'be':   
				lemma = token.text
			x_list.append(lemma)
		return(" ".join(x_list))


	def pre_process(X):
    # Lower case convertion
		X['job_title'] = X['job_title'].apply(lambda x: str(x).lower()) 
    
    # Digits Removal
		X['job_title'] = X['job_title'].apply(lambda x: re.sub('[^A-Z a-z # . ]+', '', x))
    
    # Stop word Removal
		X['job_title'] = X['job_title'].apply(lambda x: " ".join([t for t in x.split() if t not in STOP_WORDS]))
   
    # Unicodedata removal
		X['job_title'] = X['job_title'].apply(lambda x: unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('utf-8', 'ignore'))
    
    # Lemmatization
		X['job_title'] = X['job_title'].apply(lambda x: make_to_base(x))
    
    ##Single Character removal
		X['job_title']  = X['job_title'] .apply(lambda x: " ".join([t for t in x.split() if len(t) != 1]))
   
		return X
	

	# Call the preprocessing
	pre_data = pre_process(data_df)
	
	# predict the role	
	my_prediction =pipeline.predict(pre_data['job_title'])
	
	return my_prediction[0]
	


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    # Here we use unicorn to run the fastapi object app in local host 127.0.0.1 with port no 8000
    #@app is fastapi imported object app

#Below command used to run this app.py in cmd
    
#uvicorn app:app --reload
    # Here 1st app is filename app.py(If change file name to main, we need to change here also
    # 2nd app is FASTAPI Object name in this file

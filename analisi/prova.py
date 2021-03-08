import streamlit as st
import pandas as pd
import numpy as np

import streamlit.components.v1 as components
import random
from glove import Glove
from Polarization.w2v_polarity import w2vPolarity
from Polarization.glove_polarity import glovePolarity
from Polarization.bert_polarity import bertPolarity
from gensim.models import Word2Vec
from glove import Glove
import numpy as np
import csv
import ast


def load_data(data_path):
    data = pd.read_csv(data_path)
    return data

entity_trump = load_data('entities_clinton.csv')
plain_text = entity_trump["plain_text"]
original_text = []
for text in plain_text:
    original_text.append(ast.literal_eval(text))

print(original_text)
tweet_marked= pd.read_csv('highlight_trump.csv')
tweets = tweet_marked["text"]
for tweet in tweets:
    for text in original_text[0]:
        print(text)
        if(text in tweet):
            print(tweet)
#ast.literal_eval()
#plain_text =  trumo['type'].apply(literal_eval)

'''
selected = "Donald_Trump"
tweet = "CNBCOrganisation, Time magazineWrittenWork online polls say Donald TrumpPerson won the first presidential debate' via @WashTimes. #MAGA https://t.co/PGimqYKPoJ"
for index, row in entity_trump.iterrows():
    if row["label"] == selected:
        #print(row["plain_text"])
        texts = list((row["plain_text"].split(',')))
        print("List: ",texts)
        set(texts[0])
        for t in texts:
            
            print(t)

'''

     

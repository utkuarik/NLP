#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 14:56:29 2019

@author: uarik
"""

import nltk
import gensim
from xml.dom import minidom
import re  # For preprocessing
import pandas as pd  # For data handling
from time import time  # To time our operations
from collections import defaultdict  # For word frequency
import pandas as pd
from tqdm import tqdm 
import spacy  # For preprocessing

import logging  # Setting up the loggings to monitor gensim

from pprint import pprint  # pretty-printer
from collections import defaultdict
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 


#%%

# Read XML File
mydoc = minidom.parse('en-tr.tmx')
items = mydoc.getElementsByTagName('seg')

#%% get sentences
sentence_list = []

for i in items:
    sentence_list.append(i.firstChild.data)

sentence_list = str(sentence_list) 

#for i in items.loc[:,0]:
#    sentence_list.append(i)

sentence_list = str(sentence_list) 

#%%remove common words and tokenize
stoplist = set('''for a of the and to in when ve veya ama ancak fakat öyle as was were bir is are can they be or that have by an other but
               it such on at with more most from between thus has would about which ' " ,   '''.split())

texts = [     [word for word in sentence.lower().split() if word not in stoplist]
     for sentence in [sentence_list]
    ]

texts = str(texts)
  
  
stop_words = set(stopwords.words('english')) 
  
#word_tokens = word_tokenize(texts) 

filtered_sentence = [] 

filtered_sentence = [[w for w in sentence.lower().split() if not w in stop_words]
                     for sentence in [sentence_list] ]
  
texts = filtered_sentence
  
#for w in word_tokens: 
#    if w not in stop_words: 
#        filtered_sentence.append(w) 

#%% remove words that appear only once
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

texts = [
     [token for token in text if frequency[token] > 1]
     for text in texts ]

texts = texts[0]


word_list = texts

#%% create word2vec model
model = gensim.models.Word2Vec ([word_list], size=100, window=1, min_count=3, workers=10, sg = 1)

word = ["istanbul"]
model.wv.most_similar(positive = word)

model.wv.vocab
model.wv.similarity('siyah', 'black')



# %% Read txt File
mydoc = pd.read_csv('en-tr.txt', header=None, delimiter="\t", encoding = 'utf-8')
items = pd.DataFrame(data=mydoc[0])


# remove words that appear only once
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

texts = [
     [token for token in text if frequency[token] > 1]
     for text in texts ]

texts = texts[0]

# remove common words and tokenize
stoplist = set('''for a of the and to in when ve veya ama ancak fakat öyle as was were bir is are can they be or that have by an other but
               it such on at with more most from between thus has would about which   '''.split())

texts = [     [word for word in sentence.lower().split() if word not in stoplist]
     for sentence in [sentence_list]
    ]

word_list = texts

            
model = gensim.models.Word2Vec ([word_list], size=100, window=10, min_count=1, workers=10)


#model.train(word_list, total_examples=len(word_list), epochs=5)

word = ["red"]
model.wv.most_similar(positive = word)

model.wv.vocab
model.wv.similarity('man', 'woman')







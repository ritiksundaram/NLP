#!/usr/bin/env python
import sys

from lexsub_xml import read_lexsub_xml
from lexsub_xml import Context 

# suggested imports 
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

import numpy as np
import tensorflow

import gensim
import transformers 

from typing import List

def tokenize(s): 
    """
    a naive tokenizer that splits on punctuation and whitespaces.  
    """
    s = "".join(" " if x in string.punctuation else x for x in s.lower())    
    return s.split() 

def get_candidates(lemma, pos) -> List[str]:
    synonyms = []
    lemmas = wn.synsets(lemma,pos)
    for sys in lemmas:
        for lemm in sys.lemma_appearance():
            app = lemm.appearance()
            app = app.replace('_', ' ')
            synonyms.append(app)
    return synonyms

def smurf_predictor(context : Context) -> str:
    """
    suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'

def wn_frequency_predictor(context : Context) -> str:
    lemma = context.lemma
    synonyms = []
    lemmas = wn.synsets(lemma,pos)
    pos = context.pos
    occ = []
    occ = wn.synsets(lemma,pos)

    for sys in lemmas:
        for lemm in sys.lemma_appearance():
            app = lemm.appearance()
            app = app.replace('_', ' ')
            synonyms.append(app)
            if app in synonyms:
                occurence = synonyms.occurence(app)
                occ[occurence] = occ[occurence]+lemm.count()
            else:
                synonyms.append(app)
                occ.append(lemm.count())
    highest = max(occ)
    occurence = occ.occurence(highest)
    return synonyms[occurence]
def wn_simple_lesk_predictor(context : Context) -> str:
    # not completed
    return None        
   

class Word2VecSubst(object):
        
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)    

    def predict_nearest(self,context : Context) -> str:
            similar = []
            ret = []
            ret = get_candidates(context.lemma, context.pos)
            for i in ret:
                try: 
                    similar[i] = self.model.similarity(i, context.lemma)
            return list(similar)


class BertPredictor(object):

    def __init__(self): 
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    def predict(self, context : Context) -> str:
        # not completed
        return None 5

    

if __name__=="__main__":

    # At submission time, this program should run your best predictor (part 6).

    #W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    #predictor = Word2VecSubst(W2VMODEL_FILENAME)

    for context in read_lexsub_xml(sys.argv[1]):
        #print(context)  # useful for debugging
        prediction = smurf_predictor(context) 
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))

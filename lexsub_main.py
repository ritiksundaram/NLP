#!/usr/bin/env python
import sys
import string
from collections import Counter
from typing import List

from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
import numpy as np
import tensorflow
import gensim
import transformers

from lexsub_xml import read_lexsub_xml
from lexsub_xml import Context

def tokenize(s): 
    """
    A naive tokenizer that splits on punctuation and whitespaces.  
    """
    s = "".join(" " if x in string.punctuation else x for x in s.lower())    
    return s.split() 

def get_candidates(lemma, pos) -> List[str]:
    """
    Retrieves synonyms for a given lemma and part of speech from WordNet.
    """
    synonyms = []
    for synset in wn.synsets(lemma, pos=pos):
        for lemma in synset.lemmas():
            synonym = lemma.name().replace('_', ' ')
            if synonym not in synonyms:
                synonyms.append(synonym)
    return synonyms

def smurf_predictor(context: Context) -> str:
    """
    Suggests 'smurf' as a substitute for all words.
    """
    return 'smurf'

def wn_frequency_predictor(context: Context) -> str:
    """
    Suggests a substitute for a word based on its frequency of appearance in WordNet.
    """
    lemma = context.lemma
    pos = context.pos
    candidates = get_candidates(lemma, pos)
    
    frequency_dict = Counter()
    for synset in wn.synsets(lemma, pos):
        for lemma in synset.lemmas():
            if lemma.name().replace('_', ' ') in candidates:
                frequency_dict[lemma.name().replace('_', ' ')] += lemma.count()
    
    if frequency_dict:
        return frequency_dict.most_common(1)[0][0]
    else:
        return lemma
def wn_simple_lesk_predictor(context: Context) -> str:
    best_sense = None
    max_overlap = 0
    context_tokens = set(tokenize(context.left_context + ' ' + context.right_context))
    
    for synset in wn.synsets(context.lemma, pos=context.pos):
        gloss_tokens = set(tokenize(synset.definition()))
        overlap = len(context_tokens.intersection(gloss_tokens))
        
        if overlap > max_overlap:
            max_overlap = overlap
            best_sense = synset
    
    if best_sense:
        return best_sense.lemmas()[0].name().replace('_', ' ')
    else:
        return context.lemma
class Word2VecSubst(object):
    
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)    

    def predict_nearest(self, context: Context) -> str:
        candidates = get_candidates(context.lemma, context.pos)
        best_candidate = None
        best_similarity = -1
        
        for candidate in candidates:
            try:
                similarity = self.model.similarity(candidate.replace(' ', '_'), context.lemma.replace(' ', '_'))
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_candidate = candidate
            except KeyError:
                continue
        
        return best_candidate if best_candidate else context.lemma
class BertPredictor(object):

    def __init__(self): 
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    def predict(self, context: Context) -> str:
        # Replace the target lemma with the [MASK] token and tokenize
        text = context.left_context + ' [MASK] ' + context.right_context
        input_ids = self.tokenizer.encode(text, return_tensors="tf")
        
        # Predict the masked token
        mask_token_index = np.where(input_ids == self.tokenizer.mask_token_id)[1]
        token_logits = self.model(input_ids)[0]
        mask_token_logits = token_logits[0, mask_token_index, :]
        
        # Get the top token predicted
        top_token = np.argmax(mask_token_logits).numpy()
        predicted_token = self.tokenizer.decode([top_token])
        
        return predicted_token

# Add additional functionality as needed here, such as evaluation methods or integration with embeddings

if __name__ == "__main__":
   # At submission time, this program should run your best predictor (part 6).

    #W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    #predictor = Word2VecSubst(W2VMODEL_FILENAME)

    for context in read_lexsub_xml(sys.argv[1]):
        #print(context)  # useful for debugging
        prediction = smurf_predictor(context) 
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))



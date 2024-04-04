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

# Add additional functionality as needed here, such as evaluation methods or integration with embeddings

if __name__ == "__main__":
    # Example usage
    context = Context(lemma='happy', pos='a', left_context='I am very', right_context='today.')
    print("Smurf Predictor:", smurf_predictor(context))
    print("WordNet Frequency Predictor:", wn_frequency_predictor(context))

    # Load your dataset, process it, and use the above functions for lexical substitution

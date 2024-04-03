from conll_reader import DependencyStructure, DependencyEdge, conll_reader
from collections import defaultdict
import copy
import sys

import numpy as np
import keras

from extract_training_data import FeatureExtractor, State

class Parser(object):

    def __init__(self, extractor, modelfile):
        self.model = keras.models.load_model(modelfile)
        self.extractor = extractor

        # The following dictionary from indices to output actions will be useful
        self.output_labels = dict([(index, action) for (action, index) in extractor.output_labels.items()])

    '''
    This method uses the model's predictions
    to determine what is the best transition
    that can be used for the parser
    '''
    def parse_sentence(self, words, pos):

        state = State(range(1,len(words)))
        state.stack.append(0)

        # TODO: Write the body of this loop for part 4
        while state.buffer:
            arr = self.extractor.get_input_representation(words,pos,state) # getting the input representation
            input = np.reshape(arr, (-1,6)) # reshaping the input to the appropriate size
            prediction = (-self.model.predict(input)).argsort() # getting the prediction from the model, sorting the indices according to their corresponding values, and reversing that order so it's from max to min (most likely to least likely)
            for index in prediction[0]: # iterating through the highest scoring predictions
                transition = self.output_labels[index] # looking at the current best transition
                # the following checks determine if the above transition is legal
                if transition[0] == 'shift' and len(state.buffer) > 1 or (len(state.buffer) == 1 and len(state.stack) == 0):
                # if the model predicts a shift, we have to check if the buffer has more than one element OR, if the stack is empty, then the buffer has at least one element
                    state.shift() # if the conditions are met, then we can shift
                    break # once the transition has been made, we break, because only one transition can be made at a time
                elif transition[0] == 'left_arc' and len(state.stack) > 0 and state.stack[-1] != 0:
                # if we can't shift, then we see if we can do a left arc, which can only be done if the stack isn't empty and if the top of the stack isn't the root
                    state.left_arc(transition[1]) # if the conditions are met, then we can left arc transition
                    break # once the transition has been made, we break, because only one transition can be made at a time
                elif transition[0] == 'right_arc' and len(state.stack) > 0:
                # if we can't shift or left_arc, then we see if we can do a right arc, which can only be done if the stack isn't empty
                    state.right_arc(transition[1]) # if the conditions ar emet, then we can right arc transition
                    break # once the transition has been made, we break, because only one transition can be made at a time

        result = DependencyStructure()
        for p,c,r in state.deps:
            result.add_deprel(DependencyEdge(c,words[c],pos[c],p, r))

        return result


if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r')
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1)

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    parser = Parser(extractor, sys.argv[1])

    with open(sys.argv[2],'r') as in_file:
        for dtree in conll_reader(in_file):
            words = dtree.words()
            pos = dtree.pos()
            deps = parser.parse_sentence(words, pos)
            print(deps.print_conll())
            print()

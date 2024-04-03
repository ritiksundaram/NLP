from conll_reader import DependencyStructure, conll_reader
from collections import defaultdict
import copy
import sys
import keras
import numpy as np
from keras.utils import to_categorical

class State(object):
    def __init__(self, sentence = []):
        self.stack = []
        self.buffer = []
        if sentence:
            self.buffer = list(reversed(sentence))
        self.deps = set()

    def shift(self):
        self.stack.append(self.buffer.pop())

    def left_arc(self, label):
        self.deps.add( (self.buffer[-1], self.stack.pop(),label) )

    def right_arc(self, label):
        parent = self.stack.pop()
        self.deps.add( (parent, self.buffer.pop(), label) )
        self.buffer.append(parent)

    def __repr__(self):
        return "{},{},{}".format(self.stack, self.buffer, self.deps)



def apply_sequence(seq, sentence):
    state = State(sentence)
    for rel, label in seq:
        if rel == "shift":
            state.shift()
        elif rel == "left_arc":
            state.left_arc(label)
        elif rel == "right_arc":
            state.right_arc(label)

    return state.deps

class RootDummy(object):
    def __init__(self):
        self.head = None
        self.id = 0
        self.deprel = None
    def __repr__(self):
        return "<ROOT>"


def get_training_instances(dep_structure):

    deprels = dep_structure.deprels

    sorted_nodes = [k for k,v in sorted(deprels.items())]
    state = State(sorted_nodes)
    state.stack.append(0)

    childcount = defaultdict(int)
    for ident,node in deprels.items():
        childcount[node.head] += 1

    seq = []
    while state.buffer:
        if not state.stack:
            seq.append((copy.deepcopy(state),("shift",None)))
            state.shift()
            continue
        if state.stack[-1] == 0:
            stackword = RootDummy()
        else:
            stackword = deprels[state.stack[-1]]
        bufferword = deprels[state.buffer[-1]]
        if stackword.head == bufferword.id:
            childcount[bufferword.id]-=1
            seq.append((copy.deepcopy(state),("left_arc",stackword.deprel)))
            state.left_arc(stackword.deprel)
        elif bufferword.head == stackword.id and childcount[bufferword.id] == 0:
            childcount[stackword.id]-=1
            seq.append((copy.deepcopy(state),("right_arc",bufferword.deprel)))
            state.right_arc(bufferword.deprel)
        else:
            seq.append((copy.deepcopy(state),("shift",None)))
            state.shift()
    return seq


dep_relations = ['tmod', 'vmod', 'csubjpass', 'rcmod', 'ccomp', 'poss', 'parataxis', 'appos', 'dep', 'iobj', 'pobj', 'mwe', 'quantmod', 'acomp', 'number', 'csubj', 'root', 'auxpass', 'prep', 'mark', 'expl', 'cc', 'npadvmod', 'prt', 'nsubj', 'advmod', 'conj', 'advcl', 'punct', 'aux', 'pcomp', 'discourse', 'nsubjpass', 'predet', 'cop', 'possessive', 'nn', 'xcomp', 'preconj', 'num', 'amod', 'dobj', 'neg','dt','det']


class FeatureExtractor(object):

    def __init__(self, word_vocab_file, pos_vocab_file):
        self.word_vocab = self.read_vocab(word_vocab_file)
        self.pos_vocab = self.read_vocab(pos_vocab_file)
        self.output_labels = self.make_output_labels()

    def make_output_labels(self):
        labels = []
        labels.append(('shift',None))

        for rel in dep_relations:
            labels.append(("left_arc",rel))
            labels.append(("right_arc",rel))
        return dict((label, index) for (index,label) in enumerate(labels))

    def read_vocab(self,vocab_file):
        vocab = {}
        for line in vocab_file:
            word, index_s = line.strip().split()
            index = int(index_s)
            vocab[word] = index
        return vocab

    '''
    This method returns the input representation
    (a vector of length 6 with the top three words
    from the stack and the top three words from
    the buffer) for the model
    '''
    def get_input_representation(self, words, pos, state):
        # TODO: Write this method for Part 2

        input = np.array([-1, -1, -1, -1, -1, -1]) # default input array of length 6 filled with values of -1, which correspond to <NULL>

        if len(state.stack) > 0:
            input[0] = state.stack[-1] # first index filled with the top stack value
            if len(state.stack) > 1:
                input[1] = state.stack[-2] # second index filled with the second stack value
                if len(state.stack) > 2:
                    input[2] = state.stack[-3] # third index filled with the third stack value

        if len(state.buffer) > 0:
            input[3] = state.buffer[-1] # fourth index filled with the top buffer value
            if len(state.buffer) > 1:
                input[4] = state.buffer[-2] # fifth index filled with the second buffer value
                if len(state.buffer) > 2:
                    input[5] = state.buffer[-3] # last index filled with third buffer value

        i = 0 # counter to keep track of indices in input array
        for index in input: # iterating through each value in the input array (indices which correspond to words)
            if index == -1: # if the value is still its default -1, then nothing was in the stack
                input[i] = self.word_vocab['<NULL>'] # thus, that corresponding index is replaced with the dictionary value for <NULL>
            else: # if the value is no longer its default -1, then it was replaced
                word = str(words[index]) # getting the word according to the index given in the input array
                if word == 'None': # if the word is None, then it is the root
                    input[i] = self.word_vocab['<ROOT>'] # replace the index in the input array with the dictionary value for <ROOT>
                elif str(pos[index]) == 'CD': # if the POS tag for the word is CD
                    input[i] = self.word_vocab['<CD>'] # then replace the index in the input array with the dictionary value for <CD>
                elif str(pos[index]) == 'NNP': # if the POS tag for the word is NNP
                    input[i] = self.word_vocab['<NNP>'] # then replace the index in the input array with the dictionary value for <NNP>
                elif word.lower() in self.word_vocab: # if the word is in the dictionary
                    input[i] = self.word_vocab[word.lower()] # then replace the index in the input array with the dictionary value for that word
                else: # if the word is not in the dictionary
                    input[i] = self.word_vocab['<UNK>'] # replace that index in the input array with the dictionary value for <UNK>
            i = i + 1 # iterating the counter so we can move to the next index in the input array

        return input # returning the input array / representation

    '''
    This method returns the output representation
    (a one-hot vector of length 91) for the model
    '''
    def get_output_representation(self, output_pair):
        # TODO: Write this method for Part 2
        output = np.zeros(91) # creating a 91-length array full of 0s

        index = 0 # counter for which value in the output should be set to 1
        if output_pair[0] == 'shift': # if the pair is (shift, None)
            output[index] = 1 # the first index of the output array is set to 1
        for relation in dep_relations: # iterating through each relation in the list of possible relations
            if output_pair[0] == 'left_arc': # if it is a left arc transition
                if output_pair[1] == relation: # and if the pair's relation is the same as the current relation
                    output[index+1] = 1 # then the current odd index is set to 1
            if output_pair[0] == 'right_arc': # if it is a right arc transition
                if output_pair[1] == relation: # and if the pair's relation is the same as the current relation
                    output[index+2] = 1 # then the current even index is set to 1
            index = index + 2 # index is iterated by 2, because all odd indices are for left arc relations and all even indices are for right arc relations

        return output # returning the output array / representation

def get_training_matrices(extractor, in_file):
    inputs = []
    outputs = []
    count = 0
    for dtree in conll_reader(in_file):
        words = dtree.words()
        pos = dtree.pos()
        for state, output_pair in get_training_instances(dtree):
            inputs.append(extractor.get_input_representation(words, pos, state))
            outputs.append(extractor.get_output_representation(output_pair))
        if count%100 == 0:
            sys.stdout.write(".")
            sys.stdout.flush()
        count += 1
    sys.stdout.write("\n")
    return np.vstack(inputs),np.vstack(outputs)



if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r')
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1)


    with open(sys.argv[1],'r') as in_file:

        extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
        print("Starting feature extraction... (each . represents 100 sentences)")
        inputs, outputs = get_training_matrices(extractor,in_file)
        print("Writing output...")
        np.save(sys.argv[2], inputs)
        np.save(sys.argv[3], outputs)

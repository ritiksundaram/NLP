from extract_training_data import FeatureExtractor
import sys
import numpy as np
import keras
from keras import Sequential
from keras.layers import Flatten, Embedding, Dense, Activation

'''
This method builds the neural network
that will be used to predict transitions
'''
def build_model(word_types, pos_types, outputs):
    # TODO: Write this function for part 3

    model = Sequential()
    model.add(Embedding(word_types, 32, input_length=6)) # embedding layer
    model.add(Flatten()) # flattening layer
    model.add(Dense(100, activation='relu')) # dense layer of 100 units
    model.add(Dense(10, activation='relu')) # dense layer of 10 units
    model.add(Dense(91, activation='softmax')) # output layer with 91 units because there are 91 possible outputs and we need a dimension for each one
    model.compile(keras.optimizers.Adam(lr=0.01), loss="categorical_crossentropy")

    return model


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
    print("Compiling model.")
    model = build_model(len(extractor.word_vocab), len(extractor.pos_vocab), len(extractor.output_labels))
    inputs = np.load(sys.argv[1])
    outputs = np.load(sys.argv[2])
    print("Done loading data.")

    # Now train the model
    model.fit(inputs, outputs, epochs=5, batch_size=100)

    model.save(sys.argv[3])

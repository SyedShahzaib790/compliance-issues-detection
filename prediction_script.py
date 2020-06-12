#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from functions import *

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


def train(sentences_list, true_labels, maxlen=50):
    

    tokenizer = Tokenizer(num_words=1000)
    tokenizer.fit_on_texts(sentences_list)
    vocab_size=len(tokenizer.word_index)+1

    x_train = tokenizer.texts_to_sequences(sentences_list)
    x_train=pad_sequences(x_train,padding='post', maxlen=maxlen)
    
    epochs = 20
    embedding_dim=100
    model=Sequential()
    model.add(layers.Embedding(input_dim=vocab_size,output_dim=embedding_dim,input_length=maxlen))
    model.add(layers.LSTM(units=50,return_sequences=True))
    model.add(layers.LSTM(units=10))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(8))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, true_labels, epochs=epochs, batch_size=64)
    
    return [tokenizer, model]


def predict(sentences, tokenizer, model, maxlen=50):
    
    x_test = tokenizer.texts_to_sequences(sentences)
    x_test = pad_sequences(x_test, padding='post', maxlen=maxlen)
    predictions = [int(a[0]>=0.5)  for a in model.predict(x_test)]
    return predictions[0]



""" Train a Deep Learning Model """

# dataset = pd.read_pickle('./compliance_labeled_data.pkl')
# dataset.label = dataset.label.map(lambda x: 0 if x==-1 else x)
# tokenzier, model = train(dataset.sentences.values, dataset.label.values)

""" Save Model Parameters """
model_file_name = 'trained_model.pkl'
# pd.to_pickle([tokenizer, model], model_file_name)


import argparse
import sys

try:
    parser = argparse.ArgumentParser()
    parser.add_argument("sentence", help="Predict whether a sentence includes False or True compliance related discussion",type=str)
    args = parser.parse_args()
    
    sentence = args.sentence

    """  Load Trained Model """
    tokenizer, model = pd.read_pickle(model_file_name)
    prediction = predict([sentence], tokenizer, model)

    message_dictionary = {
        0: "0 (True-Postive Compliance Related Discussion)",
        1: "1 (False-Postive Compliance Related Discussion)"
    }

    print ('======================================================')
    print ("Input: ", sentence)
    print ("Output: ",message_dictionary[prediction])
    print ('======================================================')

except:
    e = sys.exc_info()[0]
    print (e)


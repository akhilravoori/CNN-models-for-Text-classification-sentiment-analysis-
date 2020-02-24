#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 19:50:45 2020

@author: akhil
"""
from __future__ import print_function

from keras.preprocessing import sequence
from keras.layers import Dense, Dropout, Activation,LeakyReLU
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D,MaxPooling1D,LSTM
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input,Flatten,concatenate

max_features =20000
maxlen = 200
batch_size = 32
embedding_dims = 125
filters = 64
kernel_size = 3
hidden_dims = 250
epochs = 2

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_train[1]
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')

main_input = Input(shape=(200,), dtype='int32', name='main_input')
x=Embedding(max_features
            ,embedding_dims
            ,input_length=maxlen)(main_input)
y=Dropout(0.25)(x)
filtersize=[2,3,4]
convs=[]
for i in filtersize:
    a=Conv1D(filters
             ,kernel_size=i
             ,padding='valid'
             ,activation='relu'
             ,strides=1)(y)
    b=MaxPooling1D(pool_size=3)(a)
    convs.append(b)

merge=concatenate(convs, axis=1)

conv=Conv1D(filters
            ,kernel_size=3
            ,padding='valid'
            ,activation='relu'
            ,strides=1)(merge)
x=MaxPooling1D(pool_size=3)(conv)

x=Dense(128,activation='relu')(x)
x=Dropout(0.2)(x)
x = Flatten()(x)
x = Dense(128)(x)
x = Dropout(0.2)(x)
x=LeakyReLU(alpha = 0.2)(x)


pred = Dense(1, activation='sigmoid')(x)
model=Model(main_input,pred)
model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))



    




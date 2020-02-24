from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation,LeakyReLU
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D,LSTM,MaxPooling1D
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer, one_hot
from keras.preprocessing.sequence import pad_sequences
tokenizer=Tokenizer()

max_features = 5000
maxlen = 400
batch_size = 32
embedding_dims = 50
filters = 250
kernel_size = 3
hidden_dims = 250
epochs = 2

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train[1]
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
x_test[1]

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_train[1]
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
model = Sequential()


model.add(Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen))
model.add(Dropout(0.2))


model.add(Conv1D(filters,
                  kernel_size,
                  padding='valid',
                  activation='relu',
                  strides=1))

model.add(MaxPooling1D(pool_size=3))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))

# model.add(Dense(hidden_dims))
# model.add(Dropout(0.2))
# model.add(LeakyReLU(alpha=0.2))



model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))

import json
input=open('opinion_article-370-indias-_55150de8954af2c527040304990edddd__amp_1580301444.html.json', 'r')
x=json.load(input)

X_sample = [x['maintext']]
X_code = tokenizer.texts_to_sequences(X_sample)
X_code = pad_sequences(X_code, padding='post', maxlen=maxlen)

y_sample = model.predict_classes(X_code).flatten().tolist()

print('Prediction: ',y_sample)

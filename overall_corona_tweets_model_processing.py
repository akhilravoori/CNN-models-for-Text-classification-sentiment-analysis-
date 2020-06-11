#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 10:41:14 2020

@author: akhil
"""
import json
import pandas as pd
import nltk
import re
import tokenizers
from ibm_watson import ToneAnalyzerV3
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

def strip_emoticons(s):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', s.strip())
"""
This is where we implement the tone-analyzer api from IBM-Watson to 
obtain the tones of the tweets and add a column in the original dataframe 
that we had.

"""

authenticator = IAMAuthenticator('bjs7HIbH9SNgnpDIDy6rx3Ynmb_5SPetzmaXXv7t3_KL')
tone_analyzer = ToneAnalyzerV3(
    version='2020-03-26',
    authenticator=authenticator
)
tone_analyzer.set_service_url('https://api.eu-gb.tone-analyzer.watson.cloud.ibm.com/instances/33756fab-977f-4818-8b4c-f8bf06e8e91e')
df=pd.read_csv('/home/akhil/Desktop/Projects/inf_project/t5.csv')
df1=pd.read_csv('/home/akhil/Desktop/Projects/inf_project/covid2019.csv')
x=df.tweet_text
# tweet=x[0]
# result = re.sub(r"http\S+", "", tweet)
# print(result)
# y=result.replace('#','')
"""
The following for loop is where I remove the emoticons and remove
urls in the data. I use the strip_emoticons function that I made above.

"""
z=x
for i in range(len(z)):
    tweet=z[i]
    result = re.sub(r"http\S+", "", tweet)
    y=result.replace('#','')
    y=result.replace('@','')
    y=strip_emoticons(y)
    z[i]=y
df.tweet_text=z   
a=[]
for text in df.tweet_text:
    tone_analysis = tone_analyzer.tone(
        {'text': text},
        content_type='text/plain',
        sentences=False
    ).get_result()
    a.append(tone_analysis["document_tone"])
df["tone"]=a    
"""
Below is the function to make a csv containing an extra column for the tones. 

"""
df.to_csv('coronatones.csv')

"""
The following section of code presents the mapping of the tones 
to neutral, positive, and negative and then further mapping them to
0,1,2 respectively.

"""
df=pd.read_csv('coronatones.csv')
d=eval(df.tone[0])
x=d['tones'][0]
x
x['score']
neutral=['tentative','analytical']
positive=['joy','confident','happy']
negative=['angry','sadness','fear','anger']
for i in range(len(df.tone)):
    df.tone[i]=eval(df.tone[i])
df.tone[0]
tones_list=[]
df.tone[0]['tones'][0]
for i in range(len(df.tone)):
    if len(df.tone[i]['tones'])==0:
        df.tone[i]=None
        continue
    x=df.tone[i]['tones'][0]
    if x['tone_id'] in positive:
        df.tone[i]='positive'
        continue
    if x['tone_id'] in neutral:
        df.tone[i]='neutral'
        continue
    if x['tone_id'] in negative:
        df.tone[i]='negative'
        continue
d={'positive':1,'negative':2,'neutral':0}
df.tone=df['tone'].map(d)
df=df.dropna()

"""
We dropped nan columns as some tones were not produced by 
the Tone-Analyzer.

"""


df.to_csv('classified_corona.csv')   
df.groupby('tone').count().tweet_text
"""
The groupby function shows us a relatively even spread of neutral, 
positive and negative tweets slightly skewed towards the neutral tones.
"""
    
    

DATASET_COLUMNS = [ "tweet_text", "tone"]
DATASET_ENCODING = "ISO-8859-1"
data = pd.read_csv('classified_corona.csv')
data=data.dropna()
"""
below we are dropping unnecessary columns in our dataframe.

"""
data=data.drop(columns=['Unnamed: 0.1','Unnamed: 0','username','followers','created_at'],axis=1)
data.columns=['tweet','tone']
data


"""
We set our X,Y for input.
The following four lines of code is to make our labels from 0,1,2
to [1,0,0],[0,1,0],[0,0,1] as our Y input to our model as it requires
a vector input for denoting multiple classes.

"""
from keras.utils import to_categorical

Y=data.tone
Y= to_categorical(Y)
Y
X=data

"""
We use nltk to preprocess our tweets.
"""
from nltk.corpus import stopwords
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

"""
The following set of lambda functions uses regular expressions to 
clean the tweets and filter out stopwords and unnecessary content.

"""
TAG_CLEANING_RE = "@\S+"
# Remove @tags
X['tweet'] = X['tweet'].map(lambda x: re.sub(TAG_CLEANING_RE, ' ', x))

# Smart lowercase
X['tweet'] = X['tweet'].map(lambda x: x.lower())

# Remove numbers
X['tweet'] = X['tweet'].map(lambda x: re.sub(r'\d+', ' ', x))

# Remove links
TEXT_CLEANING_RE = "https?:\S+|http?:\S|[^A-Za-z0-9]+"
X['tweet'] = X['tweet'].map(lambda x: re.sub(TEXT_CLEANING_RE, ' ', x))

# Remove Punctuation
X['tweet']  = X['tweet'].map(lambda x: x.translate(x.maketrans('', '', string.punctuation)))

# Remove white spaces
X['tweet'] = X['tweet'].map(lambda x: x.strip())

# Tokenize into words
X['tweet'] = X['tweet'].map(lambda x: word_tokenize(x))
 
# Remove non alphabetic tokens
X['tweet'] = X['tweet'].map(lambda x: [word for word in x if word.isalpha()])

# Filter out stop words
stop_words = set(stopwords.words('english'))
X['tweet'] = X['tweet'].map(lambda x: [w for w in x if not w in stop_words])

# Word Lemmatization
lem = WordNetLemmatizer()
X['tweet'] = X['tweet'].map(lambda x: [lem.lemmatize(word,"v") for word in x])

# Turn lists back to string
X['tweet'] = X['tweet'].map(lambda x: ' '.join(x))

X
"""
We used sklearn's train_test_split in order to split our dataset 
into the respective train data and test data. We used an 80-20 train-test
split. 

"""
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print("TRAIN size:", len(X_train))
print("TEST size:", len(X_test))
y_train

import gensim
"""
The following section of code until the next section of importing
libraries is where we incorporated word2vec to embed our words. We
found a total vocab size of 4655 words. 
"""
W2V_SIZE = 300
W2V_WINDOW = 5
W2V_EPOCH = 25
W2V_MIN_COUNT = 1

documents = [_text.split() for _text in X_train.tweet] 
w2v_model = gensim.models.word2vec.Word2Vec(size=W2V_SIZE, 
                                            window=W2V_WINDOW, 
                                            min_count=W2V_MIN_COUNT, 
                                            workers=8)
w2v_model.build_vocab(documents)

words = w2v_model.wv.vocab.keys()
vocab_size = len(words)
print("Vocab size", vocab_size)

w2v_model.train(documents, total_examples=len(documents), epochs=W2V_EPOCH)



"""
The following section of code 
is to preprocess further the data using the keras tokenizer and padding
sequences to the maximum length of a sequence that we found which was 300.
We had to pad for both the test data and train data. 

So each input we put into the model has dimensions of (300,)
hence embedding dimensions were also set to 300

"""
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train.tweet)
word_index = tokenizer.word_index
vocab_size = len(word_index)
print('Found %s unique tokens.' % len(word_index))
MAX_SEQUENCE_LENGTH = 300
EMBEDDING_DIM = 100
X_train_padded = tokenizer.texts_to_sequences(X_train.tweet)
X_train_padded = pad_sequences(X_train_padded, maxlen=MAX_SEQUENCE_LENGTH)
X_test_padded = tokenizer.texts_to_sequences(X_test.tweet)
X_test_padded = pad_sequences(X_test_padded, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X_train_padded.shape)



"""
The following section presents the model we made.
"""

from keras.preprocessing import sequence
from keras.layers import Dense, Dropout, Activation,LeakyReLU
from keras.layers import Embedding,BatchNormalization
from keras.layers import Conv1D, GlobalMaxPooling1D,MaxPooling1D,LSTM
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input,Flatten,concatenate

max_features = 4700 #The total vocab size we have
maxlen = 300 #mentioned above
batch_size = 64
embedding_dims = 100 
filters = 64 # 64 convolution filters will be applied on each sequence 
             # for each tweet.
#kernel_size = 1
hidden_dims = 100
epochs = 2

main_input = Input(shape=(300,), dtype='int32', name='main_input')
x=Embedding(max_features
            ,embedding_dims
            ,input_length=maxlen)(main_input)
y=Dropout(0.5)(x)

"""
Here is where we keep the options for the kernel size.
This allows us to keep multiple options for n-gram sizes
We took 1,2 as less number of words in tweets. 

"""
filtersize=[1,2]
convs=[]
for i in filtersize:
    a=Conv1D(128
             ,kernel_size=i
             ,padding='valid'
             ,activation='tanh'
             ,strides=1)(y)
    b=MaxPooling1D(pool_size=3)(a)
    convs.append(b)

"""
So here we merge two different convolutional layer outputs,
where two different kernel sizes were taken (1,2).
"""
merge=concatenate(convs, axis=1) 

"""
Now we input the previous merged output into another convolutional 
layer to further extract features.
"""
conv=Conv1D(128
            ,kernel_size=3
            ,padding='valid'
            ,activation='tanh'
            ,strides=1)(merge)
x=Activation('relu')(conv)
#x=BatchNormalization()(x)
x=GlobalMaxPooling1D()(x)

"""
The following Dense->Dropout->LeakyRelu is a hidden layer connection
to improve regularization and backpropogation.
"""
x=Dense(hidden_dims)(x)
x=Dropout(0.5)(x)
#x = Flatten()(x)
#x = Dense(128)(x)
#x = Dropout(0.5)(x)
x=LeakyReLU(alpha = 0.2)(x)


"""
Here is where we retrieve our final output and use a softmax layer to 
classify the outputs into the three classes, what is results is a (3,) 
shape vector with the probability of how the current input could be in 
each class
For eg, [0.1,0.8,0.1] would mean its a prediction in the [0,1,0] class 
which is positive
"""
pred = Dense(3,activation='softmax')(x)

model=Model(main_input,pred)
model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])
"""
The whole final section is us training the model, plotting losses, acc,
drawing the model structure, providing a summary of the model.

"""
model.summary()

history=model.fit(X_train_padded, y_train, epochs = 25, validation_data=(X_test_padded, y_test), batch_size=64)

import tensorflow as tf
tf.keras.utils.plot_model(
    model,
    to_file="model.png",
    show_shapes=False,
    show_layer_names=True,
    rankdir="TB",
    expand_nested=False,
    dpi=96,
)

print(history.history.keys())  
import matplotlib.pyplot as plt 
plt.figure(1)  
  
# summarize history for accuracy  
  
plt.subplot(211)  
plt.plot(history.history['acc'])  
plt.plot(history.history['val_acc'])  
plt.title('model accuracy')  
plt.ylabel('accuracy')  
plt.xlabel('epoch')  
plt.legend(['train', 'test'], loc='upper left')  
  
# summarize history for loss  
  
plt.subplot(212)  
plt.plot(history.history['loss'])  
plt.plot(history.history['val_loss'])  
plt.title('model loss')  
plt.ylabel('loss')  
plt.xlabel('epoch')  
plt.legend(['train', 'test'], loc='upper left')  
plt.show()
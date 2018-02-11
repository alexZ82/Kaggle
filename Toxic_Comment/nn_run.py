import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
import random
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score
import matplotlib
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix, vstack, lil_matrix
from os.path import isfile
import h5py
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import load_model
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from scipy import sparse, io
import os
import time
import theano
import sys
from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

if sys.argv[1]=='1':
        # load data
        train_data = pd.read_csv('train.csv')
        test_data = pd.read_csv('test.csv')

        # preprocess data
        def process_comment_text(txt):
                ntxt = re.sub(r"[^a-zA-Z]", " ", txt)
                ntxt = ntxt.lower()
                lemmatizer = WordNetLemmatizer()
                text = nltk.word_tokenize(ntxt)
                return ' '.join([lemmatizer.lemmatize(w,'v') for w in text])

        print('preprocessing...')
        train_data['processed'] = train_data.comment_text.apply(process_comment_text)
        test_data['processed'] = test_data.comment_text.apply(process_comment_text)

        # get list of words for input vector
        def getwordlist(data,minfreq,maxfreq):
                stop = set(stopwords.words('english'))
                text = nltk.word_tokenize(' '.join(data))
                dist = nltk.FreqDist(text)
                wordlist = [i for i in dist.keys() if dist[i]>minfreq and dist[i]<maxfreq and i not in stop]
                return wordlist
        print('generating input list...')
        words = getwordlist(data=train_data.processed.values,minfreq=100,maxfreq=80000)
        
        train_data.to_pickle('train.pkl')
        test_data.to_pickle('test.pkl')
        np.save('wordlist.npy',np.array(words))
else:
        train_data = pd.read_pickle('train.pkl')
        test_data = pd.read_pickle('test.pkl')
        words = np.load('wordlist.npy').tolist()
        print('using saved data... (',len(words),')')

def make_features(data,name):
    X = csr_matrix((0, len(words)))

    # for every comment check if the words corresponding to our input vector exist
    count=0
    print(name)
    for i in data.processed: #.loc[:100]:
        if count%1000==0:
            print(round(((count+1)/len(data.processed))*100,3),'%',end='   \r')
        cw = set(nltk.word_tokenize(i))
        add = [int(w in cw) for w in words]
        X = vstack([X, csr_matrix(add)], 'csr')
        count+=1
    io.mmwrite(name+'_X.mtx', X)
    
    if len(data.columns)>3:
        y = data.apply(lambda x: x[2:8],axis=1) # .loc[:100]
        y = csr_matrix(y.values)
        io.mmwrite(name+'_y.mtx', y)

if sys.argv[2]=='1':
        print('regenerating features...')
        # only if needs to regenerate
        make_features(train_data,'train_data')
        make_features(test_data,'test_data')
else:
        # read data
        print('reading data...')
        X_train = io.mmread('train_data_X.mtx').tocsr()
        y_train = io.mmread('train_data_y.mtx').tocsr()
        X_test = io.mmread('test_data_X.mtx').tocsr()

# Keras model
nout = y_train.shape[1]
nin = X_train.shape[1]
print('input size:',nin)
print('output size:',nout)
print('number of training exammples:',X_train.shape[0])

model = Sequential()
model.add(Dense(100, input_dim=nin, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(nout, activation='softmax'))

model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

# to deal with sparse matrix, add generator: https://stackoverflow.com/questions/41538692/using-sparse-matrices-with-keras-and-tensorflow
def batch_generator(X, y, batch_size):
    number_of_batches = X.shape[0]/batch_size
    counter=0
    shuffle_index = np.arange(y.shape[0])
    np.random.shuffle(shuffle_index)
    X =  X[shuffle_index, :]
    y =  y[shuffle_index]
    while 1:
        index_batch = shuffle_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X[index_batch,:].todense()
        y_batch = y[index_batch].todense()
        counter += 1
        yield(np.array(X_batch),y_batch)
        if (counter < number_of_batches):
            np.random.shuffle(shuffle_index)
            counter=0

# settings for fitting
size_batch = 32
nb_epoch=5
    
# fit
print('fitting...')
model.fit_generator(batch_generator(X_train, y_train, size_batch),X_train.shape[0], nb_epoch, verbose=1)

model.save('model.h5')


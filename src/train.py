import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
import tensorflow as tf
import pickle
import sys
import yaml

from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Flatten, TimeDistributed, Bidirectional
from keras.layers.core import Dense, Activation, Dropout
from keras.models import Sequential
from keras.utils import plot_model
from keras import optimizers
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras import backend as K

import datetime
import math
import hashlib
import time
import os
from src.datagenerator import DataGenerator

def getAgentHash(agent, agentHashRange) :
    hashret = int(hashlib.sha1(agent.encode('utf-8')).hexdigest(), 16) % agentHashRange

    return str(hashret)

def loadConfig():
    with open(sys.argv[1], "r") as ymlfile:
        cfg = yaml.load(ymlfile)
    return cfg

def getHash(fullURI, queryHashRange):
    hashret = 0

    #Handles cases with no '/'
    if(fullURI.find('/') == -1) :
        fullURI = '/' + fullURI

    uri = fullURI.split('?', 1)[0]

    if (len(uri.rsplit('/', 1)) > 1) :
        request = uri.rsplit('/', 1)[1]
    else :
        request = ''

    if (len(fullURI.split('?', 1)) > 1) :
        request = request + '?' + fullURI.split('?', 1)[1]
    else :
        request = request

    hashret = int(hashlib.sha1(request.encode('utf-8')).hexdigest(), 16) % queryHashRange
    uri = uri.rsplit('/', 1)[0]

    return str(hashret)

def getURI(fullURI):
    uri = fullURI.split('?', 1)[0]
    uri = uri.rsplit('/', 1)[0]


    if uri == '' :
        return '<EMPTY>'
    else :
        return str(uri)

def converttodatetime(x):
    if len(x)< 20:
        x += '.000'
    return datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f")

#TimeDif 0 Padding
# Start = 14
# End = 15
def getTimeClass(time_dif):
    time = 0
    if time_dif == 0.0:
        time = 0
    elif time_dif < 0.001 :
        time = 1
    elif time_dif < 0.05 :
        time = 2
    elif time_dif < 0.1 :
        time = 3
    elif time_dif < 1 :
        time = 4
    elif time_dif < 15 :
        time = 5
    elif time_dif < 180 :
        time = 6
    elif time_dif > 300 :
        time = 7

    return str(time)

def prepDataFrame(df, agentHashRange, queryHashRange) :
    #create a deepcopy of the original df
    df_temp = df[['timestamp', 'remote_addr', 'request_uri', 'status', 'http_user_agent', 'country_code']].copy()
    if type(df_temp['timestamp'].iloc[0]) == str:
        df_temp['timestamp'] = df_temp['timestamp'].apply(lambda x: converttodatetime(x))

    #converting type of status
    df_temp = df_temp.astype({'remote_addr' : 'str'})
    df_temp = df_temp.astype({'request_uri' : 'str'})
    df_temp = df_temp.astype({'status': 'str'})
    df_temp = df_temp.astype({'http_user_agent': 'str'})
    df_temp = df_temp.astype({'country_code': 'str'})

    #fill empty valuse with NONE
    df_temp['http_user_agent'] = df_temp['http_user_agent'].fillna('NONE')
    df_temp['http_user_agent'] = df_temp['http_user_agent'].apply(lambda x: getAgentHash(x, agentHashRange))

    #Get Hash
    df_temp['hash'] = df_temp['request_uri'].apply(lambda x: getHash(x, queryHashRange))

    #Get URI
    df_temp['request_uri'] = df_temp['request_uri'].apply(lambda x: getURI(x))

    #calculate the difference between requests from a specific user
    df_temp['time_diff'] = df_temp.groupby('remote_addr')['timestamp'].diff()
    df_temp['time_diff_group'] = df_temp['time_diff'].apply(lambda x: getTimeClass(x.total_seconds()))

    #Calculate the Combined Input of URI Hash Status Time
    df_temp['Input'] = df_temp['request_uri'] + '<JOIN>' + df_temp['hash'] + '<JOIN>' + df_temp['status'] + '<JOIN>' + df_temp['time_diff_group']

    return df_temp

def getSignificantRequest(dataframe, hashThreshold) :
    freq = dataframe['Input'].value_counts(normalize=True)

    ret = []

    index = freq.index
    for i in range(len(freq)):
        if freq[i] > hashThreshold :
            ret.append(index[i])

    return ret

def keepOrHash(uri, sig, inputHashRange) :
    if uri in sig :
        return uri
    else :
        return str(int(hashlib.sha1(uri.encode('utf-8')).hexdigest(), 16) % inputHashRange)

def sequentializeDataFrame(df, sig, sig2, inputHashRange):
    #create a deepcopy of the original df
    df_temp = df[['timestamp', 'remote_addr', 'request_uri', 'status', 'http_user_agent', 'country_code', 'time_diff', 'Input']].copy()
    if type(df_temp['timestamp'].iloc[0]) == str:
        df_temp['timestamp'] = df_temp['timestamp'].apply(lambda x: converttodatetime(x))
    #converting type of status
    df_temp = df_temp.astype({'remote_addr' : 'str'})
    df_temp = df_temp.astype({'request_uri' : 'str'})
    df_temp = df_temp.astype({'status': 'str'})
    df_temp = df_temp.astype({'http_user_agent': 'str'})
    df_temp = df_temp.astype({'country_code': 'str'})
    df_temp = df_temp.astype({'Input': 'str'})

    if sig2 != None :
        df_temp['Input2'] = df_temp['Input'].apply(lambda x: keepOrHash(x, sig2, inputHashRange))
    df_temp['Input'] = df_temp['Input'].apply(lambda x: keepOrHash(x, sig, inputHashRange))

    #create groups based on 5 min interval
    df_temp['groups'] = df_temp.groupby('remote_addr')['time_diff'].apply(lambda x: x.gt(pd.Timedelta(5, 'm')).cumsum())
    df_temp['time_diff'] = df_temp['time_diff'].apply(lambda x: getTimeClass(x.total_seconds()))
    #grouping in sequences of 20 length
    #df_temp['group_len'] = df_temp.groupby(['remote_addr', 'groups'])['timestamp'].rank(method = 'first')
    #df_temp['group_len'] = df_temp['group_len'].apply(lambda x: math.ceil(x/20))
    #create groups based on "remote_addr" and "groups"
    df_temp = df_temp.groupby(['remote_addr', 'groups'])

    #aggregation
    sr = df_temp['http_user_agent', 'country_code'].agg(lambda x: " ".join(x))
    sr['http_user_agent'] = sr['http_user_agent'].apply(lambda x : x.split(' '))
    sr['Input'] = df_temp['Input'].agg(lambda x: "<SEP>".join(x))
    if sig2 != None :
        sr['Input2'] = df_temp['Input2'].agg(lambda x: "<SEP>".join(x))
    sr.reset_index(inplace=True)
    sr = sr.rename(columns={'remote_addr': 'IP', 'country_code': 'CountrySeq',  'http_user_agent': 'AgentSeq', 'Input' : 'Input'})
    #converting to dataframe
    #sr = sr.to_frame()
    sr = sr.drop(columns = ['groups'])
    return sr

def getCountryAgentPair(x, y):
    agent = x[0]
    country = y.split(' ')[0]

    return str(country) + '<JOIN>' + str(agent)


def getCountryOnly(y):
    country = y.split(' ')[0]

    return str(country)

def getAgentOnly(x):
    agent = x[0]

    return str(agent)


#Remove request if Remote Addresses (IP) appear 10 or less times
#Because the data is not in chronological form, sort by timestamp to get in chronological form
def filterAndSort(df) :
    df = df.sort_values(by=['timestamp'])
    before = len(df)
    df = df.groupby('remote_addr').filter(lambda x: len(x) > 10)
    after = len(df)
    df = df.reset_index()

    print("Before = {}, After = {}".format(before, after))

    return df


def prepare_sentence(seq, maxlen, tokenizer):
    # Pads seq and slides windows
    seq = seq[:maxlen]
    seqX = np.append(tokenizer.word_index['<sos>'], seq)
    seqY = np.append(seq, tokenizer.word_index['<eos>'])

    x= pad_sequences([seqX],
        maxlen=maxlen+1,
        padding='post')[0]  # Pads before each sequence

    y= pad_sequences([seqY],
        maxlen=maxlen+1,
        padding='post')[0]  # Pads before each sequence

    return [x], [y]


def getTokenizer(df) :
    ### Dictionary for Normal ###
    tokenizer = Tokenizer(filters='', split='<sep>', oov_token='<OTHERS>' ,lower=True)
    tokenizer.fit_on_texts(df['Input'].values)

    tokenizer.fit_on_texts(['<SOS>'])
    tokenizer.fit_on_texts(['<EOS>'])

    return tokenizer

def createGeneratorData(df, tokenizer, max_len) :
    #Prepare training for normal model
    x = []
    y = []

    for seq in df['Input']:
        x_windows, y_windows = prepare_sentence(seq, max_len, tokenizer)
        x += x_windows
        y += y_windows
    x = np.array(x)
    y = np.array(y)  # The word <PAD> does not constitute a class

    x.shape = [len(x), max_len + 1, 1]
    y.shape = [len(y), max_len + 1, 1]

    print(x.shape)
    print(y.shape)

    return x, y

def trainModelP(size, max_len, vocab_size, config):
    # Parameters
    params = {'dim': (max_len + 1, 1),
          'batch_size': config['TRAININGPARAMS']['BATCH_SIZE'],
          'progress' : 'data/' + sys.argv[2],
          'n_classes': vocab_size,
          'n_channels': 1,
          'shuffle': True}

    input_emb_dim = config['MODELPARAMS']['INPUT_EMBED_DIM']
    lstm_emb_dim = config['MODELPARAMS']['LSTM_DIM']
    # Datasets generation
    cut = math.floor(size*config['TRAININGPARAMS']['PERCENTAGETRAIN'])
    partition = dict()
    partition['train'] = []
    partition['validation'] = []

    for i in range(cut):
        partition['train'].append(i)

    for i in range(cut, size) :
        partition['validation'].append(i)

    # Generators
    training_generator_uri_normal = DataGenerator(False, partition['train'], **params)
    validation_generator_uri_normal = DataGenerator(False, partition['validation'], **params)

    #Input Model for Combined Sequences
    visible1 = Input(shape=(max_len + 1, 1), name = 'Input')

    print(visible1.shape)

    embedded1 = Embedding(vocab_size[0] + 1, input_emb_dim, input_length = (max_len + 1, 1), name = 'Embedding')(visible1)

    embedded1 = Reshape((max_len + 1, input_emb_dim))(embedded1)

    print(embedded1.shape)

    merge = embedded1
    merge = LSTM(lstm_emb_dim, return_sequences=True, name = 'LSTM1')(merge)
    merge = LSTM(lstm_emb_dim, return_sequences=True, name = 'LSTM2')(merge)

    #Output Model for Combined Sequences
    output1 = TimeDistributed(Dense(vocab_size[0] + 1, activation='softmax'), name='Output')(merge)

    modelP = Model(inputs = [visible1], outputs = [output1])
    adam = optimizers.adam(lr=config['MODELPARAMS']['LEARNING_RATE'])
    modelP.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc'])

    print(modelP.summary())

    historyP = modelP.fit_generator(generator=training_generator_uri_normal,
                        validation_data=validation_generator_uri_normal,
                        use_multiprocessing = config['TRAININGPARAMS']['MULTIPROCESSING'],
                        workers = config['TRAININGPARAMS']['WORKERS'],
                        epochs = config['TRAININGPARAMS']['EPOCHS'])

    return modelP


def main():
    print("*****     Starting Preprocessing of Training Data    ******")
    config = loadConfig()

    if not os.path.exists('data/' + sys.argv[2]):
        os.makedirs('data/' + sys.argv[2])
    if not os.path.exists('data/' + sys.argv[2] + '/artefact'):
        os.makedirs('data/' + sys.argv[2] + "/artefact")


    dfN1 = pd.read_csv('data/' + sys.argv[2] + '/' + sys.argv[3], na_values = ['no info', '.'], parse_dates=['timestamp'])

    #Remove request if Remote Addresses (IP) appear 10 or less times
    #Because the data is not in chronological form, sort by timestamp to get in chronological form
    dfN1 = filterAndSort(dfN1)


    #Prepares A1 and N1 for sequentializing
    df_normal = prepDataFrame(dfN1, config['variablesHash']['agentHashRange'], config['variablesHash']['queryHashRange'])
    significantNormal = getSignificantRequest(df_normal, config['variablesHash']['inputHashThreshold'])

    #Sequentializes A1 and N1
    df_normal = sequentializeDataFrame(df_normal, significantNormal, None, config['variablesHash']['inputHashRange'])

    #Prepares the Histogram for Agent and Country
    df_normal['Histo'] = df_normal.apply(lambda row: getCountryAgentPair(row['AgentSeq'], row['CountrySeq']), axis=1)

    #Get Agent
    df_normal['Agent'] = df_normal.apply(lambda row: getAgentOnly(row['AgentSeq']), axis=1)

    #Get Country
    df_normal['Country'] = df_normal.apply(lambda row: getCountryOnly(row['CountrySeq']), axis=1)

    #Save the dataframe as artefacts.
    df_normal.to_csv(r'' + 'data/' + sys.argv[2] + '/' + 'artefact' + '/' + 'N1.csv', index = None, header=True)

    print("*****     Ending Preprocessing     ******")

    print("*****     Starting Training     ******")
    config = loadConfig()
    max_len = config['SEQUENCELENGTH']

    tokenizer_normal = getTokenizer(df_normal)

    df_normal_embedded = df_normal.copy()

    df_normal_embedded['Input'] = tokenizer_normal.texts_to_sequences(df_normal['Input'].values)

    x_normal, y_normal = createGeneratorData(df_normal_embedded, tokenizer_normal, max_len)

    np.save('data/' + sys.argv[2] + '/' + 'artefact' + '/' + 'normalURITraining.npy', x_normal)
    np.save('data/' + sys.argv[2] + '/' + 'artefact' + '/' + 'normalURILabel.npy', y_normal)

    print("*****     Training Model P     ******")
    modelP = trainModelP(len(df_normal), max_len, [len(tokenizer_normal.word_index)], config)

    modelP.save('data/' + sys.argv[2] + '/' + 'artefact' + '/' + 'modelP')

    # saving normal
    with open('data/' + sys.argv[2] + '/' + 'artefact' + '/' + 'tokenizer_normal.pickle', 'wb') as handle:
        pickle.dump(tokenizer_normal, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("*****     Ending Training     ******")

if __name__ == "__main__":
    main()

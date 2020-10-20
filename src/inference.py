import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
import tensorflow as tf
import pickle
import sys
import yaml
import os
from copy import deepcopy
from sklearn import preprocessing

#Forces usage of CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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
from keras.models import load_model


import datetime
import math
import hashlib
import time
from src.datagenerator import DataGenerator

## Added from 18_5
#Disable randomization
seed_value=2020 #

os.environ['PYTHONHASHSEED'] = str(seed_value)
np.random.seed(seed_value)
tf.set_random_seed(seed_value)

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
session_conf.gpu_options.allow_growth = True
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
##

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

    return x, y

# Compute probability of occurence of a sentence
def seqPrepX(sentence, pair, model, tokenizer, histogram, datasize, maxlen) :
    #Assigns indices to each request
    tok = tokenizer.texts_to_sequences([str(sentence)])[0]

    #Prepare sentence to get input and actual output of array
    #x_test and y_test are lists of length 21.
    x_test, y_test = prepare_sentence(tok, maxlen, tokenizer)

    x_test.shape = [maxlen + 1 ,1]

    return x_test


def seqPrepY(sentence, pair, model, tokenizer, histogram, datasize, maxlen) :
    #Assigns indices to each request
    tok = tokenizer.texts_to_sequences([str(sentence)])[0]

    #Prepare sentence to get input and actual output of array
    #x_test and y_test are lists of length 21.
    x_test, y_test = prepare_sentence(tok, maxlen, tokenizer)

    return y_test

def calcProbability(pair, agent, country, histogram, agenthistogram, countryhistogram, datasize, maxlen, p_pred_normal, y_test) :
    #Chain rule to calculate probability of sentence
    p_sentence = 1
    for i in range(maxlen + 1) :
        if (y_test[i] != 0) :
            p_sentence = p_sentence * p_pred_normal[i][y_test[i]]

    #Get probability from Histogram (Old)
    p_hist = getHistogramProbability(histogram, pair, datasize)

    #Get probability from agent (Smoothed)
    agentMiniNum = agenthistogram.min()
    p_agent = getHistogramSmoothedProbability(agenthistogram, agent, datasize, agentMiniNum)

    #Get probability from country (Smoothed)
    countryMiniNum = countryhistogram.min()
    p_country = getHistogramSmoothedProbability(countryhistogram, country, datasize, countryMiniNum)

    p_hist_new = p_agent * p_country
    final_p_hist = (p_hist + p_hist_new)/2

    if p_sentence != 0:
        return 0.5 * math.log(p_sentence) + math.log(final_p_hist)
    else:
        return math.log(final_p_hist)


def calcLogLstm(pair, agent, country, histogram, agenthistogram, countryhistogram, datasize, maxlen, p_pred_normal, y_test) :
    #Chain rule to calculate probability of sentence
    p_sentence = 1
    for i in range(maxlen + 1) :
        if (y_test[i] != 0) :
            p_sentence = p_sentence * p_pred_normal[i][y_test[i]]

    return math.log(p_sentence)

def calcLogPHist(pair, agent, country, histogram, agenthistogram, countryhistogram, datasize, maxlen, p_pred_normal, y_test) :
    #Get probability from Histogram (Old)
    p_hist = getHistogramProbability(histogram, pair, datasize)

    #Get probability from agent (Smoothed)
    agentMiniNum = agenthistogram.min()
    p_agent = getHistogramSmoothedProbability(agenthistogram, agent, datasize, agentMiniNum)

    #Get probability from country (Smoothed)
    countryMiniNum = countryhistogram.min()
    p_country = getHistogramSmoothedProbability(countryhistogram, country, datasize, countryMiniNum)

    p_hist_new = p_agent * p_country
    final_p_hist = (p_hist + p_hist_new)/2

    # return p_sentence * final_p_hist
    return math.log(final_p_hist)


# This might give the probability as 0
def getHistogramProbability(histogram, value, datasize):
    if value not in histogram :
        return 0
    else :
        return histogram[value]/datasize

def getMultiplier(histogram, mean, std, value, datasize, T, W):
    percentage = histogram[value]/datasize

    if percentage <=  mean:
        return 1
    elif percentage <=  mean + std:
        return 0.8
    elif percentage <=  mean + 2*std:
        return 0.6
    elif percentage <= mean + 3*std:
        return 0.4
    else:
        return 0.1


# This ensures that there is at least some probability
def getHistogramSmoothedProbability(histogram, value, datasize, minimumNum):
    if value not in histogram :
        return (minimumNum/2)/datasize
    else :
        return histogram[value]/datasize

#Attack Scores
def calculateSequenceScores(modelP, tokenizer_normal, df_normal, df_attack, maxlen, T, W, config) :
    scoreDictionary = dict()
    count = 0
    df_temp = df_attack[['IP','Input', 'Input2', 'Histo', 'Agent', 'Country']].copy()

    norm_size = len(df_normal)
    atk_size = len(df_attack)

    IP_Histogram_Normal = df_normal['IP'].value_counts()
    IP_Histogram_Attack = df_attack['IP'].value_counts()

    CountryAgentHistogram_Normal = df_normal['Histo'].value_counts()
    CountryAgentHistogram_Attack = df_attack['Histo'].value_counts()

    AgentHistogram_Normal = df_normal['Agent'].value_counts()
    AgentHistogram_Attack = df_attack['Agent'].value_counts()

    CountryHistogram_Normal = df_normal['Country'].value_counts()
    CountryHistogram_Attack = df_attack['Country'].value_counts()


    print('c1')
    df_temp['Px'] = df_temp.apply(lambda x: seqPrepX(x["Input2"], x["Histo"], modelP, tokenizer_normal, CountryAgentHistogram_Normal, norm_size, maxlen), axis=1)
    df_temp['Py'] = df_temp.apply(lambda x: seqPrepY(x["Input2"], x["Histo"], modelP, tokenizer_normal, CountryAgentHistogram_Normal, norm_size, maxlen), axis=1)

    print('c2')

    Pint = modelP.predict(np.array(df_temp['Px'].tolist()), batch_size=5000)

    df_temp['Pint'] = Pint.tolist()

    # just to get range
    df_temp['logHist'] = df_temp.apply(lambda x: calcLogPHist(x["Histo"], x["Agent"], x["Country"], CountryAgentHistogram_Normal, CountryHistogram_Normal, AgentHistogram_Normal, norm_size, maxlen, x['Pint'], x['Py']),axis=1)

    df_temp['logLSTM'] = df_temp.apply(lambda x: calcLogLstm(x["Histo"], x["Agent"], x["Country"], CountryAgentHistogram_Normal, CountryHistogram_Normal, AgentHistogram_Normal, norm_size, maxlen, x['Pint'], x['Py']),axis=1)

    # changed
    print("Histogram with mean",CountryAgentHistogram_Normal.mean()/norm_size,"and standard deviation",CountryAgentHistogram_Normal.mean()/norm_size)
    df_temp['multiplier'] = df_temp.apply(lambda x: getMultiplier(IP_Histogram_Attack, CountryAgentHistogram_Normal.mean(), CountryAgentHistogram_Normal.std(), x["IP"], atk_size, T, W),axis=1)


    # old
    # df_temp['P'] = df_temp.apply(lambda x: calcProbability(x["Histo"], x["Agent"], x["Country"], CountryAgentHistogram_Normal, CountryHistogram_Normal, AgentHistogram_Normal, norm_size, maxlen, x['Pint'], x['Py']) * x['multiplier'],axis=1)

    # changed
    df_temp['P'] = df_temp.apply(lambda x: calcProbability(x["Histo"], x["Agent"], x["Country"], CountryAgentHistogram_Normal, CountryHistogram_Normal, AgentHistogram_Normal, norm_size, maxlen, x['Pint'], x['Py']), axis=1)

    df_min_max = deepcopy(df_temp[['P']])
    x = df_min_max.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df_min_max = pd.DataFrame(x_scaled)
    df_temp['P'] = df_min_max

    df_temp['P'] = df_temp['P'] * df_temp['multiplier']



    print('c3')
    df_temp = df_temp.groupby(['IP'])
    print('c4')
    sr = df_temp['P'].agg(lambda x: min(list(x)))
    print('c5')
    sr = sr.reset_index()
    print('/////////////////')
    return sr


def main():
    print("*****     Starting Preprocessing of Inference Data    ******")
    config = loadConfig()

    if not os.path.exists('data/' + sys.argv[2] + '/result'):
        os.makedirs('data/' + sys.argv[2] + "/result")

    dfN1 = pd.read_csv('data/' + sys.argv[2] + '/' + sys.argv[3], na_values = ['no info', '.'], parse_dates=['timestamp'])
    dfA1 = pd.read_csv('data/' + sys.argv[2] + '/' + sys.argv[4], na_values = ['no info', '.'], parse_dates=['timestamp'])

    #Remove request if Remote Addresses (IP) appear 10 or less times
    #Because the data is not in chronological form, sort by timestamp to get in chronological form
    dfA1 = filterAndSort(dfA1)
    dfN1 = filterAndSort(dfN1)


    #Prepares A1 and N1 for sequentializing
    df_normal = prepDataFrame(dfN1, config['variablesHash']['agentHashRange'], config['variablesHash']['queryHashRange'])
    significantNormal = getSignificantRequest(df_normal, config['variablesHash']['inputHashThreshold'])

    df_attack = prepDataFrame(dfA1, config['variablesHash']['agentHashRange'], config['variablesHash']['queryHashRange'])
    significantAttack = getSignificantRequest(df_attack, config['variablesHash']['inputHashThreshold'])

    #Sequentializes A1 and N1
    df_attack = sequentializeDataFrame(df_attack, significantAttack, significantNormal, config['variablesHash']['inputHashRange'])

    #Prepares the Histogram for Agent and Country
    df_attack['Histo'] = df_attack.apply(lambda row: getCountryAgentPair(row['AgentSeq'], row['CountrySeq']), axis=1)

    #Get Agent
    df_attack['Agent'] = df_attack.apply(lambda row: getAgentOnly(row['AgentSeq']), axis=1)

    #Get Country
    df_attack['Country'] = df_attack.apply(lambda row: getCountryOnly(row['CountrySeq']), axis=1)

    #Save the dataframe as artefacts.
    df_attack.to_csv(r'' + 'data/' + sys.argv[2] + '/' + 'artefact' + '/' + 'A1.csv', index = None, header=True)

    print("*****     Ending Preprocessing     ******")

    print("*****     Starting Inferencing     ******")
    config = loadConfig()

    # load models
    modelP = load_model('data/' + sys.argv[2] + '/' + 'artefact' + '/' + 'modelP')

    # load dataset
    df_normal = pd.read_csv('data/' + sys.argv[2] + '/' + 'artefact' + '/' + 'N1.csv')
    df_attack = pd.read_csv('data/' + sys.argv[2] + '/' + 'artefact' + '/' + 'A1.csv')

    # loading tokenizer of normal
    with open('data/' + sys.argv[2] + '/' + 'artefact' + '/' + 'tokenizer_normal.pickle', 'rb') as handle:
        tokenizer_normal = pickle.load(handle)

    # loading tokenizer of attack
    start = time.time()
    sr = calculateSequenceScores(modelP, tokenizer_normal, df_normal, df_attack, config['SEQUENCELENGTH'], config['T'], config['W'], config)
    sr_P = sr[['IP', 'P']].copy()

    # df_min_max = deepcopy(sr_P[['P']])
    # x = df_min_max.values #returns a numpy array
    # min_max_scaler = preprocessing.MinMaxScaler()
    # x_scaled = min_max_scaler.fit_transform(x)
    # df_min_max = pd.DataFrame(x_scaled)
    # sr_P['P'] = df_min_max
    sr_P = sr_P.sort_values(by = ['P'])

    print(time.time() - start)

    # store the data as binary data stream
    sr_P.to_csv('data/' + sys.argv[2] + '/' + 'result' + '/' + 'PScore_' + sys.argv[4])
    sr_P.to_pickle('data/' + sys.argv[2] + '/' + 'result' + '/' + 'PScore' + sys.argv[4][:-4])
    print("*****     Ending Inferencing     ******")

if __name__ == "__main__":
    main()

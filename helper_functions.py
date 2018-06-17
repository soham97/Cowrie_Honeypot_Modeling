import numpy as np
import pandas as pd
from math import log
from tqdm import tqdm_notebook as tqdm
import pickle
from sklearn.preprocessing import LabelEncoder
import hmm

def load_pickle(name):
    with open(name,'rb') as file:
        k = pickle.load(file)
    print('pickle loading complete')
    return k

def process_data(data1,data2,data3):
    df = pd.concat([data1,data2,data3],axis=0)
    le = LabelEncoder()
    df['eventid'] = le.fit_transform(df['eventid'].astype('str').values)
    data_agg = df.groupby('session',as_index=False).agg(lambda x: x.tolist())
    agg = pd.DataFrame()
    agg['eventid'] = data_agg['eventid'].values
    return agg, le

def desired_seq(df,length):
    data = []
    for i in tqdm(range(0,len(df))):
        if len(df[i]) == length:
            data.append(df[i])
    data = np.array(data)
    print('total samples found: ' + str(len(data)) + '\n')
    print('few samples of data of sequence length ' + str(length))
    print(data[:5])
    return data

def calculate_probablity(mod,df):
    prob = []
    seq = []
    for i in tqdm(range(0,len(df))):
        seq.append(df[i])
        prob.append(mod.full_prob(mod.forward(df[i])))
    data = pd.DataFrame()
    data['seq'] = seq
    data['prob'] = prob
    print('result samples: \n')
    print(data.head())
    return data

def sample_per_seq(df,n):
    samples = pd.DataFrame()
    for index in tqdm(range(0,n)):
        samples.loc[index,'length'] = index
        data = []
        length = index
        for i in range(0,len(df)):
            if len(df[i]) == length:
                data.append(df[i])
        samples.loc[index,'no of samples'] = len(data)
    samples = samples.sort_values('no of samples',ascending = False)
    samples = samples.reset_index(drop=True)
    print('Top maximum number of sequence length: \n')
    print(samples.head(10))
    print('\n Further analysis of results:  \n')
    print(samples.describe())
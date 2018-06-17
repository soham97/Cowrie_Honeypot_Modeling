import numpy as np
import pandas as pd
from math import log
from tqdm import tqdm_notebook as tqdm
from sklearn.preprocessing import LabelEncoder

def load_pickle(name):
    with open('name','rb') as file:
        k = pickle.load(file)
    return k

def process_data(df):
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
    return data

def calculate_probablity(df):
    prob = []
    seq = []
    for i in tqdm(range(0,len(df))):
        seq.append(df[i])
        prob.append(mod.full_prob(mod.forward(df[i])))
    data = pd.DataFrame()
    data['seq'] = seq
    data['prob'] = prob
    return data
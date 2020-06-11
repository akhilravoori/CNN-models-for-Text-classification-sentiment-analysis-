#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 16:06:44 2020

@author: akhil
"""


import pandas as pd


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
df.to_csv('classified_corona.csv')   
df.groupby('tone').count().tweet_text
    
    
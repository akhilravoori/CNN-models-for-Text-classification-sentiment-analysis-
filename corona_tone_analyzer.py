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

authenticator = IAMAuthenticator('bjs7HIbH9SNgnpDIDy6rx3Ynmb_5SPetzmaXXv7t3_KL')
tone_analyzer = ToneAnalyzerV3(
    version='2020-03-26',
    authenticator=authenticator
)
tone_analyzer.set_service_url('https://api.eu-gb.tone-analyzer.watson.cloud.ibm.com/instances/33756fab-977f-4818-8b4c-f8bf06e8e91e')
df=pd.read_csv('/home/akhil/Desktop/Projects/inf_project/t5.csv')
df1=pd.read_csv('/home/akhil/Desktop/Projects/inf_project/covid2019.csv')
x=df.tweet_text
tweet=x[0]
result = re.sub(r"http\S+", "", tweet)
print(result)
y=result.replace('#','')
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
df.to_csv('coronatones.csv')
    
    
    

    

    
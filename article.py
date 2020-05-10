#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 01:00:17 2020

@author: akhil
"""


import json
import pandas as pd
import nltk
import re
import tokenizers
from ibm_watson import ToneAnalyzerV3
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import os

authenticator = IAMAuthenticator('bjs7HIbH9SNgnpDIDy6rx3Ynmb_5SPetzmaXXv7t3_KL')
tone_analyzer = ToneAnalyzerV3( version='2020-03-26',authenticator=authenticator)
   

tone_analyzer.set_service_url('https://api.eu-gb.tone-analyzer.watson.cloud.ibm.com/instances/33756fab-977f-4818-8b4c-f8bf06e8e91e')
path=os.getcwd()
tones=[]
df=pd.DataFrame()
article_content=[]
for i in range(1,10):
    x=open(path+"/Articles/output0"+str(i)+'.txt','r+')
    t=x.read()
    t=t.replace('\n','')
    tone_analysis = tone_analyzer.tone(
            {'text': t},
            content_type='text/plain',
            sentences=True
        ).get_result()
   
    tones.append(tone_analysis["document_tone"])
print(tones)

for i in range(10,21):
    x=open(path+"/Articles/output"+str(i)+'.txt','r+')
    t=x.read()
    t=t.replace('\n','')
    tone_analysis = tone_analyzer.tone(
            {'text': t},
            content_type='text/plain',
            sentences=True
        ).get_result()
    print(tone_analysis["document_tone"])
    tones.append(tone_analysis["document_tone"])
df['tones']=tones    
print(df)
df.to_csv("Articles.csv")


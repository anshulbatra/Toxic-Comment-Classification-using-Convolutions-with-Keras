#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 16:18:53 2020

@author: anshul
"""

#Importing
import pandas as pd

#Loading Dataset
train_df = pd.read_csv('train.csv').fillna(' ')
print(train_df.sample(10, random_state=1))

x = train_df['comment_text'].values
print(x)
#List of string comments

#LOCATING COMMENTS HAVING TOXIC LABEL=1
print(train_df.loc[train_df['toxic']==1].sample(10, random_state=10))
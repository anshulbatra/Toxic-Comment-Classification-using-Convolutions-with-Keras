#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 18:00:34 2020

@author: anshul
"""

import pandas as pd
from tensorflow.keras.preprocessing import text, sequence

#Loading Dataset
train_df = pd.read_csv('train.csv').fillna(' ')

x = train_df['comment_text'].values             #List of comments
#print(x)


#Tokenizing string data to numeric data
max_features = 20000
max_text_length = 400            #Rest of the comment will be padded

x_tokenizer = text.Tokenizer(max_features)
x_tokenizer.fit_on_texts(x)

x_after_tokenization = x_tokenizer.texts_to_sequences(x)

x_train_val = sequence.pad_sequences(x_after_tokenization, maxlen=max_text_length)
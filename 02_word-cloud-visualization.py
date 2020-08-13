#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 17:08:58 2020

@author: anshul
"""

import pandas as pd
from  wordcloud import WordCloud, STOPWORDS
from matplotlib import pyplot as plt           #Or matplotlib.pyplot as plt

#Loading Dataset
train_df = pd.read_csv('train.csv').fillna(' ')

comments = train_df['comment_text'].loc[ train_df['toxic']==1 ].values
wordC = WordCloud(width=640,
                  height=640,
                  background_color='black',
                  stopwords = STOPWORDS).generate(str(comments))

fig = plt.figure(figsize=(12,8),
                 facecolor='k',
                 edgecolor='k')

plt.imshow(wordC, interpolation='bilinear')
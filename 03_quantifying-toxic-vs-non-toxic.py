#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 17:34:36 2020

@author: anshul
"""

import pandas as pd

#Loading Dataset
train_df = pd.read_csv('train.csv').fillna(' ')

toxic_label_list = train_df['toxic'].values
print(toxic_label_list)
print(len(toxic_label_list))           #Inbuilt python function works on lists

#Exact toxic entries in data
print(train_df['toxic'].value_counts())        #Pandas function

#Plotting Histogram using pandas plot function
train_df['toxic'].plot(kind='hist',
                       title='Distribution of Toxic Comments')
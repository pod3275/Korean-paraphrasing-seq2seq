# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 11:14:45 2020

@author: 이상헌
"""

import pandas as pd
import numpy as np
import time
from openpyxl import Workbook

name = ["상효", "은지", "두희", "민경", "대한", "영근", "상헌", "근형"]
data = {}
data_length = 0

# method 1-----------------------------------------------
#start = time.time()
#for member in name:
#    df = pd.read_excel("한시에 데이터를 만들자.xlsx", sheet_name = member, 
#                       header=1, na_filter=False)
#    for k in df.keys():
#        if k not in data.keys():
#            data[k] = []
#        data[k] = data[k] + list(df[k])
#        
#    data_length = data_length + len(list(df[k]))
#    
#invalid = []
#for l in range(data_length):
#    for k in data.keys():
#        if k != '단위지식 type' and data[k][l] == '':
#            invalid.append(l)
#
#invalid = list(set(invalid))
#
#data2 = {}
#for k in data.keys():
#    data2[k] = []
#        
#for l in range(data_length):
#    if l not in invalid:
#        for k in data.keys():
#            data2[k].append(data[k][l])
#
##time : 3.0259318351745605 seconds
#print("time :", time.time()-start, "seconds")
##len(data2['train_x'])

# method 2-----------------------------------------------
data = {}
start = time.time()
data_len = []

for member in name:
    count = 0
    df = pd.read_excel("한시에 데이터를 만들자.xlsx", sheet_name = member, 
                       header=1, na_filter=False)
    for _, row in df.iterrows():
        data_text = [row[k] for k in df.keys() if k != '단위지식 type']
        if '' not in data_text:
            for i, k in enumerate(df.keys()):
                if k not in data.keys():
                    data[k] = []
                data[k].append(row[i])
                
            count+=1
    
    data_len.append(count)

#time : 3.0977089405059814 seconds
print("time :", time.time()-start, "seconds")

train_x = data['train_x']
train_y = data['train_y']


# 300개 data 추가
df = pd.read_excel("paraphrasing data_DH.xlsx")
for i in range(len(df)):
    train_x.append(df['train_x'][i])
    train_y.append(df['train_y'][i])

total_len = len(train_x)


# shuffling
idx = np.arange(total_len)
np.random.shuffle(idx)

train_x = list(np.array(train_x)[idx])
train_y = list(np.array(train_y)[idx])


# write .xlsx
wb = Workbook()
sheet1 = wb.active
sheet1.title = 'data'

sheet1.append(['train_x', 'train_y'])
for i in range(total_len):
    sheet1.append([train_x[i], train_y[i]])
    
wb.save(filename='Data/new_data.xlsx')

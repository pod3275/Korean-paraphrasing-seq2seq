# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 11:14:45 2020

@author: 이상헌
"""

import pandas as pd
import numpy as np
import time
from openpyxl import Workbook

import re

ENG_BRC_RGX = re.compile(r'[(][\w?.,~∼ :/]+[)]')
# ENG_BRC_RGX = re.compile(r'[(].*?[)]')
STEP_RGX = re.compile(r'step|Step')
EXCEPT_RGX = re.compile(r'[(][0-9]+[)]')
EXCEPTIONS = ['(수식)','(미지수)','(등호)','(화살표)']

def clean_txt(txt):
    eng_brc_rgx_ = ENG_BRC_RGX.findall(txt)
    step_rgx_ = STEP_RGX.findall(txt)
    except_rgx = EXCEPT_RGX.findall(txt)

    for r_ in eng_brc_rgx_ + step_rgx_:
        if r_ in EXCEPTIONS + except_rgx: continue
        txt = txt.replace(r_, '')

    # E.g. '- 부호가 다른 수' --> '부호가 다른 수'
    if txt != '':
        if txt[0] == '-':
            txt = txt[1:]

    return txt.strip()

def test_clean_txt():
    tcs = ['Step(수식)를 없애기 위하여 (1)에서 (2)를 변끼리 빼면',
            'A반의 학생 수는 27명, B반의 학생 수는 31명이고 철도 박물관 입장료는 (미지수)원이다.',
            '두 번째로, SAS 닮음은 두 쌍의 대응하는 변의 길이의 비가 같고, 끼인각의 크기가 같은 경우에요.',
            '즉, RHS 합동이에요.',
            '소수(素數) : (수식)',
            '오른쪽과 같이 소수를 찾는 방법은 고대 그리스의 수학자인 에라토스테네스(Eratosthenes: B.C. 275 ~ B.C. 194)가 고안한 것이다.',
            '고대 그리스의 천문학자인 히파르코스(Hipparchos, B.C.190?∼B.C.125?)는 맨눈으로 보이는 별들 중 가장 밝게 보이는 별을 1등급, 가장 어둡게 보이는 별을 6등급으로 정하여 별의 밝기를 구분하였어요.',
            '17세기에 드메레(de Méré, C.,   )는 게임을 하던 두 사람이 도중에 게임을 그만두었을 때 내기에 건 돈을 어떻게 나누어야 하는지에 관한 문제를 파스칼(Pascal, B.,   )에게 문의했어요.',
            '''Side(변)와 Angle(각)의 첫 글자를 사용하여 삼각형의 닮음 조건을 간단히
1. SSS 닮음
2. SAS 닮음
3. AA 닮음
으로 나타내기도 한다.''',
            "D는 판별식을 뜻하는 'Discriminant'의 첫 글자이다."]

    for tc in tcs:
        print(clean_txt(tc))


def main():

    name = ["상효", "은지", "두희", "민경", "대한", "영근", "상헌", "근형", "형기"]
    data = {}

    data = {}
    start = time.time()

    total_count = 0
    for member in name:
        count = 0
        df = pd.read_excel("Data/한시에 데이터를 만들자.xlsx", sheet_name = member, 
                           header=1, na_filter=False)
        for _, row in df.iterrows():
            data_text = [row[k] for k in df.keys() \
                            if (k == 'train_x') or (k == 'train_y')]
            if '' not in data_text:
                for i, k in enumerate(df.keys()):
                    if k not in data.keys():
                        data[k] = []
                    data[k].append(clean_txt(row[i]))
                    
                count+=1
        print(member, count)
        total_count += count

    #time : 3.0977089405059814 seconds
    print("한시에 데이터를 만들자: %d" % total_count)
    print("time :", time.time()-start, "seconds")

    train_x = data['train_x']
    train_y = data['train_y']


    # 300개 data 추가
    df = pd.read_excel("Data/paraphrasing data_DH.xlsx")
    for i in range(len(df)):
        train_x.append(clean_txt(df['train_x'][i]))
        train_y.append(clean_txt(df['train_y'][i]))

    total_len = len(train_x)


    # shuffling
    idx = np.arange(total_len)
    print("총 데이터: %d" % total_len)
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
        
    wb.save(filename='Data/paraphrase_data.xlsx')


if __name__ == '__main__':
    # test_clean_txt()
    main()
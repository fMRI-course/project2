# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 00:53:07 2019

@author: Joyce
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


for i in range(1,11):
    data=[]
    for j in range(7):
        load_data=np.load('data_trans_npy/person_'+str(1)+'/sub_'+str(i)+'/'+str(j)+'.npy')
        ravel_data=load_data.ravel()
        data.append(ravel_data)
    #print(len(data[0]))
    #plt.subplots(figsize=(9, 9)) # 设置画面大小

    sns.heatmap(data, annot=True, vmax=1, square=True, cmap="Blues")
    plt.savefig('data_trans_npy/person_'+str(1)+'/sub_'+str(i)+'/'+'corr_'+str(j)+'.jpg',np.corrcoef(data))
    plt.show()
    np.savetxt('data_trans_npy/person_'+str(1)+'/sub_'+str(i)+'/'+'corr_'+str(j)+'.txt',np.corrcoef(data))


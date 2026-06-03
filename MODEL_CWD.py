# -*- coding: utf-8 -*-
"""
Created on Wed May 27 14:21:53 2026

@author: AIFS
"""



import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

data_df = pd.read_excel("CWD_dataset.xlsx")

npdata=data_df.to_numpy()
[numrows,numcols]=np.shape(npdata)


res=npdata[:,10]              # volume estimated from h and d pairs
res[res>=0.006]=1                    # 1==sediment stored, 0=no sediment stored
res[res<0.006]=0
data=npdata

nu_yes=np.sum(data[:,10])
nu_total=len(data[:,10])
nu_no=nu_total-nu_yes


kfold=5
len_no=int(nu_no/kfold)
len_yes=int(nu_yes/kfold)
feature_list = data_df.columns.tolist()

feature_combos_names = ['CWDlength', 'phi', 'FractionGroundContact', 'IsModerate']

number_permutation=10


features_used = []


CWD=data[:,[1, 2, 5, 8,10]]        # combo 3

re=data[:,10]
length_test=re.size
np.random.seed(587)

CWD_datas=CWD[np.random.permutation(length_test),:]




CWD_response=CWD_datas[:,-1]
CWD_data=CWD_datas[:,:-1]





no_total=re.size-np.sum(re)
yes_total=np.sum(re)




print('Model CWD')
print('Features: CWDlength, phi, FractionGroundContact, IsModerate')
model_CWD = LogisticRegression(penalty="l2", C=.3,solver="lbfgs",max_iter=500, class_weight={0:1, 1:np.sqrt((no_total)/(yes_total))},fit_intercept=True)
CWD_s=model_CWD.fit(CWD_data, CWD_response)
print(f"\u03B2\u2080 = {CWD_s.intercept_[0]}")
CWD_betas = CWD_s.coef_.ravel()

print(f"\u03B2\u2081 = {CWD_betas[0]}")
print(f"\u03B2\u2082 = {CWD_betas[1]}")
print(f"\u03B2\u2083 = {CWD_betas[2]}")
print(f"\u03B2\u2084 = {CWD_betas[3]}")






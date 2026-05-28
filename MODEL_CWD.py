# -*- coding: utf-8 -*-
"""
Created on Wed May 27 14:21:53 2026

@author: AIFS
"""



import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

data_df=pd.read_excel(r'D:/filed_work_class/CWD_Results_new.xlsx')

npdata=data_df.to_numpy()
[numrows,numcols]=np.shape(npdata)

# Select a subset of the columns to focus on features that we think will be the best predictors of storing sediment
# You can open the excel file or look at the output from "data.describe()" shown above to see what the different columns correspond to


# Basin response data
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

feature_combos_names = {
   
     "set3":['CWDlength', 'phi', 'FractionGroundContact', 'IsModerate']
}

number_permutation=10
count=0
# Store all model performance results
model_performance = []
permuation_feature_imp = []
all_importances = []
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
model_CWD = LogisticRegression(penalty="l2", C=.3,solver="lbfgs",max_iter=500, class_weight={0:1, 1:np.sqrt((no_total)/(yes_total))},fit_intercept=True)
CWD_s=model_CWD.fit(CWD_data, CWD_response)
CWD_beta0=print(CWD_s.intercept_)
CWD_beta2=print( CWD_s.coef_)






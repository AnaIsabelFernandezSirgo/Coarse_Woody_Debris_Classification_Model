# -*- coding: utf-8 -*-
"""
Created on Thu May 28 12:24:56 2026

@author: AIFS
"""


import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression


from sklearn.inspection import partial_dependence
import matplotlib.pyplot as plt
import shap
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


feature_combos_names = ['CWDlength', 'phi', 'FractionGroundContact', 'IsModerate']




C=data[:,[1, 2, 5, 8,10]]      

re=data[:,10]
length_test=re.size
np.random.seed(587)

C_datas=C[np.random.permutation(length_test),:]


C_response=C_datas[:,-1]
C_data=C_datas[:,:-1]




no_total=re.size-np.sum(re)
yes_total=np.sum(re)





## C

model_C = LogisticRegression(penalty="l2", C=.3,solver="lbfgs",max_iter=500, class_weight={0:1, 1:np.sqrt((no_total)/(yes_total))},fit_intercept=True)
C_s=model_C.fit(C_data, C_response)
explainer_C =shap.Explainer(C_s.predict, C_data)
shap_values_C = explainer_C(C_data)
shap_values_C.feature_names = feature_combos_names
shap.plots.bar(shap_values_C)
shap.plots.beeswarm(shap_values_C)

pd_C_1 = partial_dependence(C_s, C_data, features=0)
df1_C = np.concatenate([pd_C_1.average,pd_C_1.grid_values])
pd_C_2 = partial_dependence(C_s, C_data, features=1)
df2_C = np.concatenate([pd_C_2.average,pd_C_2.grid_values])
pd_C_3 = partial_dependence(C_s, C_data, features=2)
df3_C = np.concatenate([pd_C_3.average,pd_C_3.grid_values])
pd_C_4 = partial_dependence(C_s, C_data, features=3)
df4_C = np.concatenate([pd_C_4.average,pd_C_4.grid_values])



plt.plot(df1_C[1,:], df1_C[0,:], color='black', linewidth=3.0)

plt.xlabel(feature_combos_names[0],fontsize=18)
plt.ylabel("Partial Dependence",fontsize=18)
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the tick labels

plt.show()


plt.plot(df2_C[1,:], df2_C[0,:], color='black', linewidth=3.0)

plt.xlabel(r'$\phi$',fontsize=18)
plt.ylabel("Partial Dependence",fontsize=18)
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the tick labels

plt.show()



plt.plot(df3_C[1,:], df3_C[0,:], color='black', linewidth=3.0)

plt.xlabel(feature_combos_names[2],fontsize=18)
plt.ylabel("Partial Dependence",fontsize=18)
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the tick labels

plt.show()



plt.plot(df4_C[1,:], df4_C[0,:], color='black', linewidth=3.0)

plt.xlabel(feature_combos_names[3],fontsize=18)
plt.ylabel("Partial Dependence",fontsize=18)
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the tick labels

plt.show()


# -*- coding: utf-8 -*-
"""
Created on Wed May 27 14:28:51 2026

@author: AIFS
"""


import numpy as np
import pandas as pd
import sklearn

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score



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

number_permutation=100



features_used = []

data_new = data[:, [1, 2, 5, 8,10] ] 


results_final=[]
results_avg = np.empty((4,100,12))
results_avg1 = np.empty((4,12))


logisticRegr = LogisticRegression(penalty="l2", C=.3,solver="lbfgs",max_iter=500, class_weight={0:1, 1:np.sqrt((nu_no)/(nu_yes))},fit_intercept=True)
for nf in range (0,4):
    r=data_new.shape[1]-1
    data_0 = data_new[data_new[:, r] == 0]   
    data_1 = data_new[data_new[:, r] == 1] 
    np.random.seed(13)   # or any seed you want
    
    XY_no= data_0[np.random.permutation(data_0.shape[0])]
    XY_yes = data_1[np.random.permutation(data_1.shape[0])]

    
    
    for npp in range (0,number_permutation):
        np.random.seed(13+npp*3)
        results = np.empty((kfold,12))
        
        XY_no=XY_no[np.random.permutation(XY_no.shape[0])]
        XY_yes=XY_yes[np.random.permutation(XY_yes.shape[0])]
        for nk in range (0,kfold): 
            XY_test_no=XY_no[nk*len_no: len_no*(nk+1),:]
            XY_test_yes=XY_yes[nk*len_yes: len_yes*(nk+1),:]
            XY_test=np.vstack([XY_test_no,XY_test_yes])
            XY_test=XY_test[np.random.permutation(XY_test.shape[0])]
            X_test=XY_test[:,0:r]
            X_test[:, nf] = np.random.permutation(X_test[:, nf])
            Y_test=XY_test[:,[r]].ravel()
            XY_train_no  = np.delete(XY_no,  slice(nk*len_no,  len_no*(nk+1)),  axis=0)
            XY_train_yes = np.delete(XY_yes, slice(nk*len_yes, len_yes*(nk+1)), axis=0)
            XY_train=np.vstack([XY_train_no,XY_train_yes])
            XY_train=XY_train[np.random.permutation(XY_train.shape[0])]
            X_train=XY_train[:,0:r]
            Y_train=XY_train[:,[r]].ravel()
            model = logisticRegr.fit(X_train, Y_train)
        
            # Predict on test data
            y_pred = logisticRegr.predict(X_test)
        
            # Calculate accuracy and store it
            accuracy = accuracy_score(Y_test, y_pred)
        
            confmat = sklearn.metrics.confusion_matrix(Y_test, y_pred)
        
            tp=confmat[1,1]
            tn=confmat[0,0]
            fp=confmat[0,1]
            fn=confmat[1,0]
        
            TS=confmat[1,1]/(confmat[1,1]+confmat[0,1]+confmat[1,0])
            
            if tp+fp==0:
                continue
            precision=tp/(tp+fp)
            recall=tp/(tp+fn)
            f1score=2*tp/(2*tp+fp+fn)
            sensitivity=tp/(tp+fn)    
            specificity=tn/(tn+fp)
            falseposrate=fp/(fp+tn)
            
            results[nk,:]=([accuracy,tp,tn,fp,fn,TS,precision,recall,f1score,sensitivity,specificity,falseposrate])
        
        results_avg[nf,npp, :] = np.mean(results, axis=0)

    results_avg1[nf,  :] = np.mean(results_avg[nf,:,:], axis=0)




    


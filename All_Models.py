# -*- coding: utf-8 -*-
"""
Created on Wed May 27 23:58:37 2026

@author: AIFS
"""


import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from itertools import combinations
#from sklearn.metrics import plot_confusion_matrix

data_df=pd.read_excel(r"D:/CWD/CWD_dataset.xlsx")
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
nu_no=nu_total-nu_yes                 # 1==sediment stored, 0=no sediment stored




kfold=5
len_no=int(nu_no/kfold)
len_yes=int(nu_yes/kfold)

feature_list = data_df.columns.tolist()
remove_these = ['PlotNum','Volume','Volume_alpha_1deg','Volume_alpha_3deg']
feature_list = [f for f in feature_list if f not in remove_these]
cols_to_remove = [0,10,11,12]
data_new = np.delete(data, cols_to_remove, axis=1)



features_used = []
results_final=[]

n_features = data_new.shape[1]
number_permutation=10

logisticRegr = LogisticRegression(penalty="l2", C=.3,solver="lbfgs",max_iter=500, class_weight={0:1, 1:np.sqrt((nu_no)/(nu_yes))},fit_intercept=True)
with pd.ExcelWriter("All_Models.xlsx") as w:
    for r in range(3, 8):
        print(r)
        for combo in combinations(range(n_features), r):
            subset = list(combo)
            feature_names = [feature_list[i] for i in subset]
            features_used.append(feature_names)
            
    
            data_subset=data_new[:, subset]
            data_subset=np.column_stack((data_subset, data[:,10]))
            idx = np.argsort(data_subset[:, r])   
            data_sorted = data_subset[idx]
            data_0 = data_subset[data_subset[:, r] == 0]   # all rows where col 39 == 0
            data_1 = data_subset[data_subset[:, r] == 1] 
            np.random.seed(13)   
    
            XY_no= data_0[np.random.permutation(data_0.shape[0])]
            XY_yes = data_1[np.random.permutation(data_1.shape[0])]
    
        
            col_avg=np.empty((number_permutation,12))
            for p in range (0,number_permutation):
                np.random.seed(13+p*3)
                results = np.empty((kfold,12))
                
                XY_no=XY_no[np.random.permutation(XY_no.shape[0])]
                XY_yes=XY_yes[np.random.permutation(XY_yes.shape[0])]
                for k in range (0,kfold): 
                    XY_test_no=XY_no[k*len_no: len_no*(k+1),:]
                    XY_test_yes=XY_yes[k*len_yes: len_yes*(k+1),:]
                    XY_test=np.vstack([XY_test_no,XY_test_yes])
                    XY_test=XY_test[np.random.permutation(XY_test.shape[0])]
                    X_test=XY_test[:,0:r]
                    Y_test=XY_test[:,[r]].ravel()
                    XY_train_no  = np.delete(XY_no,  slice(k*len_no,  len_no*(k+1)),  axis=0)
                    XY_train_yes = np.delete(XY_yes, slice(k*len_yes, len_yes*(k+1)), axis=0)
                    XY_train=np.vstack([XY_train_no,XY_train_yes])
                    XY_train=XY_train[np.random.permutation(XY_train.shape[0])]
                    X_train=XY_train[:,0:r]
                    Y_train=XY_train[:,[r]].ravel()
                    model = logisticRegr.fit(X_train, Y_train)
                
                    
                    y_pred = logisticRegr.predict(X_test)
                
                    
                    accuracy = accuracy_score(Y_test, y_pred)
                
                    confmat = sklearn.metrics.confusion_matrix(Y_test, y_pred)
                
                    tp=confmat[1,1]
                    tn=confmat[0,0]
                    fp=confmat[0,1]
                    fn=confmat[1,0]
                
                    TS=confmat[1,1]/(confmat[1,1]+confmat[0,1]+confmat[1,0])
                    #print("Threat Score: " + str(round(TS,2)))
                    if tp+fp==0:
                        continue
                    precision=tp/(tp+fp)
                    recall=tp/(tp+fn)
                    f1score=2*tp/(2*tp+fp+fn)
                    sensitivity=tp/(tp+fn)    # also known as true positive rate
                    specificity=tn/(tn+fp)
                    falseposrate=fp/(fp+tn)
                
                    
                    results[k,:]=([accuracy,tp,tn,fp,fn,TS,precision,recall,f1score,sensitivity,specificity,falseposrate])
                
                col_avg[k,:] = np.mean(results, axis=0)
            col_avg2=np.mean(col_avg, axis=0)
            final_row = [feature_names] + col_avg2.tolist()
            results_final.append(final_row)
                
        results_pd= pd.DataFrame(results_final)           
        
        results_pd.columns=['Features','Accuracy','TP','TN','FP','FN','TS','Precision','Recall','F1 Score','Sensitivity','Specificity','FPR']
        
        
        results_pd.to_excel(w, sheet_name=f"{r} Features", index=False)

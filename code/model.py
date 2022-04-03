import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score,recall_score,accuracy_score,f1_score,matthews_corrcoef,roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
import os
from sklearn.model_selection import StratifiedKFold
from mrmr import mrmr_classif
from sklearn import preprocessing
from sklearn.metrics import auc
from sklearn.model_selection import cross_val_predict
import multiprocessing as mp
from sklearn.model_selection import GridSearchCV
from math import sqrt
from tqdm import tqdm, trange
from itertools import combinations
import sys
sys.path.append(r'F:\\zhulin\\monsh\\paper-anti-CRISPR\\latex\\outcomes\\vot_outcome\\anti_CRISPR\\code')
import data ### read data from data.py

###doing feature selection and normalization with mrmr mrthod
def mrmr_feature_selection(data):
    x,y=data.iloc[:,1:],data.iloc[:,0]
    x=pd.DataFrame(preprocessing.minmax_scale(x),columns=x.columns)
    y=pd.Series(list(y))
    if x.columns.size>200:
        x=x[mrmr_classif(X=x,y=y, K = 200)]
    x.insert(0,'class',list(y))
    return x 

###performance assessment
def metrix(proba_y,verified_y):
    fpr,tpr,threshols=roc_curve(list(verified_y),list(proba_y))
    aucs=auc(fpr,tpr)#AUC value
    
    pred_y=np.where(np.array(proba_y)>=0.5,1,np.array(proba_y))
    pred_y=np.where(pred_y<0.5,0,pred_y)
    #SN,SP,PRE,ACC,F-Score,MCC
    SN=recall_score(verified_y,pred_y)
    PRE=precision_score(verified_y,pred_y)
    ACC=accuracy_score(verified_y, pred_y)
    F_score=f1_score(verified_y,pred_y)
    MCC=matthews_corrcoef(verified_y, pred_y)
    tn, fp, fn, tp = confusion_matrix(verified_y, pred_y).ravel()
    SP= tn / (tn+fp)
    per=np.array([PRE,SN,SP,F_score,ACC,MCC,aucs])
    return per,fpr,tpr

###build ensmeble model
def ensemble_model(train_data):
    x_train,y_train=train_data.iloc[:,1:],train_data.iloc[:,0]
    max_par=int(max(x_train.columns.size/2,sqrt(x_train.columns.size)))
    model_name=[]
    model_str=['SVM','KNN','RF','MLP','LR','XGB','Light']
    #SVM
    par=[2**i for i in range(-6,7)]
    param_grid_sv=[{'kernel':['rbf'],'gamma':par,'C':par},{'kernel':['linear'],'C':par}]
    svm_clf=GridSearchCV(SVC(probability=True),param_grid_sv,cv=10,n_jobs=-1).fit(x_train,y_train)
    svm=svm_clf.best_estimator_.fit(x_train,y_train)
    model_name.append(svm)
    #KNN
    param_grid_knn={'n_neighbors':range(1,max_par)}
    knn_clf=GridSearchCV(KNeighborsClassifier(),param_grid_knn,cv=10,n_jobs=-1).fit(x_train,y_train)
    knn=knn_clf.best_estimator_.fit(x_train,y_train)
    model_name.append(knn)
    #RF
    param_grid_rf={'n_estimators':range(1,max_par),'max_features':range(1,20,5)}
    rf_clf=GridSearchCV(RandomForestClassifier(),param_grid_rf,cv=10,n_jobs=-1).fit(x_train,y_train)
    rf=rf_clf.best_estimator_.fit(x_train,y_train)
    model_name.append(rf)
    #MLP
    mlp=MLPClassifier(hidden_layer_sizes=[64,32],max_iter=1000).fit(x_train,y_train)
    model_name.append(mlp)
    #LR
    lr=LogisticRegression().fit(x_train,y_train)
    model_name.append(lr)
    #XGB
    XGB=XGBClassifier(learning_rate=0.1,eval_metric=['logloss','auc','error'],use_label_encoder=False,objective="binary:logistic").fit(x_train,y_train)
    model_name.append(XGB)
    #Light
    light=LGBMClassifier().fit(x_train,y_train)
    model_name.append(light)
    #ensemble
    ensem = VotingClassifier(estimators=list(zip(model_str,model_name)),voting='soft',weights=[1]*(len(model_name))).fit(x_train,y_train) 
    return ensem
####
def get_result(train_data,test_data):   
    tpr_all,valida_per_all,y_proba_test_all=[],[],[]
    mean_fpr=np.linspace(0,1,100)
     
    kf=StratifiedKFold(n_splits=5) ###5-fold-cross-validation
    x,y=train_data[0].iloc[:,1:],train_data[0].iloc[:,0]  
    for train_site, valida_site in tqdm(kf.split(x,y)):
        y_proba_test_ense,y_proba_valid_ense=[],[]
        
        for i in range(0,len(train_data)):
            train,valida=mrmr_feature_selection(train_data[i].iloc[train_site,:]),mrmr_feature_selection(train_data[i].iloc[valida_site,:])
            test=mrmr_feature_selection(test_data[i])
            ensemble_clf=ensemble_model(train)
            y_proba_valid=ensemble_clf.predict_proba(valida.iloc[:,1:])[:,1]
            y_proba_test=ensemble_clf.predict_proba(test.iloc[:,1:])[:,1]
            y_proba_valid_ense.append(y_proba_valid)
            y_proba_test_ense.append(y_proba_test)  
             
        y_proba_valid_once=np.mean(y_proba_valid_ense,axis=0) ##ensemble three features 
        y_proba_test_once=np.mean(y_proba_test_ense,axis=0)
        ##
        valida_per_once,valida_fpr_once,valida_tpr_once=metrix(y_proba_valid_once,valida.iloc[:,0])## obtain the validation performance of the ensemble model with ensemble three features
        inter_tpr=np.interp(mean_fpr,valida_fpr_once,valida_tpr_once)
        tpr_all.append(inter_tpr)
        valida_per_all.append(valida_per_once)
        y_proba_test_all.append(y_proba_test_once)##

    test_pred_score=np.mean(y_proba_test_all,axis=0)#the predicted probability of the test data
    valid_per_mean,valid_per_std=np.mean(valida_per_all,axis=0),np.std(valida_per_all,axis=0) ##the average of 5-fold performance
    valid_mean_tpr,valid_std_tpr=np.mean(tpr_all,axis=0),np.std(tpr_all,axis=0) 
    return valid_per_mean,valid_per_std,valid_mean_tpr,valid_std_tpr,test_pred_score

train_data,test_data=data.get_data()
featurename=['PSSM_AC','RPSSM','SSA']
valid_per_mean,valid_per_std,valid_mean_tpr,valid_std_tpr,test_pred_score=get_result(train_data,test_data)

####Outcomes
##the performance of 5-fold cross validation
Validation_performance=['%.3f' % valid_per_mean[j]+chr(177)+'%.3f' % valid_per_std[j] for j in range(0,len(valid_per_mean))]
Validation_performance=pd.DataFrame(Validation_performance,columns=['PRE','SN','SP','F_score','ACC','MCC','AUC'])
##the performance of test data
test_performance,test_fpr,test_trp=metrix(test_pred_score,test_data[0].iloc[:,0])
test_performance=pd.DataFrame(test_performance,columns=['PRE','SN','SP','F_score','ACC','MCC','AUC'])

def plot_roc(mean_tpr,std_tpr,mean_auc,std_auc,featurename):
    plt.rcParams['font.family']=['Arial']
    fig=plt.figure(figsize=(16,10)) 
    mean_fpr=np.linspace(0,1,100)
    col=['cyan','orange','green','blue','red']
    plt.plot(mean_fpr, mean_tpr, color=col[0],lw=2, label='ROC curve of {} (AUC=%0.3f $\\pm$ %0.3f)'.format(featurename)%(mean_auc,std_auc))
    tprs_upper=np.minimum(mean_tpr+std_tpr,1)
    tprs_lower=np.maximum(mean_tpr-std_tpr,0)
    plt.fill_between(mean_fpr,tprs_lower,tprs_upper,alpha=0.3)
    plt.legend(loc="lower right",fontsize=14)   
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.set_xlim([-0.05, 1.05])
    plt.set_ylim([-0.05, 1.05])
    plt.set_yticks([0.0,0.2,0.4,0.6,0.8,1.0])
    plt.set_yticklabels(['0.0','0.2','0.4','0.6','0.8','1.0'],fontsize=14)
    plt.set_xticks([0.0,0.2,0.4,0.6,0.8,1.0])
    plt.set_xticklabels(['0.0','0.2','0.4','0.6','0.8','1.0'],fontsize=14)
    plt.set_ylabel('True Positive Rate',fontsize=15)
    plt.set_xlabel('False Positive Rate',fontsize=15)
    plt.show()


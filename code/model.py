import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score,recall_score,accuracy_score,f1_score,matthews_corrcoef,roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
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
import random
from tqdm import tqdm, trange
from itertools import combinations
import sys
sys.path.append(r'F:\\zhulin\\monsh\\paper-anti-CRISPR\\latex\\outcomes\\vot_outcome\\anti_CRISPR\\code')
from data import get_data

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
def performance(y_train,y_train_pred):
    #SN,SP,PRE,ACC,F-Score,MCC
    SN=recall_score(y_train,y_train_pred)
    PRE=precision_score(y_train,y_train_pred)
    ACC=accuracy_score(y_train, y_train_pred)
    F_score=f1_score(y_train,y_train_pred)
    MCC=matthews_corrcoef(y_train, y_train_pred)
    tn, fp, fn, tp = confusion_matrix(y_train, y_train_pred).ravel()
    SP= tn / (tn+fp)
    per=np.array([PRE,SN,SP,F_score,ACC,MCC])
    return per 

###choose various models
def func1(model_str,train_data):
    x_train,y_train=train_data.iloc[:,1:],train_data.iloc[:,0]
    max_par=int(max(x_train.columns.size/2,sqrt(x_train.columns.size)))
    model_name=[]
    model=[]
    if 'SVM' in model_str:
        ##
        par=[2**i for i in range(-6,7)]
        param_grid_sv=[{'kernel':['rbf'],'gamma':par,'C':par},{'kernel':['linear'],'C':par}]
        svm_clf=GridSearchCV(SVC(probability=True),param_grid_sv,cv=10,n_jobs=-1).fit(x_train,y_train)
        svm=svm_clf.best_estimator_.fit(x_train,y_train)
        model_name.append(svm)
        model.append('SVM')
    if 'KNN' in model_str:
        param_grid_knn={'n_neighbors':range(1,max_par)}
        knn_clf=GridSearchCV(KNeighborsClassifier(),param_grid_knn,cv=10,n_jobs=-1).fit(x_train,y_train)
        knn=knn_clf.best_estimator_.fit(x_train,y_train)
        model_name.append(knn)
        model.append('KNN')
    if 'RF' in model_str:
        param_grid_rf={'n_estimators':range(1,max_par),'max_features':range(1,20,5)}
        rf_clf=GridSearchCV(RandomForestClassifier(),param_grid_rf,cv=10,n_jobs=-1).fit(x_train,y_train)
        rf=rf_clf.best_estimator_.fit(x_train,y_train)
        model_name.append(rf)
        model.append('RF')
    if 'GNB' in model_str:
        gnb = GaussianNB().fit(x_train,y_train)
        model_name.append(gnb)
        model.append('GNB')
    if 'MLP' in model_str:
        mlp=MLPClassifier(hidden_layer_sizes=[64,32],max_iter=1000).fit(x_train,y_train)
        model_name.append(mlp)
        model.append('MLP')
    if 'LR' in model_str:
        lr=LogisticRegression().fit(x_train,y_train)
        model_name.append(lr)
        model.append('LR')
    if 'XGB' in model_str:
        XGB=XGBClassifier(learning_rate=0.1,eval_metric=['logloss','auc','error'],use_label_encoder=False,objective="binary:logistic").fit(x_train,y_train)
        model_name.append(XGB)
        model.append('XGB')
    if 'Light' in model_str:
        light=LGBMClassifier().fit(x_train,y_train)
        model_name.append(light)
        model.append('Light')
    if 'ensemble' in model_str:
       ensem = VotingClassifier(estimators=list(zip(model,model_name)),voting='soft',weights=[1]*(len(model_name))).fit(x_train,y_train)
       model_name.append(ensem)
       model.append('ensemble')    
    return model_name,model

def model_all(train_data,test_data,independent_data,model_str):
    x_valid,y_valid=test_data.iloc[:,1:],test_data.iloc[:,0]
    x_inde,y_inde=independent_data.iloc[:,1:],independent_data.iloc[:,0]
    y_proba_inde_all=np.empty(shape=(0,len(y_inde)))
    each_model=np.empty(shape=(0,len(y_valid)))
    model_name,model_str=func1(model_str,train_data)
    for clf in model_name:
        y_proba_valid=clf.predict_proba(x_valid)[:,1]
        each_model=np.r_[each_model,[np.array(y_proba_valid)]]
        y_inde_proba=clf.predict_proba(x_inde)[:,1]
        y_proba_inde_all=np.r_[y_proba_inde_all,[np.array(y_inde_proba)]]    
    return each_model,y_proba_inde_all

def mean_all(data):
    mean_value=[]
    std_value=[]
    for j in range(0,len(data[0])):
        mean_value.append(np.mean([data[i].iloc[j,:] for i in range(0,5)],axis=0))
        std_value.append(np.std([data[i].iloc[j,:] for i in range(0,5)],axis=0))
    return mean_value,std_value

def inde_pred(y_pred_inde):
    y_pred=[]
    for j in range(0,len(y_pred_inde[0])):
        y_pred.append(np.mean([y_pred_inde[i][j] for i in range(0,5)],axis=0))
    y_pred=np.array(y_pred)
    y=np.where(y_pred>0.5,1,0)
    return y

def ensemble_proba(indepen_all):
    ##feature
    a=[0,1,2,3,4]
    train_feature=np.empty(shape=(0,len(indepen_all[0][0])))
    for k in indepen_all:
        train_feature=np.concatenate((train_feature,k),axis=0)
    two_site=list(combinations(a,2))
    for i in two_site:
        proba_sum_two=(indepen_all[i[0]]+indepen_all[i[1]])/2
        train_feature=np.concatenate((train_feature,proba_sum_two),axis=0)
    for j in list(combinations(a,3)):
        proba_sum_three=(indepen_all[j[0]]+indepen_all[j[1]]+indepen_all[j[2]])/3
        train_feature=np.concatenate((train_feature,proba_sum_three),axis=0)
    for b in list(combinations(a,4)):
        proba_sum_four=(indepen_all[b[0]]+indepen_all[b[1]]+indepen_all[b[2]]+indepen_all[b[3]])/4
        train_feature=np.concatenate((train_feature,proba_sum_four),axis=0)
    aa=indepen_all[0]
    for ii in range(1,len(indepen_all)):
        aa=aa+indepen_all[ii]
    train_feature=np.concatenate((train_feature,aa/5),axis=0)
    return train_feature

def obtain_data(train_data,test_data,independent_data,model_str):
    valid_proba=[0]*len(train_data)
    inde_proba=[0]*len(train_data)
    for i in range(0,len(train_data)):
        train=mrmr_feature_selection(train_data[i])
        test=mrmr_feature_selection(test_data[i])
        independent=mrmr_feature_selection(independent_data[i])
        each_model,y_proba_inde_all=model_all(train,test,independent,model_str)
        valid_proba[i]=each_model
        inde_proba[i]=y_proba_inde_all
    valida=ensemble_proba(valid_proba)
    inde=ensemble_proba(inde_proba)
    return valida,inde

def metrix(proba_data,independent_y):
    auc1_inde=[]
    fprall=[]
    tprall=[]
    for i in range(0,len(proba_data)):
        fpr_inde,tpr_inde,threshols_inde=roc_curve(list(independent_y),list(proba_data[i]))
        fprall.append(fpr_inde)
        tprall.append(tpr_inde)
        auc1_inde.append(auc(fpr_inde,tpr_inde))
        
    performance_inde_all=[]
    pred3=np.where(np.array(proba_data)>=0.5,1,np.array(proba_data))
    pred_3=np.where(pred3<0.5,0,pred3)
    for i in range(0,len(pred_3)):
        performance_inde_all.append(performance(independent_y,pred_3[i]))
    metrix=pd.DataFrame(performance_inde_all,columns=['PRE','SN','SP','F_score','ACC','MCC'])
    metrix['AUC']=list(auc1_inde)
    return metrix,tprall,fprall

def one_feature(data,independent,model_str):   
    test_data=[0]*len(data)
    train_data=[0]*len(data) 
    tpr_fold=[]
    mean_fpr=np.linspace(0,1,100)
    kf=StratifiedKFold(n_splits=5)
    x,y=data[0].iloc[:,1:],data[0].iloc[:,0]
    valida_all=[]
    indep_all=[]
    for train, test in tqdm(kf.split(x,y)):
        for i in range(0,len(data)):
            train_data[i],test_data[i]=data[i].iloc[train,:],data[i].iloc[test,:]
        ###
        valida_y=test_data[0].iloc[:,0]
        Valida,Indepen=obtain_data(train_data,test_data,independent,model_str)
        ##
        valida_per,tprall,fprall=metrix(Valida,valida_y)
        inter_tpr=[np.interp(mean_fpr,fprall[i],tprall[i]) for i in range(0,len(tprall))]
        tpr_fold.append(inter_tpr)
        valida_all.append(valida_per)
        indep_all.append(Indepen)

    y_pred_score=(indep_all[0]+indep_all[1]+indep_all[2]+indep_all[3]+indep_all[4])/5
    valid_mean,valid_std=mean_all(valida_all) 
    mean_tpr,std_tpr=mean_roc(tpr_fold) 
    return valid_mean,valid_std,mean_tpr,std_tpr,y_pred_score

def trans(data,model_str,featurename):
    pd_test_mean=pd.DataFrame(data)
    pd_test_mean['featurename']=np.repeat(featurename,len(model_str))
    pd_test_mean['model']=model_str*len(featurename)
    return pd_test_mean

def save(data,name):
    outpath="/home/user/zhulin/outcome"
    data.to_csv(os.path.join(outpath,"{}.csv".format(name)))

def result_save(train_data,test_data,model_str,featurename,number):
    valid_mean,valid_std,mean_tpr,std_tpr,y_pred_score=one_feature(train_data,test_data,model_str)
    save(trans(valid_mean,model_str,featurename),'Train_mean_{}'.format(number))
    save(trans(valid_std,model_str,featurename),'Train_std_{}'.format(number))
    save(trans(mean_tpr,model_str,featurename),'Train_mean_tpr_{}'.format(number))
    save(trans(std_tpr,model_str,featurename),'rain_std_tpr_{}'.format(number))
    pd_y_pred=pd.DataFrame(y_pred_score)
    pd_y_pred['featurename']=np.repeat(featurename,len(model_str))
    pd_y_pred['model']=model_str*len(featurename)
    save(pd_y_pred,'Y_pred_proba{}'.format(number))

train_data,test_data=get_data()
featurename=['AAC','PAAC','PSSM_AC','RPSSM','SSA']
model_str=['SVM','KNN','RF','MLP','LR','XGB','Light','ensemble']
featurename_two=[i[0]+'&'+i[1] for i in list(combinations(featurename,2))]
featurename_three=[i[0]+'&'+i[1] +'&'+i[2] for i in list(combinations(featurename,3))]
featurename_four=[i[0]+'&'+i[1] +'&'+i[2]+'&'+i[3] for i in list(combinations(featurename,4))]
featurename_five=['five_feature']
vot_featurename=featurename+featurename_two+featurename_three+featurename_four+featurename_five


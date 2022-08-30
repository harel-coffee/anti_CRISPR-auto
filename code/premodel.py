import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,precision_recall_curve,auc
from sklearn.metrics import precision_score,recall_score,accuracy_score,f1_score,matthews_corrcoef,roc_curve,auc
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
import multiprocessing as mp
from sklearn.model_selection import GridSearchCV
from math import sqrt
from tqdm import tqdm, trange
from itertools import combinations
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier,GradientBoostingClassifier
from catboost import CatBoostClassifier

###doing feature selection and normalization with mrmr mrthod
def mrmr_feature_selection(data):
    x,y=data.iloc[:,1:],data.iloc[:,0]
    x=pd.DataFrame(preprocessing.minmax_scale(x),columns=x.columns)
    y=pd.Series(list(y))
    if x.columns.size>200:
        x=x[mrmr_classif(X=x,y=y, K = 200)]
    return x 

###performance assessment

def performance(y_pred,y_verified):
    fpr,tpr,threshols=roc_curve(list(y_verified),list(y_pred))
    
    pred_y=np.where(np.array(y_pred)>=0.5,1,np.array(y_pred))
    pred_y=np.where(pred_y<0.5,0,pred_y)
    #SN,SP,PRE,ACC,F-Score,MCC
    SN=recall_score(y_verified,pred_y)
    PRE=precision_score(y_verified,pred_y)
    ACC=accuracy_score(y_verified, pred_y)
    F_score=f1_score(y_verified,pred_y)
    MCC=matthews_corrcoef(y_verified, pred_y)
    tn, fp, fn, tp = confusion_matrix(y_verified, pred_y).ravel()
    SP= tn / (tn+fp)
    per=[PRE,SN,SP,F_score,ACC,MCC,auc(fpr,tpr)]
    return per

def metrix(proba_y,verified_y):
    mean_fpr=np.linspace(0,1,100)
    per,tprs=[],[]
    for y_pred,y_verified in zip(proba_y,verified_y):
        fpr,tpr,threshols=roc_curve(list(y_verified),list(y_pred))
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        per.append(performance(y_pred,y_verified))
            
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    mean_per=np.mean(per,axis=0)
    mean_per[-1]=mean_auc
    metrics=pd.DataFrame(per,columns=['PRE','SN','SP','F_score','ACC','MCC','AUC'])
    metrics.loc[5]=list(mean_per)
    metrics.loc[6]=list(np.std(per,axis=0))
    metrics.insert(0,'category',['fold1','fold2','fold3','fold4','fold5','mean','std'])
    return metrics

###build ensmeble model
def individual_model(x_train,y_train):
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
    #Catboost
    cat=CatBoostClassifier(verbose=1000).fit(x_train,y_train)
    model_name.append(cat)
    #Light
    light=LGBMClassifier().fit(x_train,y_train)
    model_name.append(light) 
    return model_str,model_name

def ensemble_model1(x_train,y_train):
#    x_train,y_train=train_data.iloc[:,1:],train_data.iloc[:,0]
    model_str,model_name=individual_model(x_train,y_train)
    #ensemble1:hard voting 
    #ensem1 = VotingClassifier(estimators=list(zip(model_str,model_name)),voting='soft',weights=[1]*(len(model_name))).fit(x_train,y_train)
    ensem2 = StackingClassifier(estimators=list(zip(model_str,model_name)),final_estimator=LogisticRegression()).fit(x_train,y_train)
    #ensem3 = StackingClassifier(estimators=list(zip(model_str,model_name)),final_estimator=GradientBoostingClassifier()).fit(x_train,y_train)
    return ensem2    

####the ROC curve of 5-fold cross validation
def ROC_5_fold(y_proba_valid,y_validation):
    plt.figure(figsize=(10,8))
    i = 0
    mean_fpr=np.linspace(0,1,100)
    tprs,aucs=[],[]
    for y_proba, y_verified in zip(y_proba_valid, y_validation):
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(list(y_verified), list(y_proba))
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0]= 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                label='ROC fold %d (AUC = %0.3f)' % (i, roc_auc))

        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
            label='Mean ROC (AUC = %0.3f $\\pm$ %0.3f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3,
                    label='$\\pm$ 1 std. dev.')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0],labels=['0.0','0.2','0.4','0.6','0.8','1.0'])
    plt.xticks([0.0,0.2,0.4,0.6,0.8,1.0],labels=['0.0','0.2','0.4','0.6','0.8','1.0'])
    plt.ylabel('True Positive Rate',fontsize=16)
    plt.xlabel('False Positive Rate',fontsize=16)
    plt.legend(loc="lower right",fontsize=15)
    plt.show()

def auc_pred(test_pred_score,test_verified):
    fpr,tpr,threshold = roc_curve(test_verified, test_pred_score) ###
    roc_auc = auc(fpr,tpr) ###
    lw = 2
    plt.figure(figsize=(10,10))
    plt.plot(fpr, tpr, color='b',
            lw=lw, label='ROC curve (area = %0.3f)' % roc_auc) ###
    plt.plot([0, 1], [0, 1], color='r', lw=lw, linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0],labels=['0.0','0.2','0.4','0.6','0.8','1.0'])
    plt.xticks([0.0,0.2,0.4,0.6,0.8,1.0],labels=['0.0','0.2','0.4','0.6','0.8','1.0'])
    plt.ylabel('True Positive Rate',fontsize=16)
    plt.xlabel('False Positive Rate',fontsize=16)
    plt.legend(loc="lower right",fontsize=15)
    plt.show()

def pr_curve(test_pred_score,test_verified):
    precision,recall,thresholds=precision_recall_curve(test_verified,test_pred_score)
    plt.figure(figsize=(10,10))
    plt.plot(recall,precision)
    plt.rc('legend',fontsize=16)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall',fontsize=16)
    plt.ylabel('Precision',fontsize=16)
    plt.title('Precision/Recall Curve of {}'.format('PreAcrs'),fontsize=18)
    plt.show()
####
def get_result(train_data,test_data):   
    y_verified_valid_all,y_proba_valid_all,y_proba_test_all=[],[],[]
    mean_fpr=np.linspace(0,1,100)
     
    kf=StratifiedKFold(n_splits=5) ###5-fold-cross-validation
    x,y=train_data[0].iloc[:,1:],train_data[0].iloc[:,0]  
    for train_site, valida_site in kf.split(x,y):
        y_proba_test_ense,y_proba_valid_ense=[],[]
        
        for i in range(0,len(train_data)):
            Y_train,Y_valid=train_data[i].iloc[train_site,:].iloc[:,0],train_data[i].iloc[valida_site,:].iloc[:,0]
            X_train=mrmr_feature_selection(train_data[i].iloc[train_site,:])
            X_valid=pd.DataFrame(preprocessing.minmax_scale(train_data[i].iloc[valida_site,:][X_train.columns]),columns=X_train.columns)
            X_test=pd.DataFrame(preprocessing.minmax_scale(test_data[i][X_train.columns]),columns=X_train.columns)

            ensemble_clf=ensemble_model1(X_train,Y_train)
            y_proba_valid=ensemble_clf.predict_proba(X_valid)[:,1]
            y_proba_test=ensemble_clf.predict_proba(X_test)[:,1]
            y_proba_valid_ense.append(y_proba_valid)
            y_proba_test_ense.append(y_proba_test)  
             
        y_proba_valid_all.append(np.mean(y_proba_valid_ense,axis=0)) ##ensemble three features 
        y_proba_test_all.append(np.mean(y_proba_test_ense,axis=0))
        y_verified_valid_all.append(Y_valid)
        ##
    test_pred_score=np.mean(y_proba_test_all,axis=0)#the predicted probability of the test data
#    valid_per_mean,valid_per_std=np.mean(valida_per_all,axis=0),np.std(valida_per_all,axis=0) ##the average of 5-fold performance
#    valid_mean_tpr,valid_std_tpr=np.mean(tpr_all,axis=0),np.std(tpr_all,axis=0) 
    return y_proba_valid_all,y_verified_valid_all,test_pred_score

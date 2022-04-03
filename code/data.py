import pandas as pd
import numpy as np
import random 
def get_data():
    ###read data from: /anti_CRISPR/data
    ###
    ##
    AAC_positive=pd.read_csv("F:\\zhulin\monsh\\paper-anti-CRISPR\\latex\outcomes\\vot_outcome\\anti_CRISPR\\data\\AAC_Positive.csv").iloc[:,1:]
    CKSAAP_positive=pd.read_csv("F:\\zhulin\monsh\\paper-anti-CRISPR\\latex\\outcomes\\vot_outcome\\anti_CRISPR\\data\\CKSAAP_Positive.csv").iloc[:,1:]
    DDE_positive=pd.read_csv("F:\\zhulin\monsh\\paper-anti-CRISPR\\latex\\outcomes\\vot_outcome\\anti_CRISPR\\data\\DDE_Positive.csv").iloc[:,1:]
    DPC_positive=pd.read_csv("F:\\zhulin\monsh\\paper-anti-CRISPR\\latex\\outcomes\\vot_outcome\\anti_CRISPR\\data\\DPC_Positive.csv").iloc[:,1:]
    PAAC_positive=pd.read_csv("F:\\zhulin\monsh\\paper-anti-CRISPR\\latex\\outcomes\\vot_outcome\\anti_CRISPR\\data\\PAAC_Positive.csv").iloc[:,1:]
    
    DPC_PSSM_positive=pd.read_csv("F:\\zhulin\monsh\\paper-anti-CRISPR\\latex\\outcomes\\vot_outcome\\anti_CRISPR\\data\\DPC_PSSM_Positive.csv").iloc[:,1:]
    PSSM_AC_positive=pd.read_csv("F:\\zhulin\monsh\\paper-anti-CRISPR\\latex\\outcomes\\vot_outcome\\anti_CRISPR\\data\\PSSM_AC_Positive.csv").iloc[:,1:]
    PSSM_COM_positive=pd.read_csv("F:\\zhulin\monsh\\paper-anti-CRISPR\\latex\\outcomes\\vot_outcome\\anti_CRISPR\\data\\PSSM_COM_Positive.csv").iloc[:,1:]
    RPSSM_positive=pd.read_csv("F:\\zhulin\monsh\\paper-anti-CRISPR\\latex\\outcomes\\vot_outcome\\anti_CRISPR\\data\\RPSSM_Positive.csv").iloc[:,1:]
    SMO_PSSM_positive=pd.read_csv("F:\\zhulin\monsh\\paper-anti-CRISPR\\latex\\outcomes\\vot_outcome\\anti_CRISPR\\data\\SMO_PSSM_Positive.csv").iloc[:,1:]
  
    BiLSTM_positive=pd.read_csv("F:\\zhulin\monsh\\paper-anti-CRISPR\\latex\\outcomes\\vot_outcome\\anti_CRISPR\\data\\BiLSTM_Positive.csv").iloc[:,1:]
    LM_positive=pd.read_csv("F:\\zhulin\monsh\\paper-anti-CRISPR\\latex\\outcomes\\vot_outcome\\anti_CRISPR\\data\\LM_Positive.csv").iloc[:,1:]
    SSA_positive=pd.read_csv("F:\\zhulin\monsh\\paper-anti-CRISPR\\latex\\outcomes\\vot_outcome\\anti_CRISPR\\data\\SSA_Positive.csv").iloc[:,1:]
    BERT_positive=pd.read_csv("F:\\zhulin\monsh\\paper-anti-CRISPR\\latex\\outcomes\\vot_outcome\\anti_CRISPR\\data\\BERT_Positive.csv").iloc[:,1:]
    UniRep_positive=pd.read_csv("F:\\zhulin\monsh\\paper-anti-CRISPR\\latex\\outcomes\\vot_outcome\\anti_CRISPR\\data\\UniRep_Positive.csv").iloc[:,1:]
    W2V_positive=pd.read_csv("F:\\zhulin\monsh\\paper-anti-CRISPR\\latex\\outcomes\\vot_outcome\\anti_CRISPR\\data\\W2V_Positive.csv").iloc[:,1:]

    esm_positive=pd.read_csv("F:\\zhulin\monsh\\paper-anti-CRISPR\\latex\\outcomes\\vot_outcome\\anti_CRISPR\\data\\esm_Positive.csv").iloc[:,1:]
    prott_positive=pd.read_csv("F:\\zhulin\monsh\\paper-anti-CRISPR\\latex\\outcomes\\vot_outcome\\anti_CRISPR\\data\\prott_Positive.csv").iloc[:,1:]

    ###########
    AAC_negative=pd.read_csv("F:\\zhulin\monsh\\paper-anti-CRISPR\\latex\outcomes\\vot_outcome\\anti_CRISPR\\data\\AAC_negative.csv").iloc[:,1:]
    CKSAAP_negative=pd.read_csv("F:\\zhulin\monsh\\paper-anti-CRISPR\\latex\outcomes\\vot_outcome\\anti_CRISPR\\data\\CKSAAP_negative.csv").iloc[:,1:]
    DDE_negative=pd.read_csv("F:\\zhulin\monsh\\paper-anti-CRISPR\\latex\outcomes\\vot_outcome\\anti_CRISPR\\data\\DDE_negative.csv").iloc[:,1:]
    DPC_negative=pd.read_csv("F:\\zhulin\monsh\\paper-anti-CRISPR\\latex\outcomes\\vot_outcome\\anti_CRISPR\\data\\DPC_negative.csv").iloc[:,1:]
    PAAC_negative=pd.read_csv("F:\\zhulin\monsh\\paper-anti-CRISPR\\latex\outcomes\\vot_outcome\\anti_CRISPR\\data\\PAAC_negative.csv").iloc[:,1:]

    DPC_PSSM_negative=pd.read_csv("F:\\zhulin\monsh\\paper-anti-CRISPR\\latex\outcomes\\vot_outcome\\anti_CRISPR\\data\\DPC_PSSM_negative.csv").iloc[:,1:]
    PSSM_AC_negative=pd.read_csv("F:\\zhulin\monsh\\paper-anti-CRISPR\\latex\outcomes\\vot_outcome\\anti_CRISPR\\data\\PSSM_AC_negative.csv").iloc[:,1:]
    PSSM_COM_negative=pd.read_csv("F:\\zhulin\monsh\\paper-anti-CRISPR\\latex\outcomes\\vot_outcome\\anti_CRISPR\\data\\PSSM_COM_negative.csv").iloc[:,1:]
    RPSSM_negative=pd.read_csv("F:\\zhulin\monsh\\paper-anti-CRISPR\\latex\outcomes\\vot_outcome\\anti_CRISPR\\data\\RPSSM_negative.csv").iloc[:,1:]
    SMO_PSSM_negative=pd.read_csv("F:\\zhulin\monsh\\paper-anti-CRISPR\\latex\outcomes\\vot_outcome\\anti_CRISPR\\data\\SMO_PSSM_negative.csv").iloc[:,1:]

    BiLSTM_negative=pd.read_csv("F:\\zhulin\monsh\\paper-anti-CRISPR\\latex\outcomes\\vot_outcome\\anti_CRISPR\\data\\BiLSTM_negative.csv").iloc[:,1:]
    LM_negative=pd.read_csv("F:\\zhulin\monsh\\paper-anti-CRISPR\\latex\outcomes\\vot_outcome\\anti_CRISPR\\data\\LM_negative.csv").iloc[:,1:]
    SSA_negative=pd.read_csv("F:\\zhulin\monsh\\paper-anti-CRISPR\\latex\outcomes\\vot_outcome\\anti_CRISPR\\data\\SSA_negative.csv").iloc[:,1:]
    BERT_negative=pd.read_csv("F:\\zhulin\monsh\\paper-anti-CRISPR\\latex\outcomes\\vot_outcome\\anti_CRISPR\\data\\BERT_negative.csv").iloc[:,1:]
    UniRep_negative=pd.read_csv("F:\\zhulin\monsh\\paper-anti-CRISPR\\latex\outcomes\\vot_outcome\\anti_CRISPR\\data\\UniRep_negative.csv").iloc[:,1:]
    W2V_negative=pd.read_csv("F:\\zhulin\monsh\\paper-anti-CRISPR\\latex\outcomes\\vot_outcome\\anti_CRISPR\\data\\W2V_negative.csv").iloc[:,1:]

    esm_negative=pd.read_csv("F:\\zhulin\monsh\\paper-anti-CRISPR\\latex\outcomes\\vot_outcome\\anti_CRISPR\\data\\esm_negative.csv").iloc[:,1:]
    prott_negative=pd.read_csv("F:\\zhulin\monsh\\paper-anti-CRISPR\\latex\outcomes\\vot_outcome\\anti_CRISPR\\data\\prott_negative.csv").iloc[:,1:]

    random.seed(1)
    train_site_positive=random.sample(list(range(0,588)),412)
    test_site_positive=list(range(0,588))
    for i in train_site_positive:
        test_site_positive.remove(i)
    random.seed(1)
    train_site_negative=random.sample(list(range(0,1571)),412)
    test_site_negative1=list(range(0,1571))
    for i in train_site_negative:
        test_site_negative1.remove(i)
    random.seed(1)
    test_site_negative=random.sample(test_site_negative1,176)

    AAC_data_train=pd.concat([AAC_positive.iloc[train_site_positive,:],AAC_negative.iloc[train_site_negative,:]])
    CKSAAP_data_train=pd.concat([CKSAAP_positive.iloc[train_site_positive,:],CKSAAP_negative.iloc[train_site_negative,:]])
    DDE_data_train=pd.concat([DDE_positive.iloc[train_site_positive,:],DDE_negative.iloc[train_site_negative,:]])
    DPC_data_train=pd.concat([DPC_positive.iloc[train_site_positive,:],DPC_negative.iloc[train_site_negative,:]])
    PAAC_data_train=pd.concat([PAAC_positive.iloc[train_site_positive,:],PAAC_negative.iloc[train_site_negative,:]])
    
    DPC_PSSM_data_train=pd.concat([DPC_PSSM_positive.iloc[train_site_positive,:],DPC_PSSM_negative.iloc[train_site_negative,:]])
    PSSM_AC_data_train=pd.concat([PSSM_AC_positive.iloc[train_site_positive,:],PSSM_AC_negative.iloc[train_site_negative,:]])
    PSSM_COM_data_train=pd.concat([PSSM_COM_positive.iloc[train_site_positive,:],PSSM_COM_negative.iloc[train_site_negative,:]])
    RPSSM_data_train=pd.concat([RPSSM_positive.iloc[train_site_positive,:],RPSSM_negative.iloc[train_site_negative,:]])
    SMO_PSSM_data_train=pd.concat([SMO_PSSM_positive.iloc[train_site_positive,:],SMO_PSSM_negative.iloc[train_site_negative,:]])
    
    BiLSTM_PSSM_data_train=pd.concat([BiLSTM_positive.iloc[train_site_positive,:],BiLSTM_negative.iloc[train_site_negative,:]])
    LM_data_train=pd.concat([LM_positive.iloc[train_site_positive,:],LM_negative.iloc[train_site_negative,:]])
    SSA_data_train=pd.concat([SSA_positive.iloc[train_site_positive,:],SSA_negative.iloc[train_site_negative,:]])
    BERT_data_train=pd.concat([BERT_positive.iloc[train_site_positive,:],BERT_negative.iloc[train_site_negative,:]])
    UniRep_data_train=pd.concat([UniRep_positive.iloc[train_site_positive,:],UniRep_negative.iloc[train_site_negative,:]])
    W2V_data_train=pd.concat([W2V_positive.iloc[train_site_positive,:],W2V_negative.iloc[train_site_negative,:]])
    
    esm_data_train=pd.concat([esm_positive.iloc[train_site_positive,:],esm_negative.iloc[train_site_negative,:]])
    prott_data_train=pd.concat([prott_positive.iloc[train_site_positive,:],prott_negative.iloc[train_site_negative,:]])
    
    data_train_all=[AAC_data_train,CKSAAP_data_train,DDE_data_train,DPC_data_train,PAAC_data_train,DPC_PSSM_data_train,PSSM_AC_data_train,
        PSSM_COM_data_train,RPSSM_data_train,SMO_PSSM_data_train,BiLSTM_PSSM_data_train,LM_data_train,SSA_data_train,BERT_data_train,
        UniRep_data_train,W2V_data_train,esm_data_train,prott_data_train]

    AAC_data_test=pd.concat([AAC_positive.iloc[test_site_positive,:],AAC_negative.iloc[test_site_negative,:]])
    CKSAAP_data_test=pd.concat([CKSAAP_positive.iloc[test_site_positive,:],CKSAAP_negative.iloc[test_site_negative,:]])
    DDE_data_test=pd.concat([DDE_positive.iloc[test_site_positive,:],DDE_negative.iloc[test_site_negative,:]])
    DPC_data_test=pd.concat([DPC_positive.iloc[test_site_positive,:],DPC_negative.iloc[test_site_negative,:]])
    PAAC_data_test=pd.concat([PAAC_positive.iloc[test_site_positive,:],PAAC_negative.iloc[test_site_negative,:]])
    
    DPC_PSSM_data_test=pd.concat([DPC_PSSM_positive.iloc[test_site_positive,:],DPC_PSSM_negative.iloc[test_site_negative,:]])
    PSSM_AC_data_test=pd.concat([PSSM_AC_positive.iloc[test_site_positive,:],PSSM_AC_negative.iloc[test_site_negative,:]])
    PSSM_COM_data_test=pd.concat([PSSM_COM_positive.iloc[test_site_positive,:],PSSM_COM_negative.iloc[test_site_negative,:]])
    RPSSM_data_test=pd.concat([RPSSM_positive.iloc[test_site_positive,:],RPSSM_negative.iloc[test_site_negative,:]])
    SMO_PSSM_data_test=pd.concat([SMO_PSSM_positive.iloc[test_site_positive,:],SMO_PSSM_negative.iloc[test_site_negative,:]])
    
    BiLSTM_PSSM_data_test=pd.concat([BiLSTM_positive.iloc[test_site_positive,:],BiLSTM_negative.iloc[test_site_negative,:]])
    LM_data_test=pd.concat([LM_positive.iloc[test_site_positive,:],LM_negative.iloc[test_site_negative,:]])
    SSA_data_test=pd.concat([SSA_positive.iloc[test_site_positive,:],SSA_negative.iloc[test_site_negative,:]])
    BERT_data_test=pd.concat([BERT_positive.iloc[test_site_positive,:],BERT_negative.iloc[test_site_negative,:]])
    UniRep_data_test=pd.concat([UniRep_positive.iloc[test_site_positive,:],UniRep_negative.iloc[test_site_negative,:]])
    W2V_data_test=pd.concat([W2V_positive.iloc[test_site_positive,:],W2V_negative.iloc[test_site_negative,:]])
    
    esm_data_test=pd.concat([esm_positive.iloc[test_site_positive,:],esm_negative.iloc[test_site_negative,:]])
    prott_data_test=pd.concat([prott_positive.iloc[test_site_positive,:],prott_negative.iloc[test_site_negative,:]])
    
    data_test_all=[AAC_data_test,CKSAAP_data_test,DDE_data_test,DPC_data_test,PAAC_data_test,DPC_PSSM_data_test,PSSM_AC_data_test,
        PSSM_COM_data_test,RPSSM_data_test,SMO_PSSM_data_test,BiLSTM_PSSM_data_test,LM_data_test,SSA_data_test,BERT_data_test,
        UniRep_data_test,W2V_data_test,esm_data_test,prott_data_test]
    data_train=[AAC_data_train,PAAC_data_train,PSSM_AC_data_train,RPSSM_data_train,SSA_data_train]
    data_test=[AAC_data_test,PAAC_data_test,PSSM_AC_data_test,RPSSM_data_test,SSA_data_test]
    return data_train,data_test,data_train_all,data_test_all
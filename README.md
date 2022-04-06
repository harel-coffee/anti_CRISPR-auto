# PreAcrs

PreAcrs: one powerful model to predict anti-CRISPR proteins

## How to use:

### Data preparetion:

After downloading the amino acid sequences of anti-CRISPR proteins and un-anti-CRISPR proteins from NCBI and UniProt, you should extract three features from the below steps before using PreAcrs:you should extract three features from belowing steps before you use PreAcrs.

1. extracting the PSSM_AC feature and RPSSM feature from an online service named POSSUM [https://possum.erc.monash.edu/server.jsp]()
2. extracting the SSA feature from a python-based toolkit named eFeature [http://lab.malab.cn/soft/eFeature/index.html](http://lab.malab.cn/soft/eFeature/index.html)

### Architecture

The architecture of the PreAcrs is displayed in the following picture.

![](image/README/1649252127497.png)

### output

After preparing three features of the training dataset and the testing dataset, Download the code folder.

Read the three features .csv files of the training and testing datasets in the data.py, return train_data and test_data, respectively.


Run ‘PreAcrs.ipynb’

the result includes:

performance evaluation dataframe ‘validation_perfromance’ of the training dataset based on the 5-fold cross-validation

The ROC image of the training dataset based on  the  5-fold cross-valion

The predicted scores of the testing proteins are saved in the ‘pred_test’ dataframe.

The performance evaluation form and ROC image of the testing data are showed.

#!/usr/bin/env python
# coding: utf-8
'''
The code was a demo for brain age prediction using relevance vector regression

Author: Allard.W.Shi
github: Allard-Shi (allard-shi.github.io)
'''

import xlrd   # read excel
import time   # tick toc
import numpy as np 
from skrvm import RVR
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA 
from sklearn.model_selection import cross_val_score  # cross-validation
from sklearn.model_selection import StratifiedKFold,KFold

# import data from xlsx file
def xlsxfilein(filepath,sheet):
    data = []
    wb=xlrd.open_workbook(file_path)
    ws=wb.sheet_by_name(sheet)
    for r in range(ws.nrows):
        col=[]
        for c in range(ws.ncols):
            col.append(ws.cell(r,c).value)
        data.append(col)
    return data

def coff_p(X):
    N = X.shape[1] - 1
    P = np.zeros((N,1))
    C = np.zeros((N,1))
    for i in range(N):
        C[i],P[i]=stats.pearsonr(X[:,0].T,X[:,i+1].T)
    return C,P

def choose_fiber(data, data_test, coff, coff_p, cor_threshold = 0.15, p = 0.001):
    select = (np.abs(coff) > cor_threshold) & (coff_p < p)
    select = select.reshape(select.shape[0],)
    result = data[:,1:].T
    result_t = data_test[:,1:].T
    result = result[select]
    result_t = result_t[select]
    result = result.T
    result_t = result_t.T
    print(result.shape)
    age = data[:,0]
    age_t = data_test[:,0]
    return age,result,select,age_t,result_t

def rvr_pipeline(x,y,pca,kernel,x_p=0,y_p=0,fold=10,seed=2019,predict_data=False):
    
    rvr = RVR(kernel=kernel)
    kf = KFold(n_splits=fold, shuffle=True, random_state=seed)
    score = np.zeros((fold,))
    i = 0
    for train,test in kf.split(x,y):
        t1 = time.time()
        x_train, y_train = x[train], y[train]
        x_test, y_test = x[test], y[test]
        pca.fit(x_train)
        new_train = pca.transform(x_train)
        new_test = pca.transform(x_test)
        scaler.fit(new_train)
        new_train = scaler.transform(new_train)
        new_test = scaler.transform(new_test)
        rvr.fit(new_train,y_train)
        pred = rvr.predict(new_test)
        mse = abs(pred-y_test)
        score[i] = sum(mse)/mse.shape[0]
        i+=1
        t2 = time.time()
        print('fold '+str(i)+':',t2-t1,'sec')
    print('='*40)
    print('MAE:',np.mean(score))
    
    if predict_data:
        pca.fit(x)
        new_train = pca.transform(x)
        new_test = pca.transform(x_p)
        scaler.fit(new_train)
        new_train = scaler.transform(new_train)
        new_test = scaler.transform(new_test)
        rvr.fit(new_train,y)
        pred = rvr.predict(new_test)
        error = abs(pred-y_p)
        print('Test MAE:',sum(error)/error.shape[0])
    
    return pred

if __name__ == '__main__':
    # training set
    file_path = r'C:\Users\Shi Wen\Desktop\DTI_fiber\mean_rawindex.xlsx'
    sheet = 'mean1'
    data = xlsxfilein(file_path,sheet)
    data = np.array(data)
    print(data.shape,data.dtype) #identification
    N1 = data.shape[0]
    N2 = 50  ##the number of fiber
    rawdata = data[:,0:N2+1]
    rawdata = rawdata.astype(float)
    print('Number of training data:',N1)
    print('Shape of data:',rawdata.shape,rawdata.dtype)
    Age, FA = rawdata[:,0], rawdata[:,1:]
    print(Age.shape,FA.shape)

    # test set
    print('='*20+' TEST '+'='*20)
    file_path = r'C:\Users\Shi Wen\Desktop\DTI_fiber\test_data_all.xlsx'
    sheet = 'mean_com'
    data = xlsxfilein(file_path,sheet)
    data = np.array(data)
    print(data.shape,data.dtype) #identification
    N1 = data.shape[0]
    N2 = 50  ##the number of fiber
    rawdata_test = data[:,0:N2+1]
    rawdata_test = rawdata_test.astype(float)
    print('Number of test data:',N1)
    print('Shape of data:',rawdata_test.shape,rawdata_test.dtype)
    Age_t, FA_t = rawdata_test[:,0], rawdata_test[:,1:]
    print(Age_t.shape,FA_t.shape)

    # Hyperparameter
    Sec = [32,26,48,42]
    Sec_t = [29,45]
    p = 0.001
    y_cor_thres = 0.2
    m_cor_thres = 0.2
    e_cor_thres = 0.2

    young = Age<Sec[0]
    middle = (Age>Sec[1])&(Age<Sec[2])
    elderly = Age>Sec[3]

    young_t = Age_t<Sec_t[0]
    middle_t = (Age_t>Sec_t[0])&(Age_t<Sec_t[1])
    elderly_t = Age_t>Sec_t[1]

    Young = rawdata[young]
    Middle = rawdata[middle]
    Elderly = rawdata[elderly]
    Y_t = rawdata_test[young_t]
    M_t = rawdata_test[middle_t]
    E_t = rawdata_test[elderly_t]

    # fiber selection
    y_coff,y_p = coff_p(Young)
    m_coff,m_p = coff_p(Middle)
    e_coff,e_p = coff_p(Elderly)

    Y_age, Y_data, Y_slc, Y_t_age, Y_test = choose_fiber(Young,Y_t,y_coff,y_p)
    M_age, M_data, M_slc, M_t_age, M_test = choose_fiber(Middle,M_t,m_coff,m_p)
    E_age, E_data, E_slc, E_t_age, E_test = choose_fiber(Elderly,E_t,e_coff,e_p)

    kernel = 'linear'
    scaler = preprocessing.StandardScaler()
    pca = PCA(n_components=0.9)

    # Predict brain age
    print('='*20,'Training','='*20)
    Y_pred = rvr_pipeline(Y_data,Y_age,pca,kernel,x_p=Y_test,y_p=Y_t_age,predict_data=True)
    M_pred = rvr_pipeline(M_data,M_age,pca,kernel,x_p=M_test,y_p=M_t_age,predict_data=True)
    E_pred = rvr_pipeline(E_data,E_age,pca,kernel,x_p=E_test,y_p=E_t_age,predict_data=True)

    test_pred = np.hstack((Y_pred,M_pred))
    test_pred = np.hstack((test_pred,E_pred))
    new_age_t = np.hstack((Y_t_age,M_t_age))
    new_age_t = np.hstack((new_age_t,E_t_age))

    print('-'*20,'RESULT','-'*20)
    pad = new_age_t - test_pred
    print('Mean PAD:',np.mean(pad))
    mae = abs(pad)
    print('MAD:',np.mean(mae))






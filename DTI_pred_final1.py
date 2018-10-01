
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from pprint import pprint
import xlrd   ##read excel
from sklearn.model_selection import cross_val_score  # 交叉检验
from sklearn.decomposition import PCA  
from sklearn.svm import SVR,SVC
from sklearn.model_selection import GridSearchCV,learning_curve   
import time
from skrvm import RVR
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
from sklearn.feature_selection import RFE, RFECV
from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
from  sklearn import model_selection
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV


# In[3]:


def xlsxfilein(filepath,sheet,data):
    wb=xlrd.open_workbook(file_path)
    ws=wb.sheet_by_name(sheet)
    for r in range(ws.nrows):
        col=[]
        for c in range(ws.ncols):
            col.append(ws.cell(r,c).value)
        data.append(col)
    return data

def labelxy(xl,yl,labelsize=12):
    plt.xlabel(xl)
    plt.ylabel(yl)
#split-age
def SplitAge(Age,rawdata,N2):    ## N2 is the whole feature
    N1=rawdata.shape[0]
    k1,k2,k3=0,0,0
    Seg1=np.zeros((N1,N2+1))
    Seg2=np.zeros((N1,N2+1))
    Seg3=np.zeros((N1,N2+1))
    for i in range(N1):
        if rawdata[i,0]<Age[0]:
            Seg1[k1,:]=rawdata[i,:]
            k1=k1+1
    for i in range(N1):
        if (rawdata[i,0]>=Age[1])and(rawdata[i,0]<Age[2]):
            Seg2[k2,:]=rawdata[i,:]
            k2=k2+1
    for i in range(N1):
        if rawdata[i,0]>=Age[3]:
            Seg3[k3,:]=rawdata[i,:]
            k3=k3+1
    Young=Seg1[0:k1,:]
    Middle=Seg2[0:k2,:]
    Elderly=Seg3[0:k3,:]
    return Young,Middle,Elderly
#-----------------------------------------------------------Append three arrays together
def ArrayAppend(X,Y,Z):
    N1,N4=X.shape[0],X.shape[1]
    N2=Y.shape[0]
    N3=Z.shape[0]
    N=N1+N2+N3
    out=np.zeros((N,N4))
    for i in range(N):
        if i<N1:
            out[i]=X[i]
        elif i<(N1+N2):
            out[i]=Y[i-N1]
        else:
            out[i]=Z[i-N1-N2]
    return out
#-------------------------SVR model
def DTISVR3(pca,age,test,test_age,input_kernel,C1):
    svr=SVR(kernel=input_kernel,C=C1)
    N=age.shape[0]
    num_train=test.shape[0]
    svrerror,testerror=np.zeros((N,1)),np.zeros((test.shape[0],1))
    svr_age,pred_age=np.zeros((N,1)),np.zeros((test.shape[0],1))     ####svm_training data age and prediction age 
    t0=time.time()
    svr.fit(pca,age)
    t=time.time()-t0
    print('Traning time:',t,'s')
    s=svr.score(pca,age)
    for i in range(N):
        svr_age[i]=svr.predict(pca[i:i+1,:])
        svrerror[i]=(svr_age[i]-age[i])     # train data - real data
        #print('Real age:',age[i],'SVR age:',svr_age[i],'Difference:',svr_age[i]-age[i])
    for j in range(num_train):
        pred_age[j]=svr.predict(test[j:j+1,:])
        testerror[j]=(pred_age[j]-test_age[j])  
        #print('Test Real age:',test_age[j],'Test SVR age:',pred_age[j],'Difference:',pred_age[j]-test_age[j])
    print('Scores:',s)
    print('Training error:',svrerror[0])
    print('Test error:',testerror[0])
    print('Finished Prediction')
    print('-----------------------------------------------------------------------')
    return s,svrerror,testerror,svr_age,pred_age
 

#-------------------------RVR model
def DTIRVR3(inp,output,test,test_age,kerl):
    rvr=RVR(kernel=kerl)
    N=output.shape[0]
    rvrerror,testerror=np.zeros((N,1)),np.zeros((test.shape[0],1))
    rvr_age,pred_age=np.zeros((N,1)),np.zeros((test.shape[0],1))
    t0=time.time()
    rvr.fit(inp,output)
    t=time.time()-t0
    print('Training time:',t)
    s=rvr.score(inp,output)
    for i in range(N):
        rvr_age[i]=rvr.predict(inp[i:i+1,:])
        rvrerror[i]=(rvr_age[i]-output[i])
        #print('Real age:',output[i],'RVR age:',rvr_age[i],'Difference:',rvr_age[i]-output[i])
    for j in range(test.shape[0]):
        pred_age[j]=rvr.predict(test[j:j+1,:])
        testerror[j]=(pred_age[j]-test_age[j])
        #print('Test Real age:',test_age[j],'Test RVR age:',pred_age[j],'Difference:',pred_age[j]-test_age[j])
    print('Training error:',rvrerror[0])
    print('Test error:',testerror[0])
    print('Score:',s)
    return s,rvrerror,testerror,rvr_age,pred_age
#--------------------------------------------------------------
# Age is a array of fraction intervining the saction of the age. 
def SplitMore(Age,rawdata):
    N1=Age.shape[0] ### number is 9
    N2,N3=rawdata.shape[0],rawdata.shape[1]
    k1,k2,k3,k4,k5,k6,k7,k8,k9,k10=0,0,0,0,0,0,0,0,0,0
    Seg1=np.zeros((N2,N3))
    Seg2=np.zeros((N2,N3))
    Seg3=np.zeros((N2,N3))
    Seg4=np.zeros((N2,N3))
    Seg5=np.zeros((N2,N3))
    Seg6=np.zeros((N2,N3))
    Seg7=np.zeros((N2,N3))
    Seg8=np.zeros((N2,N3))
    Seg9=np.zeros((N2,N3))
    Seg10=np.zeros((N2,N3))
    for i in range(N2):
        if rawdata[i,0]<Age[0]:
            Seg1[k1,:]=rawdata[i,:]
            k1=k1+1
        elif rawdata[i,0]<Age[1]:
            Seg2[k2,:]=rawdata[i,:]
            k2=k2+1
        elif rawdata[i,0]<Age[2]:
            Seg3[k3,:]=rawdata[i,:]
            k3=k3+1
        elif rawdata[i,0]<Age[3]:
            Seg4[k4,:]=rawdata[i,:]
            k4=k4+1
        elif rawdata[i,0]<Age[4]:
            Seg5[k5,:]=rawdata[i,:]
            k5=k5+1
        elif rawdata[i,0]<Age[5]:
            Seg6[k6,:]=rawdata[i,:]
            k6=k6+1
        elif rawdata[i,0]<Age[6]:
            Seg7[k7,:]=rawdata[i,:]
            k7=k7+1
        elif rawdata[i,0]<Age[7]:
            Seg8[k8,:]=rawdata[i,:]
            k8=k8+1
        elif rawdata[i,0]<Age[8]:
            Seg9[k9,:]=rawdata[i,:]
            k9=k9+1
        else:
            Seg10[k10,:]=rawdata[i,:]
            k10=k10+1
    Seg1=Seg1[0:k1,:]
    Seg2=Seg2[0:k2,:]
    Seg3=Seg3[0:k3,:]
    Seg4=Seg4[0:k4,:]
    Seg5=Seg5[0:k5,:]
    Seg6=Seg6[0:k6,:]
    Seg7=Seg7[0:k7,:]
    Seg8=Seg8[0:k8,:]
    Seg9=Seg9[0:k9,:]
    Seg10=Seg10[0:k10,:]
    return Seg1,Seg2,Seg3,Seg4,Seg5,Seg6,Seg7,Seg8,Seg9,Seg10

def choose_fiber(Data,coff,threshold,x):
    New=Data[:,0]
    n=0
    Q=np.array([])
    if x==1:
        for i in range(50):
            if coff[i]>threshold:
                New=np.vstack((New,Data[:,i+1]))
                Q=np.hstack((Q,i+1))
                
    elif x==2:
        for i in range(50):
            if coff[i]<threshold:
                New=np.vstack((New,Data[:,i+1]))
                Q=np.hstack((Q,i+1))
                
    else:
        for i in range(50):
            if abs(coff[i])>threshold:
                New=np.vstack((New,Data[:,i+1]))
                Q=np.hstack((Q,i+1))
                
    return New.T,Q


def choose_test(Data,x):
    Test=Data[:,0]
    for i in range(len(x)):
        Test=np.vstack((Test,Data[:,int(x[i])]))
    return Test.T


# In[4]:


def ave(A,B):
    N=A.shape[0]
    C=np.zeros((N,1))
    for i in range(N):
        C[i]=(A[i]+B[i])/2
    return C


# In[5]:


file_path=r'C:\Users\Shi Wen\Desktop\DTI_fiber\mean_rawindex.xlsx'
sheet='mean1'
data=[]
xlsxfilein(file_path,sheet,data)
data=np.array(data)
print(data.shape,data.dtype) #identification
N1=data.shape[0]
N2=50  ##the number of fiber
rawdata=data[:,0:N2+1]
rawdata=rawdata.astype(float)
print('Type of raw data:',rawdata.dtype)
print('Size of raw data:',rawdata.shape)
print('Number of all training data:',rawdata.shape[0])
sele_data=rawdata


# In[ ]:


All_age=sele_data[:,0]
Year_fic11=29.5 #29.5 
Year_fic21=25.1 #26
Year_fic31=48
Year_fic41=42
All_data=sele_data[:,1:51]                            # training data all fiber
print('All Train Shape:',All_data.shape)
u=0
#training data all age
#Hyperparams
for x0 in range(5):
    Year_fic11=28+x0*2
    for y0 in range(5):
        Year_fic21=22+y0
        #if (Year_fic21+2)<Year_fic11:
        for z0 in range(10):
            Year_fic31=46+z0
            for a0 in range(8):
                Year_fic41=41+a0
                if (Year_fic41+6)>Year_fic31:
                    break
                else:
                            # training data all fiber
                    Age=np.array([Year_fic11,Year_fic21,Year_fic31,Year_fic41])                        #<25/25-55/>=55   1-30/20/55/40
                    Age1=np.array([Year_fic11,Year_fic11,Year_fic31,Year_fic31]) 
                    Age2=np.array([Year_fic21,Year_fic21,Year_fic41,Year_fic41]) 
                    YoungAll,MiddleAll,ElderlyAll=SplitAge(Age,rawdata,N2)
                    #Nu=100
                    #Young,Middle,Elderly=YoungAll[0:Nu,:],MiddleAll[0:Nu,:],ElderlyAll[0:Nu,:]
                    #All_data=ArrayAppend(Young,Middle,Elderly)[:,1:51]
                    #All_age=ArrayAppend(Young,Middle,Elderly)[:,0]
                    Young=YoungAll[:,:] 
                    Middle=MiddleAll[:,:] 
                    Elderly=ElderlyAll[:,:] 
                    Young_num=Young.shape[0]
                    Middle_num=Middle.shape[0]
                    Elderly_num=Elderly.shape[0]  
                    print('Young Matrix shape:',Young.shape)
                    Se=All_data.shape[1]
                    print('Young number:',Young_num,'  Middle number:',Middle_num,' Elderly number:',Elderly_num)
                    # age
                    y_cof,m_cof,e_cof=np.corrcoef(Young,rowvar=0),np.corrcoef(Middle,rowvar=0),np.corrcoef(Elderly,rowvar=0)
                    y_coff,m_coff,e_coff=y_cof[0,1:51],m_cof[0,1:51],e_cof[0,1:51]
                    print('Young coff size:',y_coff.shape,'Max:',np.max(y_coff),'Min:',np.min(y_coff))
                    print('Middle coff size:',m_coff.shape,'Max:',np.max(m_coff),'Min:',np.min(m_coff))
                    print('Elderly coff size:',e_coff.shape,'Max:',np.max(e_coff),'Min:',np.min(e_coff))
                    Yo,Yo_fiber=choose_fiber(Young,y_coff,0.1,3)
                    Mi,Mi_fiber=choose_fiber(Middle,m_coff,0.1,3)
                    El,El_fiber=choose_fiber(Elderly,e_coff,0.1,3)
                    print('Young choose fiber:',Yo_fiber)
                    print('Middle choose fiber:',Mi_fiber)
                    print('Elderly choose fiber:',El_fiber)
                    Young_age=Yo[:,0]
                    Middle_age=Mi[:,0]
                    Elderly_age=El[:,0]
                    # fiber preprocessing z-score
                    Y_fiber=Yo[:,1:len(Yo_fiber)+1]
                    M_fiber=Mi[:,1:len(Mi_fiber)+1]
                    E_fiber=El[:,1:len(El_fiber)+1]
                    ##----------------------------------Load Test Healthy Data
                    file_path=r'C:\Users\Shi Wen\Desktop\DTI_fiber\test_data_all.xlsx'
                    #sheet='median_com'
                    sheet='mean_com'
                    testdata=[]
                    a=0.3
                    b=0.7
                    Year_fic12=a*Year_fic11+(1-a)*Year_fic21
                    Year_fic22=Year_fic12
                    Year_fic32=b*Year_fic31+(1-b)*Year_fic41
                    Year_fic42=Year_fic32
                    xlsxfilein(file_path,sheet,testdata)
                    testdata=np.array(testdata)
                    print('Test Data Size:',testdata.shape,'Test Data Type:',testdata.dtype)                              #  identification
                    All_test_age=testdata[:,0]                                        #  test age
                    All_test=testdata[:,1:51]                                         #  test data
                    print('Shape of all healthy test:',All_test.shape)

                    Age=np.array([Year_fic12,Year_fic22,Year_fic32,Year_fic42])
                    Y_rawtest1,M_rawtest1,E_rawtest1=SplitAge(Age1,testdata,50)
                    Y_rawtest2,M_rawtest2,E_rawtest2=SplitAge(Age2,testdata,50)

                    print(Y_rawtest1.shape,M_rawtest1.shape,E_rawtest1.shape)
                    Y_test_age1,Y_test1=Y_rawtest1[:,0],choose_test(Y_rawtest1,Yo_fiber)[:,1:len(Yo_fiber)+1]
                    M_test_age1,M_test1=M_rawtest1[:,0],choose_test(M_rawtest1,Mi_fiber)[:,1:len(Mi_fiber)+1]
                    E_test_age1,E_test1=E_rawtest1[:,0],choose_test(E_rawtest1,El_fiber)[:,1:len(El_fiber)+1]
                    Y_test_age2,Y_test2=Y_rawtest2[:,0],choose_test(Y_rawtest2,Yo_fiber)[:,1:len(Yo_fiber)+1]
                    M_test_age2,M_test2=M_rawtest2[:,0],choose_test(M_rawtest2,Mi_fiber)[:,1:len(Mi_fiber)+1]
                    E_test_age2,E_test2=E_rawtest2[:,0],choose_test(E_rawtest2,El_fiber)[:,1:len(El_fiber)+1]
                    #------------------------------------------Load Patient Data
                    print('-----------------------------------------------------')
                    file_path=r'C:\Users\Shi Wen\Desktop\DTI_fiber\test_data_all.xlsx'
                    #sheet='median_pat'
                    sheet='mean_pat'
                    P_data=[]
                    xlsxfilein(file_path,sheet,P_data)
                    P_data=np.array(P_data)
                    print('Test Data Size:',P_data.shape,'Test Data Type:',P_data.dtype) 
                    P_All_test_age=P_data[:,0]                                             # patient test age
                    P_All_test=P_data[:,1:51]                                              #  patient test data
                    print('Shape of First Patients:',P_All_test.shape)
                    ###################################################################
                    PY_rawtest1,PM_rawtest1,PE_rawtest1=SplitAge(Age1,P_data,50)
                    PY_rawtest2,PM_rawtest2,PE_rawtest2=SplitAge(Age2,P_data,50)
                    print(PY_rawtest1.shape,PM_rawtest1.shape,PE_rawtest1.shape)
                    PY_age1,PY_test1=PY_rawtest1[:,0],choose_test(PY_rawtest1,Yo_fiber)[:,1:len(Yo_fiber)+1]
                    PM_age1,PM_test1=PM_rawtest1[:,0],choose_test(PM_rawtest1,Mi_fiber)[:,1:len(Mi_fiber)+1]
                    PE_age1,PE_test1=PE_rawtest1[:,0],choose_test(PE_rawtest1,El_fiber)[:,1:len(El_fiber)+1]
                    PY_age2,PY_test2=PY_rawtest2[:,0],choose_test(PY_rawtest2,Yo_fiber)[:,1:len(Yo_fiber)+1]
                    PM_age2,PM_test2=PM_rawtest2[:,0],choose_test(PM_rawtest2,Mi_fiber)[:,1:len(Mi_fiber)+1]
                    PE_age2,PE_test2=PE_rawtest2[:,0],choose_test(PE_rawtest2,El_fiber)[:,1:len(El_fiber)+1]
                    #################################################################### FA t2
                    file_path=r'C:\Users\Shi Wen\Desktop\DTI_fiber\FA_t2.xlsx'
                    P2_data=[]
                    xlsxfilein(file_path,'Sheet1',P2_data)
                    P2_data=np.array(P2_data)
                    print('Test Data Size:',P2_data.shape,'Test Data Type:',P2_data.dtype) 
                    P2_All_test_age=P2_data[:,0]                                             # patient test age
                    P2_All_test=P2_data[:,1:51]                                              #  patient test data
                    print('Shape of Second Patients:',P2_All_test.shape)
                    PY2_rawtest1,PM2_rawtest1,PE2_rawtest1=SplitAge(Age1,P2_data,50)
                    PY2_rawtest2,PM2_rawtest2,PE2_rawtest2=SplitAge(Age2,P2_data,50)
                    print(PY2_rawtest1.shape,PM2_rawtest1.shape,PE2_rawtest1.shape)
                    PY2_age1,PY2_test1=PY2_rawtest1[:,0],choose_test(PY2_rawtest1,Yo_fiber)[:,1:len(Yo_fiber)+1]
                    PM2_age1,PM2_test1=PM2_rawtest1[:,0],choose_test(PM2_rawtest1,Mi_fiber)[:,1:len(Mi_fiber)+1]
                    PE2_age1,PE2_test1=PE2_rawtest1[:,0],choose_test(PE2_rawtest1,El_fiber)[:,1:len(El_fiber)+1]
                    PY2_age2,PY2_test2=PY2_rawtest2[:,0],choose_test(PY2_rawtest2,Yo_fiber)[:,1:len(Yo_fiber)+1]
                    PM2_age2,PM2_test2=PM2_rawtest2[:,0],choose_test(PM2_rawtest2,Mi_fiber)[:,1:len(Mi_fiber)+1]
                    PE2_age2,PE2_test2=PE2_rawtest2[:,0],choose_test(PE2_rawtest2,El_fiber)[:,1:len(El_fiber)+1]
                    #####################################################################
                    file_path=r'C:\Users\Shi Wen\Desktop\DTI_fiber\FA_t3.xlsx'
                    P3_data=[]
                    xlsxfilein(file_path,'Sheet1',P3_data)
                    P3_data=np.array(P3_data)
                    print('Test Data Size:',P3_data.shape,'Test Data Type:',P3_data.dtype) 
                    P3_All_test_age=P3_data[:,0]                                             # patient test age
                    P3_All_test=P3_data[:,1:51]                                              #  patient test data
                    print('Shape of Second Patients:',P3_All_test.shape)
                    PY3_rawtest1,PM3_rawtest1,PE3_rawtest1=SplitAge(Age1,P3_data,50)
                    PY3_rawtest2,PM3_rawtest2,PE3_rawtest2=SplitAge(Age2,P3_data,50)
                    print(PY3_rawtest1.shape,PM3_rawtest1.shape,PE3_rawtest1.shape)
                    PY3_age1,PY3_test1=PY3_rawtest1[:,0],choose_test(PY3_rawtest1,Yo_fiber)[:,1:len(Yo_fiber)+1]
                    PM3_age1,PM3_test1=PM3_rawtest1[:,0],choose_test(PM3_rawtest1,Mi_fiber)[:,1:len(Mi_fiber)+1]
                    PE3_age1,PE3_test1=PE3_rawtest1[:,0],choose_test(PE3_rawtest1,El_fiber)[:,1:len(El_fiber)+1]
                    PY3_age2,PY3_test2=PY3_rawtest2[:,0],choose_test(PY3_rawtest2,Yo_fiber)[:,1:len(Yo_fiber)+1]
                    PM3_age2,PM3_test2=PM3_rawtest2[:,0],choose_test(PM3_rawtest2,Mi_fiber)[:,1:len(Mi_fiber)+1]
                    PE3_age2,PE3_test2=PE3_rawtest2[:,0],choose_test(PE3_rawtest2,El_fiber)[:,1:len(El_fiber)+1]
                    #####################################################################
    #                 file_path=r'C:\Users\Shi Wen\Desktop\DTI_fiber\mean_index.xlsx'
    #                 sheet='ank'
    #                 adata=[]
    #                 xlsxfilein(file_path,sheet,adata)
    #                 adata=np.array(adata)
    #                 print(adata.shape,adata.dtype) #identification
    #                 N1=adata.shape[0]
    #                 awdata=adata[:,0:51]
    #                 awdata=awdata.astype(float)
    #                 print('Test Data Size:',awdata.shape,'Test Data Type:',awdata.dtype) 
    #                 a_All_test_age=awdata[:,0]                                             # patient test age
    #                 a_All_test=awdata[:,1:51]                                              #  patient test data
    #                 print('Shape of Second Patients:',a_All_test.shape)
    #                 AY_rawtest,AM_rawtest,AE_rawtest=SplitAge(Age,awdata,50)
    #                 print(AY_rawtest.shape,AM_rawtest.shape,AE_rawtest.shape)
    #                 AY_age,AY_test=AY_rawtest[:,0],choose_test(AY_rawtest,Yo_fiber)[:,1:len(Yo_fiber)+1]
    #                 AM_age,AM_test=AM_rawtest[:,0],choose_test(AM_rawtest,Mi_fiber)[:,1:len(Mi_fiber)+1]
    #                 AE_age,AE_test=AE_rawtest[:,0],choose_test(AE_rawtest,El_fiber)[:,1:len(El_fiber)+1]
                    ####################################################################
                    file_path=r'C:\Users\Shi Wen\Desktop\DTI_fiber\FA_ank.xlsx'
                    P4_data=[]
                    xlsxfilein(file_path,'Sheet1',P4_data)
                    P4_data=np.array(P4_data)
                    print('Test Data Size:',P4_data.shape,'Test Data Type:',P4_data.dtype) 
                    P4_All_test_age=P4_data[:,0]                                             # patient test age
                    P4_All_test=P4_data[:,1:51]                                              #  patient test data
                    print('Shape of Second Patients:',P4_All_test.shape)
                    PY4_rawtest,PM4_rawtest,PE4_rawtest=SplitAge(Age,P4_data,50)
                    print(PY4_rawtest.shape,PM4_rawtest.shape,PE4_rawtest.shape)
                    PY4_age,PY4_test=PY4_rawtest[:,0],choose_test(PY4_rawtest,Yo_fiber)[:,1:len(Yo_fiber)+1]
                    PM4_age,PM4_test=PM4_rawtest[:,0],choose_test(PM4_rawtest,Mi_fiber)[:,1:len(Mi_fiber)+1]
                    PE4_age,PE4_test=PE4_rawtest[:,0],choose_test(PE4_rawtest,El_fiber)[:,1:len(El_fiber)+1]
                    print('Shape of Patients 1:',P_All_test.shape)
                    print('Shape of Patients 2:',P2_All_test.shape)
                    print('Shape of Patients 2:',P3_All_test.shape)
                    print('Shape of Patients 3:',P4_All_test.shape)
                    print('Young Train Samples Size:',Y_fiber.shape)
                    print('Middle Train Samples Size:',M_fiber.shape)
                    print('Elderly Train Samples Size:',E_fiber.shape)
                    print('Young Healthy Test shape:',Y_test1.shape)
                    print('Middle Healthy Test shape:',M_test1.shape)
                    print('Elderly Healthy Test shape:',E_test1.shape)
                    print('Young Patient:',PY_test1.shape,'Middle patients:',PM_test1.shape,'Elderly patients:',PE_test1.shape)
                    ###################
                    for a in range(1):
                        for b in range(1):
                            for c in range(1):
                                n1,n2,n3=8,4,4
                                PCA_ratio=n1#主成分所占权重
                                pca = PCA(n_components=PCA_ratio)
                                #--------------------Young
                                pca.fit(Y_fiber)
                                print('Young:',pca.explained_variance_ratio_,'Var:',pca.explained_variance_) 
                                Y_pca=pca.transform(Y_fiber)     #Young People Training PCA
#                                 pca.fit(Y_test1)
#                                 pca.fit(Y_test2)
                                Y_test_pca1=pca.transform(Y_test1)
                                Y_test_pca2=pca.transform(Y_test2)
                                pca.fit(PY_test1)
                                pca.fit(PY_test2)
                                PY_pca1=pca.transform(PY_test1)
                                PY_pca2=pca.transform(PY_test2)
                                pca.fit(PY2_test1)
                                pca.fit(PY2_test2)
                                PY2_pca1=pca.transform(PY2_test1)
                                PY2_pca2=pca.transform(PY2_test2)
                                pca.fit(PY3_test1)
                                pca.fit(PY3_test2)
                                PY3_pca1=pca.transform(PY3_test1)
                                PY3_pca2=pca.transform(PY3_test2)
                                PY4_pca=pca.transform(PY4_test)
                                #AY_pca=pca.transform(AY_test)
                                print('Young fiber PCA shape:',Y_pca.shape,'Young Test Sample Shape:',Y_test1.shape,'to',Y_test_pca1.shape,'PYpca:',PY_pca1.shape)

                                PCA_ratio=n2 #主成分所占权重
                                pca = PCA(n_components=PCA_ratio)
                                #--------------------Middle
                                pca.fit(M_fiber)
                                M_pca=pca.transform(M_fiber)
#                                 pca.fit(M_test1)
#                                 pca.fit(M_test2)
                                M_test_pca1=pca.transform(M_test1)
                                M_test_pca2=pca.transform(M_test2)
                                pca.fit(PM_test1)
                                pca.fit(PM_test2)
                                PM_pca1=pca.transform(PM_test1)
                                PM_pca2=pca.transform(PM_test2)
                                pca.fit(PM2_test1)
                                pca.fit(PM2_test2)
                                PM2_pca1=pca.transform(PM2_test1)
                                PM2_pca2=pca.transform(PM2_test2)
                                pca.fit(PM3_test1)
                                pca.fit(PM3_test2)
                                PM3_pca1=pca.transform(PM3_test1)
                                PM3_pca2=pca.transform(PM3_test2)
                                PM4_pca=pca.transform(PM4_test)
                                #AM_pca=pca.transform(AM_test)
                                print('Middle:',pca.explained_variance_ratio_,'Var:',pca.explained_variance_)

                                PCA_ratio=n3 #主成分所占权重
                                pca = PCA(n_components=PCA_ratio)
                                #--------------------Elderly
                                pca.fit(E_fiber)
                                E_pca=pca.transform(E_fiber)
                                pca.fit(E_test1)
                                pca.fit(E_test2)
                                E_test_pca1=pca.transform(E_test1)
                                E_test_pca2=pca.transform(E_test2)
                                pca.fit(PE_test1)
                                pca.fit(PE_test2)
                                PE_pca1=pca.transform(PE_test1)
                                PE_pca2=pca.transform(PE_test2)
                                pca.fit(PE2_test1)
                                pca.fit(PE2_test2)
                                PE2_pca1=pca.transform(PE2_test1)
                                PE2_pca2=pca.transform(PE2_test2)
                                pca.fit(PE3_test1)
                                pca.fit(PE3_test2)
                                PE3_pca1=pca.transform(PE3_test1)
                                PE3_pca2=pca.transform(PE3_test2)
                                PE4_pca=pca.transform(PE4_test)
                                #AE_pca=pca.transform(AE_test)
                                print('Elderly:',pca.explained_variance_ratio_,'Var:',pca.explained_variance_) 

                                # PCA_ratio=n4 #主成分所占权重
    #                             pca = PCA(n_components=PCA_ratio)
    #                             #----------------------All
    #                             pca.fit(All_data)
    #                             All_pca=pca.transform(All_data)
    #                             pca.fit(All_test)
    #                             All_test_pca=pca.transform(All_test)
    #                             pca.fit(P_All_test)
    #                             P_All_pca=pca.transform(P_All_test)

                                #---------------------display calculation
                #                 print(Y_pca.shape)
                #                 print(M_pca.shape)
                #                 print(E_pca.shape)
                #                 # print(All_pca.shape)
                #                 print('Young Patient Pca:',PY_pca.shape[0])
                #                 print('Middle Patient Pca:',PM_pca.shape[0])
                #                 print('Elderly Patient Pca:',PE_pca.shape[0])
                                # print('All Patient Pca:',P_All_pca.shape[0])
                                ########################################################### Model1-SVR
                                kernal='linear'
                                #ga=0.001
    #                             C=1e5
    #                             Y_score,Y_svrerror,Y_testerror,Y_svr_age,Y_pred_age=DTISVR3(Y_pca,Young_age,Y_test_pca,Y_test_age,kernal,C)
    #                             C=1e5
    #                             M_score,M_svrerror,M_testerror,M_svr_age,M_pred_age=DTISVR3(M_pca,Middle_age,M_test_pca,M_test_age,kernal,C)
    #                             C=1e5               
    #                             E_score,E_svrerror,E_testerror,E_svr_age,E_pred_age=DTISVR3(E_pca,Elderly_age,E_test_pca,E_test_age,kernal,C)
    #                             C=1e5
    #                             PY_score,PY_svrerror,PY_testerror,PY_svr_age,PY_pred_age=DTISVR3(Y_pca,Young_age,PY_pca,PY_age,kernal,C)
    #                             PY2_score,PY2_svrerror,PY2_testerror,PY2_svr_age,PY2_pred_age=DTISVR3(Y_pca,Young_age,PY2_pca,PY2_age,kernal,C)
    #                             PY3_score,PY3_svrerror,PY3_testerror,PY3_svr_age,PY3_pred_age=DTISVR3(Y_pca,Young_age,PY3_pca,PY3_age,kernal,C)
    #                             PY4_score,PY4_svrerror,PY4_testerror,PY4_svr_age,PY4_pred_age=DTISVR3(Y_pca,Young_age,PY4_pca,PY4_age,kernal,C)
    #                             #AY_score,AY_svrerror,AY_testerror,AY_svr_age,AY_pred_age=DTISVR3(Y_pca,Young_age,AY_pca,AY_age,kernal,C,ga)
    #                             C=1e5
    #                             PM_score,PM_svrerror,PM_testerror,PM_svr_age,PM_pred_age=DTISVR3(M_pca,Middle_age,PM_pca,PM_age,kernal,C)
    #                             PM2_score,PM2_svrerror,PM2_testerror,PM2_svr_age,PM2_pred_age=DTISVR3(M_pca,Middle_age,PM2_pca,PM2_age,kernal,C)
    #                             PM3_score,PM3_svrerror,PM3_testerror,PM3_svr_age,PM3_pred_age=DTISVR3(M_pca,Middle_age,PM3_pca,PM3_age,kernal,C)
    #                             PM4_score,PM4_svrerror,PM4_testerror,PM4_svr_age,PM4_pred_age=DTISVR3(M_pca,Middle_age,PM4_pca,PM4_age,kernal,C)
    #                             #AM_score,AM_svrerror,AM_testerror,AM_svr_age,AM_pred_age=DTISVR3(M_pca,Middle_age,AM_pca,AM_age,kernal,C,ga)
    #                             C=1e5
    #                             PE_score,PE_svrerror,PE_testerror,PE_svr_age,PE_pred_age=DTISVR3(E_pca,Elderly_age,PE_pca,PE_age,kernal,C)
    #                             PE2_score,PE2_svrerror,PE2_testerror,PE2_svr_age,PE2_pred_age=DTISVR3(E_pca,Elderly_age,PE2_pca,PE2_age,kernal,C)
    #                             PE3_score,PE3_svrerror,PE3_testerror,PE3_svr_age,PE3_pred_age=DTISVR3(E_pca,Elderly_age,PE3_pca,PE3_age,kernal,C)
    #                             PE4_score,PE4_svrerror,PE4_testerror,PE4_svr_age,PE4_pred_age=DTISVR3(E_pca,Elderly_age,PE4_pca,PE4_age,kernal,C)
                                #AE_score,AE_svrerror,AE_testerror,AE_svr_age,AE_pred_age=DTISVR3(E_pca,Elderly_age,AE_pca,AE_age,kernal,C,ga)

                                Y_score1,Y_svrerror1,Y_testerror1,Y_svr_age1,Y_pred_age1=DTIRVR3(Y_pca,Young_age,Y_test_pca1,Y_test_age1,kernal)
                                M_score1,M_svrerror1,M_testerror1,M_svr_age1,M_pred_age1=DTIRVR3(M_pca,Middle_age,M_test_pca1,M_test_age1,kernal)              
                                E_score1,E_svrerror1,E_testerror1,E_svr_age1,E_pred_age1=DTIRVR3(E_pca,Elderly_age,E_test_pca1,E_test_age1,kernal)
                                Y_score2,Y_svrerror2,Y_testerror2,Y_svr_age2,Y_pred_age2=DTIRVR3(Y_pca,Young_age,Y_test_pca2,Y_test_age2,kernal)
                                M_score2,M_svrerror2,M_testerror2,M_svr_age2,M_pred_age2=DTIRVR3(M_pca,Middle_age,M_test_pca2,M_test_age2,kernal)              
                                E_score2,E_svrerror2,E_testerror2,E_svr_age2,E_pred_age2=DTIRVR3(E_pca,Elderly_age,E_test_pca2,E_test_age2,kernal)
                                PY_score1,PY_svrerror1,PY_testerror1,PY_svr_age1,PY_pred_age1=DTIRVR3(Y_pca,Young_age,PY_pca1,PY_age1,kernal)
                                PY_score2,PY_svrerror2,PY_testerror2,PY_svr_age2,PY_pred_age2=DTIRVR3(Y_pca,Young_age,PY_pca2,PY_age2,kernal)
                                PY2_score1,PY2_svrerror1,PY2_testerror1,PY2_svr_age1,PY2_pred_age1=DTIRVR3(Y_pca,Young_age,PY2_pca1,PY2_age1,kernal)
                                PY2_score2,PY2_svrerror2,PY2_testerror2,PY2_svr_age2,PY2_pred_age2=DTIRVR3(Y_pca,Young_age,PY2_pca2,PY2_age2,kernal)
                                PY3_score1,PY3_svrerror1,PY3_testerror1,PY3_svr_age1,PY3_pred_age1=DTIRVR3(Y_pca,Young_age,PY3_pca1,PY3_age1,kernal)
                                PY3_score2,PY3_svrerror2,PY3_testerror2,PY3_svr_age2,PY3_pred_age2=DTIRVR3(Y_pca,Young_age,PY3_pca2,PY3_age2,kernal)
                                PY4_score,PY4_svrerror,PY4_testerror,PY4_svr_age,PY4_pred_age=DTIRVR3(Y_pca,Young_age,PY4_pca,PY4_age,kernal)
                                #AY_score,AY_svrerror,AY_testerror,AY_svr_age,AY_pred_age=DTIRVR3(Y_pca,Young_age,AY_pca,AY_age,kernal)
                                PM_score1,PM_svrerror1,PM_testerror1,PM_svr_age1,PM_pred_age1=DTIRVR3(M_pca,Middle_age,PM_pca1,PM_age1,kernal)
                                PM_score2,PM_svrerror2,PM_testerror2,PM_svr_age2,PM_pred_age2=DTIRVR3(M_pca,Middle_age,PM_pca2,PM_age2,kernal)
                                PM2_score1,PM2_svrerror1,PM2_testerror1,PM2_svr_age1,PM2_pred_age1=DTIRVR3(M_pca,Middle_age,PM2_pca1,PM2_age1,kernal)
                                PM2_score2,PM2_svrerror2,PM2_testerror2,PM2_svr_age2,PM2_pred_age2=DTIRVR3(M_pca,Middle_age,PM2_pca2,PM2_age2,kernal)
                                PM3_score1,PM3_svrerror1,PM3_testerror1,PM3_svr_age1,PM3_pred_age1=DTIRVR3(M_pca,Middle_age,PM3_pca1,PM3_age1,kernal)
                                PM3_score2,PM3_svrerror2,PM3_testerror2,PM3_svr_age2,PM3_pred_age2=DTIRVR3(M_pca,Middle_age,PM3_pca2,PM3_age2,kernal)
                                PM4_score,PM4_svrerror,PM4_testerror,PM4_svr_age,PM4_pred_age=DTIRVR3(M_pca,Middle_age,PM4_pca,PM4_age,kernal)
                                #AM_score,AM_svrerror,AM_testerror,AM_svr_age,AM_pred_age=DTIRVR3(M_pca,Middle_age,AM_pca,AM_age,kernal)
                                PE_score1,PE_svrerror1,PE_testerror1,PE_svr_age1,PE_pred_age1=DTIRVR3(E_pca,Elderly_age,PE_pca1,PE_age1,kernal)
                                PE_score2,PE_svrerror2,PE_testerror2,PE_svr_age2,PE_pred_age2=DTIRVR3(E_pca,Elderly_age,PE_pca2,PE_age2,kernal)
                                PE2_score1,PE2_svrerror1,PE2_testerror1,PE2_svr_age1,PE2_pred_age1=DTIRVR3(E_pca,Elderly_age,PE2_pca1,PE2_age1,kernal)
                                PE2_score2,PE2_svrerror2,PE2_testerror2,PE2_svr_age2,PE2_pred_age2=DTIRVR3(E_pca,Elderly_age,PE2_pca2,PE2_age2,kernal)
                                PE3_score1,PE3_svrerror1,PE3_testerror1,PE3_svr_age1,PE3_pred_age1=DTIRVR3(E_pca,Elderly_age,PE3_pca1,PE3_age1,kernal)
                                PE3_score2,PE3_svrerror2,PE3_testerror2,PE3_svr_age2,PE3_pred_age2=DTIRVR3(E_pca,Elderly_age,PE3_pca2,PE3_age2,kernal)
                                PE4_score,PE4_svrerror,PE4_testerror,PE4_svr_age,PE4_pred_age=DTIRVR3(E_pca,Elderly_age,PE4_pca,PE4_age,kernal)
                                #AE_score,AE_svrerror,AE_testerror,AE_svr_age,AE_pred_age=DTIRVR3(E_pca,Elderly_age,AE_pca,AE_age,kernal)

                                P_pred_age=ave(ArrayAppend(PY_pred_age1,PM_pred_age1,PE_pred_age1),ArrayAppend(PY_pred_age2,PM_pred_age2,PE_pred_age2))
                                svr_Splittest1=ArrayAppend(Y_testerror1,M_testerror1,E_testerror1)
                                svr_Splittest2=ArrayAppend(Y_testerror2,M_testerror2,E_testerror2)
                                print('1:',svr_Splittest1.shape[0],'2:',svr_Splittest2.shape)
                                svr_Splittest=ave(svr_Splittest1,svr_Splittest2)
                                svr_SplitPatients1=ArrayAppend(PY_testerror1,PM_testerror1,PE_testerror1)
                                svr_SplitPatients2=ArrayAppend(PY_testerror2,PM_testerror2,PE_testerror2)
                                svr_SplitPatients=ave(svr_SplitPatients1,svr_SplitPatients2)
                                svr_SplitPatients21=ArrayAppend(PY2_testerror1,PM2_testerror1,PE2_testerror1)
                                svr_SplitPatients22=ArrayAppend(PY2_testerror2,PM2_testerror2,PE2_testerror2)
                                svr_SplitPatients2=ave(svr_SplitPatients21,svr_SplitPatients22)
                                svr_SplitPatients31=ArrayAppend(PY3_testerror1,PM3_testerror1,PE3_testerror1)
                                svr_SplitPatients32=ArrayAppend(PY3_testerror2,PM3_testerror2,PE3_testerror2)
                                svr_SplitPatients3=ave(svr_SplitPatients31,svr_SplitPatients32)
                                #svr_SplitPatients4=ArrayAppend(AY_testerror,AM_testerror,AE_testerror)
                                svr_SplitPatients5=ArrayAppend(PY4_testerror,PM4_testerror,PE4_testerror)
                                print('SVR Patient Shape:',svr_SplitPatients.shape)
                                plt.figure(figsize=(8,8))
                                c1=pd.Series(svr_Splittest.T[0])
                                c2=pd.Series(svr_SplitPatients.T[0])
                                c3=pd.Series(svr_SplitPatients2.T[0])
                                c4=pd.Series(svr_SplitPatients3.T[0])
                                #c5=pd.Series(svr_SplitPatients4.T[0])
                                c6=pd.Series(svr_SplitPatients5.T[0])
                                form=pd.DataFrame({
                                    "Healthy Test":c1, 
                                    "Patients 1":c2,
                                    "Patients 2":c3,
                                    "Patients 3":c4,
                                    #"Ank Healthy":c5,
                                    "ANK Patients":c6,
                                })  
                                form.boxplot()  
                                plt.ylabel("MAE(Year)")  
                                plt.xlabel("Different Samples") 
                                plt.title("RVR Test Bias Comparison")
                                if np.mean(svr_Splittest)<np.mean(svr_SplitPatients) and (np.max(svr_Splittest)-np.min(svr_Splittest))<30 and np.mean(svr_SplitPatients)>0:
                                    filepath='C:/Users/Shi Wen/Desktop/DTI_fiber/Box/'+str(u)+" PCA="+str(n1)+str(n2)+str(n3)+' f1='+str(Year_fic11)+' f2='+str(Year_fic21)+' f3='+str(Year_fic31)+' f4='+str(Year_fic41)+' f11='+str(Year_fic12)+' f22'+str(Year_fic32)+' Healthy='+str(np.mean(svr_Splittest))+' Patients='+str(np.mean(svr_SplitPatients))+'.jpg' 
                                    plt.savefig(filepath,dpi=160)

                                    print("Elderly Train Data Picture "+str(u)+" Save!")
                                plt.figure(figsize=(8,8))
                                if np.corrcoef(P_All_test_age,svr_SplitPatients,rowvar=0)[0,1]>0:
                                    plt.scatter(P_All_test_age,svr_SplitPatients, color='black',label='All Age')
                                    plt.plot([0,90],[0,0],color='black',label='Real=Prediction')
                                    plt.xlabel('Real Age')
                                    plt.ylabel('Prediction Age')
                                    plt.xticks([-10,10,20,30,40,50,60,70,80,90],[-10,10,20,30,40,50,60,70,80,90])
                                    plt.yticks([-10,10,20,30,40,50,60,70,80,90],[-10,10,20,30,40,50,60,70,80,90])
                                    plt.title('Support Vector Regression Training Result (Training number=650)')
                                    plt.legend(loc='best')
                                    filepath='C:/Users/Shi Wen/Desktop/DTI_fiber/Box/'+str(u)+' coff='+str(np.corrcoef(P_All_test_age,svr_SplitPatients,rowvar=0)[0,1])+' f1='+str(Year_fic11)+' f2='+str(Year_fic21)+' f3='+str(Year_fic31)+' f4='+str(Year_fic41)+'.jpg'
                                    plt.savefig(filepath,dpi=160)
                                
                                u=u+1
                                    #------------------------------------------------


# In[9]:


age_t=np.append(Y_test_age1,M_test_age1)
age_t=np.append(age_t,E_test_age1)
# print(age_t.shape)
# age_t=age_t.tolist()
age_e1=np.append(Y_testerror1,M_testerror1)
age_e1=np.append(age_e1,E_testerror1)
age_e2=np.append(Y_testerror2,M_testerror2)
age_e2=np.append(age_e2,E_testerror2)
age_e=ave(age_e1,age_e2)
# print(age_e.shape)
# age_e=age_e.tolist()
age_pt=np.append(PY_age1,PM_age1)
age_pt=np.append(age_pt,PE_age1)
age_pe1=np.append(PY_testerror1,PM_testerror1)
age_pe1=np.append(age_pe1,PE_testerror1)
age_pe2=np.append(PY_testerror2,PM_testerror2)
age_pe2=np.append(age_pe2,PE_testerror2)
age_pe=ave(age_pe1,age_pe2)
####################################
# age_pt=np.append(Young_age,Middle_age)
# age_pt=np.append(age_pt,Elderly_age)
# age_pe=np.append(PY_svrerror,PM_svrerror)
# age_pe=np.append(age_pe,PE_svrerror)
##########################################
age_pt2=P2_All_test_age
age_pe21=np.append(PY2_testerror1,PM2_testerror1)
age_pe21=np.append(age_pe21,PE2_testerror1)
age_pe22=np.append(PY2_testerror2,PM2_testerror2)
age_pe22=np.append(age_pe22,PE2_testerror2)
age_pe2=ave(age_pe21,age_pe22)

age_pt3=P3_All_test_age
age_pe31=np.append(PY3_testerror1,PM3_testerror1)
age_pe31=np.append(age_pe31,PE3_testerror1)
age_pe32=np.append(PY3_testerror2,PM3_testerror2)
age_pe32=np.append(age_pe32,PE3_testerror2)
age_pe3=ave(age_pe31,age_pe32)

age_pt4=np.append(PY4_age,PM4_age)
age_pt4=np.append(age_pt4,PE4_age)
age_pe4=np.append(PY4_testerror,PM4_testerror)
age_pe4=np.append(age_pe4,PE4_testerror)

# age_pt5=np.append(AY_age,AM_age)
# age_pt5=np.append(age_pt5,AE_age)
# age_pe5=np.append(AY_testerror,AM_testerror)
# age_pe5=np.append(age_pe5,AE_testerror)

print(age_pt.shape)
def sumpa(age_t,age_e,age_pt,age_pe,age_pt2,age_pe2,age_pt3,age_pe3,age_pt4,age_pe4):
    X=np.zeros((len(age_pe),13))
    for i in range((len(age_e))):
        X[i,0]=age_t[i]
        X[i,1]=age_e[i]
    for i in range(len(age_pe)):
        X[i,2]=age_pt[i]
        X[i,3]=age_pe[i]
    for i in range(len(age_pe2)):
        X[i,4]=age_pt2[i]
        X[i,5]=age_pe2[i]
    for i in range(len(age_pe3)):
        X[i,6]=age_pt3[i]
        X[i,7]=age_pe3[i]
    for i in range(len(age_pe4)):
        X[i,8]=age_pt4[i]
        X[i,9]=age_pe4[i]
#     for i in range(len(u)):
#         X[i,10]=u[i]
#     for i in range(len(age_pe5)):
#         X[i,10]=age_pt5[i]
#         X[i,11]=age_pe5[i]
    return X

X=sumpa(age_t,age_e,age_pt,age_pe,age_pt2,age_pe2,age_pt3,age_pe3,age_pt4,age_pe4)    
print(X.shape)
name=['Healthy Real','Age','Patient Real Age1','Pred Age','2','Age','3','Age','Ank_H','Age','Ank_P','Age','1']
X=X.tolist()
#lt=lt.reverse()
print(X)
#print(age_t)
#print(age_e)
test=pd.DataFrame(data=X,columns=name)
test.to_csv('C:/Users/Shi Wen/Desktop/1.csv')


# In[ ]:


All_score,All_svrerror,All_testerror,All_svr_age,All_pred_age=DTIRVR3(All_data,All_age,All_test,All_test_age,kernal)
P_score,P_svrerror,P_testerror,P_svr_age,P_pred_age=DTIRVR3(All_data,All_age,P_All_test,P_All_test_age,kernal)
plt.figure(figsize=(8,8))
plt.scatter(P_All_test_age,P_pred_age, color='black',label='All Age')
plt.plot([0,90],[0,90],color='black',label='Real=Prediction')
plt.xlabel('Real Age')
plt.ylabel('Prediction Age')
plt.xticks([10,20,30,40,50,60,70,80,90],[10,20,30,40,50,60,70,80,90])
plt.yticks([10,20,30,40,50,60,70,80,90],[10,20,30,40,50,60,70,80,90])
plt.title('Support Vector Regression Training Result (Training number=650)')
plt.legend(loc='best')


# In[ ]:


print(P_data.shape,P_All_test.shape)


# In[ ]:


P_cof=np.corrcoef(P_data,rowvar=0)
print(P_cof[0,:])


# In[ ]:


a=np.corrcoef(P_All_test_age,svr_SplitPatients,rowvar=0)
print(a)


# In[10]:


print(m_coff)


# In[69]:


Mo,Mo_fiber=choose_fiber(Young,m_coff,0.2,3)
print(Mo_fiber)


# In[24]:


print(m_coff)


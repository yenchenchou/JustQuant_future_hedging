#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 22:50:13 2018

@author: yc
"""
#sudo pip install quandl
import quandl
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

os.getcwd()

# =============================================================================
# DATA API CODE FOR EACH TIME PERIOD
# =============================================================================

# API_CBOT
corn = {'CME/CN2018':'Corn1','CME/CN2017':'Corn2','CME/CN2016':'Corn3','CME/CN2015':'Corn4', 'CME/CN2014':'Corn5'}
soybean = {'CME/SN2018':'Soyoil1','CME/SN2017':'Soyoil2','CME/SN2016':'Soyoil3','CME/SN2015':'Soyoil4','CME/SN2014':'Soyoil5'}
soybeanoil = {'CME/BON2018':'Soyflour1','CME/BON2017':'Soyflour2','CME/BON2016':'Soyflour3','CME/BON2015':'Soyflour4','CME/BON2014':'Soyflour5'}
wheat = {'CME/WN2018':'Wheat1','CME/WN2017':'Wheat2','CME/WN2016':'Wheat3','CME/WN2015':'Wheat4','CME/WN2014':'Wheat5'}
#mini_dow = {'CME/YMU2018':'Mini-Dow1', 'CME/YMU2017':'Mini-Dow2', 'CME/YMU2016':'Mini-Dow3','CME/YMU2015':'Mini-Dow4','CME/YMU2014':'Mini-Dow5'}

# API_CME
sp500 = {'CME/SPU2018':'SPM1', 'CME/SPU2017':'SPM2', 'CME/SPU2016':'SPM3', 'CME/SPU2015':'SPM4', 'CME/SPU2014':'SPM5'}
#mini_nasdaq = {'CME/NQU2018':'NQU1','CME/NQU2017':'NQU2','CME/NQU2016':'NQU3','CME/NQU2015':'NQU4','CME/NQU2014':'NQU5'}
jap_yen = {'CME/JYU2018':'JY1','CME/JYU2017':'JY2','CME/JYU2016':'JY3','CME/JYU2015':'JY4','CME/JYU2014':'JY5'}
swiss_france = {'CME/SFU2018':'SFZ1','CME/SFU2017':'SFZ2','CME/SFU2016':'SFZ3','CME/SFU2015':'SFZ4','CME/SFU2014':'SFZ5'}
british_bound = {'CME/BPU2018':'BP1','CME/BPU2017':'BP2','CME/BPU2016':'BP3','CME/BPU2015':'BP4','CME/BPU2014':'BP5'}
canada_dollar = {'CME/CDU2018':'CD1','CME/CDU2017':'CD2','CME/CDU2016':'CD3','CME/CDU2015':'CD4','CME/CDU2014':'CD5'}
australia_dollar = {'CME/ADU2018':'AD1','CME/ADU2017':'AD2','CME/ADU2016':'AD3','CME/ADU2015':'AD4','CME/ADU2014':'AD5'}
euro_dollar = {'CME/EDU2018':'ED1','CME/EDU2017':'ED2','CME/EDU2016':'ED3','CME/EDU2015':'ED4','CME/EDU2014':'ED5'}
newzeland_dollar = {'CME/NEU2018':'NEU1','CME/NEU2017':'NEU2','CME/NEU2016':'NEU3','CME/NEU2015':'NEU4','CME/NEU2014':'NEU5'}

# API_NYMEX 
gold = {'CME/GCM2018':'GC1','CME/GCM2017':'GC2','CME/GCM2016':'GC3','CME/GCM2015':'GC4','CME/GCM2014':'GC5'}
silver = {'CME/SIM2018':'SI1','CME/SIM2017':'SI2','CME/SIM2016':'SI3','CME/SIM2015':'SI4','CME/SIM2014':'SI5'}####need more api code
copper = {'CME/HGM2018':'HG1','CME/HGM2017':'HG2','CME/HGM2016':'HG3','CME/HGM2015':'HG4','CME/HGM2014':'HG5'}
crudeoil = {'CME/CLU2018':'CL1','CME/CLU2017':'CL4','CME/CLU2016':'CL3','CME/CLU2015':'CL4','CME/CLU2014':'CL5'}
RBOBgasoline = {'ICE/NN2018':'RBOB1','ICE/NN2017':'RBOB2','ICE/NN2016':'RBOB3','ICE/NN2015':'RBOB4','ICE/NN2014':'RBOB5'}#####problem!!!!
naturalgas = {'CME/NGN2018':'NG1','CME/NGN2017':'NG2','CME/NGN2016':'NG3','CME/NGN2015':'NG4','CME/NGN2014':'NG5'}
Platinum = {'CME/PLN2018':'PLV1','CME/PLN2017':'PLV2','CME/PLN2016':'PLV3','CME/PLN2015':'PLV4','CME/PLN2014':'PLV5'}
palladium = {'CME/PAM2018':'PAZ1','CME/PAM2017':'PAZ2','CME/PAM2016':'PAZ3','CME/PAM2015':'PAZ4','CME/PAM2014':'PAZ5'}

# API_CFE 
volatility = {'CHRIS/CBOE_VX1':'VIX'}#####

# API_ICE
coffee = {'ICE/KCN2018':'KC1','ICE/KCN2017':'KC2','ICE/KCN2016':'KC3','ICE/KCN2015':'KC4','ICE/KCN2014':'KC5'}
sugar = {'ICE/SBN2018':'SB1','ICE/SBN2017':'SB2','ICE/SBN2016':'SB3','ICE/SBN2015':'SB4','ICE/SBN2014':'SB5'}
cocoa= {'ICE/CCN2018':'CC1','ICE/CCN2017':'CC2','ICE/CCN2016':'CC3','ICE/CCN2015':'CC4','ICE/CCN2014':'CC5'}
cotton = {'ICE/CTN2018':'CT1','ICE/CTN2017':'CT4','ICE/CTN2016':'CT3','ICE/CTN2015':'CT4','ICE/CTN2014':'CT5'}
orangejuice = {'ICE/OJN2018':'OJ1','ICE/OJN2017':'OJ2','ICE/OJN2016':'OJ3','ICE/OJN2015':'OJ4','ICE/OJN2014':'OJ5'}
usindex = {'ICE/DXU2018':'USDINDEX1','ICE/DXU2017':'USDINDEX2','ICE/DXU2016':'USDINDEX3','ICE/DXU2015':'USDINDEX4','ICE/DXU2014':'USDINDEX5'}
brentoil = {'ICE/BU2018':'B1','ICE/BU2017':'B2','ICE/BU2016':'B3','ICE/BU2015':'B4','ICE/BU2014':'B5'}
dax = {'EUREX/FDAXU2018':'DAX1','EUREX/FDAXU2017':'DAX2','EUREX/FDAXU2016':'DAX3','EUREX/FDAXU2015':'DAX4','EUREX/FDAXU2014':'DAX5'}####need more api code
            
# API_LME
API_LME = {'LME/PR_CU':'PR_CU',
           'LME/PR_AL':'PR_AL',
           'LME/PR_TN':'PR_TN',
           'LME/PR_NI':'PR_NI',
           'LME/PR_ZI':'PR_ZI',
           'LME/PR_PB':'PR_PB'            
           }

# API_SGX 
nikkei225 = {'SGX/NKU2018':'NKU1','SGX/NKU2017':'NKU2','SGX/NKU2016':'NKU3','SGX/NKU2015':'NKU4','SGX/NKU2014':'NKU5'}
a50 = {'SGX/CNM2018':'A501','SGX/CNM2017':'A502','SGX/CNM2016':'A503','SGX/CNM2015':'A504','SGX/CNM2014':'A505'}


# API_HKEx
hangseng = {'CHRIS/HKEX_HSI1':'HSI', 'HKEX/HSIM2018':'HSIM1','HKEX/HSIZ2017':'HSIM2','HKEX/HSIM2017':'HSIM3','HKEX/HSIZ2016':'HSIM4',
            'HKEX/HSIM2016':'HSIM5','HKEX/HSIZ2015':'HSIM6','HKEX/HSIN2015':'HSIM7','HKEX/HSIU2014':'HSIM8','HKEX/HSIM2014':'HSIM9'
            }


# =============================================================================
# GRAB THE DATA FROM API CODE
# =============================================================================
def get_data_from_quandl(data):
    
    '''
    Get the data from Quandl API and put the data in a list format
    '''
    global data_list
    data_list=[]      
    for data_api, data_name in data.items():
        data_name = quandl.get(data_api,authtoken='r4HFZkruAmhhszee71tK')
        data_list.append(data_name)        

            
def get_info(data_list):
    
    '''
    Printing info of each dataset
    '''
    for df in data_list:
        print(df.info(),'\n')
        
get_data_from_quandl(corn)        
get_info(data_list)       

# =============================================================================
# EXRTRACT DATA BASIC ON VOLUME OF TRANSACTION
# =============================================================================
def sort_volumne_data(data_list, file_name):
    
    '''
    Since products with in the same day may appear in 
    different contract. We need sort the data with largest 
    volume of transaction
    '''      
    data_list = pd.concat(data_list, axis=0)
    data_list = data_list.sort_index()
    data_list = data_list.reset_index()
    data_list = data_list.sort_values(['Date','Volume'], ascending=False)
    data_list = data_list.drop_duplicates(subset=['Date'],keep='first')    
    data_list.to_csv(path_or_buf='/Users/yc/Desktop/AppWorks_StartupHackers/Fintech/JustQuant/'+ file_name +'.csv')

# =============================================================================
# IMPORT INDIVIDUAL DATA AND EXPORT THE MERGED DATA FROM ALL PRODUCTS
# =============================================================================
os.chdir('/Users/yc/Desktop/AppWorks_StartupHackers/Fintech/JustQuant/180628-Invest.com資料爬蟲-海期')

CBOT=['US Corn Futures Historical Data',
      'US Soybeans Futures Historical Data',
      'US Soybean Oil Futures Historical Data',
      'US Soybean Meal Futures Historical Data',
      'US Wheat Futures Historical Data',
      'Dow 30 Futures Historical Data',
      'S&P 500 Futures Historical Data',
      'Nasdaq Futures Historical Data']
CBOT_name = ['Corn','Soybean','Soybean_oil','Soybean_meal','Wheat','Mini_dow','SP500','NASDAQ']

CME=['日圓期貨價格_130318-180626 拷貝',
     '瑞士法郎期貨價格_130318-180626 拷貝',
     '英鎊期貨價格_130318-180626 拷貝',
     '加幣期貨價格_130319-180626 拷貝',
     '澳幣期貨價格_130318-180626 拷貝',
     '歐元期貨價格_041001-180626 拷貝',
     '紐西蘭幣期貨價格_130318-180626 拷貝',
     '日經指數期貨價格_081114-180626 拷貝']
CME_name = ['JapYen','SwissFranc','British','Canadian',
            'Australian','Europe','Newzealand','Nikkei']
          
NYMEX=['Gold Futures Historical Data',
       'Silver Futures Historical Data',
       'Copper HG Futures Historical Data',
       'Crude Oil WTI Futures Historical Data',
       'Gasoline RBOB Futures Historical Data',
       'Natural Gas Futures Historical Data',
       'Heating Oil Futures Historical Data',
       'Platinum Futures Historical Data',
       'Palladium Futures Historical Data']
NYMEX_name=['Gold','Silver','Copper_HG','Crudeoil','Gasoline',
       'Naturalgas','Heatoil','Platinum','Palladium']

CFE=['S&P 500 VIX Futures Historical Data']         
CFE_name=['SP500VIX']

NYBOT=['US Coffee C Futures Historical Data',
       'US Sugar #11 Futures Historical Data',
       'US Cocoa Futures Historical Data',
       'US Cotton #2 Futures Historical Data',
       'Orange Juice Futures Historical Data',
       'US Dollar Index Futures Historical Data']
NYBOT_name=['Coffee','Sugar','Cocoa','Cotton','Orange','USindex']

ICE = ['Brent Oil Futures Historical Data']
ICE_name = ['Brentoil']

EUREX=['DAX Futures Historical Data']
EUREX_name=['DAX']

LME=['Copper MCU Futures Historical Data',
     'Aluminum Futures Historical Data',
     'Tin Futures Historical Data',
     'Nickel Futures Historical Data',
     'Zinc Futures Historical Data',
     'Lead Futures Historical Data']
LME_name=['Copper_MCU','Aluminum','Tin','Nickel','Zinc','Lead']

SGX=['China A50 Futures Historical Data']
SGX_name=['A50']

HKEx=['Hang Seng Futures Historical Data']
HKEx_name=['HangSeng']

def read_df(datalist):
    '''
    Read all the downlaoded dataframe for Quandl & Invest.com
    '''
    global all_df
    all_df=[]
    for filename in datalist:
        df = pd.read_csv(filename + '.csv', usecols=['Date','Price'], 
                         parse_dates=True, index_col='Date')
        df.sort_index(ascending=False)
        all_df.append(df)
    return all_df  


def concat_and_colname(all_df, col_names):  
     
    x = pd.concat(all_df, axis=1, join='outer')
    x.columns = col_names
    x.sort_index(ascending=False)
    return x


read_df(LME)
df_LME = concat_and_colname(all_df, LME_name)
df_LME.info()

df_outerjoin = pd.concat([df_CBOT, df_CME, df_NYMEX,
                          df_CFE, df_NYBOT, df_ICE, df_EUREX,
                          df_LME, df_SGX, df_HKEx], axis=1)
df_outerjoin.info()
df = df_outerjoin.apply(replace('',))
df_outerjoin.to_excel('/Users/yc/Desktop/AppWorks_StartupHackers/Fintech/JustQuant/180628-海期42項Hedge資料-v1r00.xlsx')

# =============================================================================
# IMPORT ALL THE PRODUCTS AND DATA ANALYSING
# =============================================================================
df = pd.read_excel('/Users/yc/Desktop/AppWorks_StartupHackers/'\
                   'Fintech/JustQuant/180628-海期42項Hedge資料-v1r00.xlsx',
                   parse_dates=True, index_col='Date')
df.columns
df = df.drop(['Mini_dow','NASDAQ'],axis=1)

def get_time_period(df):
    '''Get the shortest data and use that column as base track'''
    nulldata = df.isnull().sum()
    nulldata = nulldata.reset_index()
    nulldata.columns = ['col_names', 'null_count']
    sns.factorplot(x = 'null_count', y='col_names', data = nulldata, kind='bar')

#['2017-07-01':'2017-12-31']
def fill_na():
    global df
    df = df['2016-01-01':'2018-06-30'].copy()
    a = df.drop(['Silver','Platinum'], axis=1)
    a = a.dropna(how='all')
    b = a.drop(['Copper_HG'], axis=1)
    b = b.dropna(how='all')
    df1 = df[['Silver','Platinum','Copper_HG']]
    result = b.join(df1, how='left')
    result = result.fillna(method='ffill')
    result = result.fillna(method='bfill')
    result.isnull().sum()
    df = result
    return df


get_time_period(df)
df = fill_na()
df.info()

# =============================================================================
# DATA ANALYSING
# =============================================================================
'''Find unusual value'''
def find_unusual_value():
    
    '''
    When drawing the plot, we can see
    there are some unsuitable in the data.
    So the function is to seek the problematic
    columns out
    '''
    find_zero=df.columns
    a=[]
    for value in range(len(df.columns)):
        a.append(any(df.iloc[:,value]==0))
    print(a)
    number = 0 
    for i in a:
        if i is False:
            number += 1
        else:
            break
    else:
        print('Nothing Bad!!')
    return find_zero[number]

find_unusual_value()


'''Replace the 0 value'''
sum(df['Nikkei']==0)
df[df['Nikkei']==0]
df.loc['2017-09-07',:]['Nikkei']#19430.0
df.loc['2017-09-08',:]['Nikkei']#19430.0
df[df['Nikkei']==0] = df[df['Nikkei']==0].replace(0,19430.0)
df.loc['2017-09-07':'2017-09-10',:]['Nikkei']#19430.0

'''Select different time period'''
df.info()

'''Data Visualization'''
plt.figure(figsize=(40, 30))
correlation = df.corr()
mask = np.zeros_like(correlation)
mask[np.triu_indices_from(mask)] = True
sns.set(font_scale = 2)
sns.heatmap(correlation,annot=True,annot_kws={"size": 14},mask=mask)

'''Feature Scaling'''
scaler = StandardScaler()
standard_df = scaler.fit_transform(df.iloc[:,:])

''' function'''
pca_hedge(standard_df)
PCA_importance()
pca_table = select_product()

'''PCA: Hedging Start'''
def pca_hedge(standard_df):
    global pca
    pca = PCA()
    standard_df = pca.fit_transform(standard_df)
    index_names = df.iloc[:,:len(df.columns)].columns
    col_names = ['PC' + str(x) for x in range(1,len(df.columns)+1)]
    PCA_Loadings = DataFrame(pca.components_.T, index=index_names, columns=col_names)
    PCA_Loadings = PCA_Loadings.round(3)
    print([round(float(i),4) for i in pca.explained_variance_])#like the eigen value
    print([round(float(i),4) for i in pca.explained_variance_ratio_])


#The following code constructs the Screen plot
def PCA_importance():
    each_var = np.round(pca.explained_variance_ratio_* 100,2)
    labels = ['PC' + str(x) for x in range(1,len(df.columns)+1)]
    plt.figure(figsize=(15,11))
    plt.bar(range(1,len(each_var)+1),height=each_var, tick_label=labels,alpha=0.7)
    plt.ylabel('Percentage of Explained Variance %')
    plt.xlabel('Principal Component')
    plt.xticks(rotation=60)
    plt.titles('Screen Plot')

            
def select_product():
    each_ration = np.round(pca.explained_variance_ratio_* 100,2)
    pca_table = pd.DataFrame(pca.components_.T, 
                             columns=['PC' + str(x) for x in range(1,len(df.columns)+1)], 
                             index=df.columns)
#    for i in range(len(pca_table.columns)):
#        pca_table.iloc[:,i] = pca_table.iloc[:,i]*each_ration[i]
    pca_table = round(pca_table, 3)

    return pca_table











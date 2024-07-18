# shifted rets t+1
import numpy as np
import pandas as pd
import os
import time
import gc
import pickle
import warnings
from sklearn.model_selection import ParameterGrid
from sklearn.linear_model import LinearRegression, HuberRegressor, ElasticNet
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
import lightgbm as lgb
from lightgbm import LGBMRegressor



warnings.filterwarnings('ignore')
os.chdir()


################################################################################
# Metrics  and validation function
################################################################################

def save_variable(variable, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(variable, file)

# out-of-sample R squared
def R_oos(actual, predicted):
    actual, predicted = np.array(actual).flatten(), np.array(predicted).flatten()
    #predicted = np.where(predicted<0,0,predicted)
    return 1 - (np.dot((actual-predicted),(actual-predicted)))/(np.dot(actual,actual))


## validation function to find the best model
def val_fun(model, params: dict, X_trn, y_trn, X_vld, y_vld):
    best_ros = None
    lst_params = list(ParameterGrid(params))
    for param in lst_params:
        if best_ros == None:
            mod = model().set_params(**param).fit(X_trn, y_trn)
            best_mod = mod
            y_pred = mod.predict(X_vld)
            best_ros = R_oos(y_vld, y_pred)
            best_param = param
        else:
            mod = model().set_params(**param).fit(X_trn, y_trn)
            y_pred = mod.predict(X_vld)
            ros = R_oos(y_vld, y_pred)
            if ros > best_ros:
                best_ros = ros
                best_mod = mod
                best_param = param
    return best_mod




################################################################################
# Linear Models
################################################################################
## OLS with expanding window
def expanding_OLS(data,fts,tgt,yrs_trn, version=None):
    X = data[fts]
    y = data[[tgt]]
    y_pred = []
    OLS_dump = {}
    OLS_sub_dump = {}
    
    # deal with dates
    all_dates = pd.Series(data.index).drop_duplicates().sort_values().reset_index(drop=True)
    
    # train test split and prediction
    for i in np.arange(12*yrs_trn-1,len(all_dates[:-12]),12):
        nddt_trn = all_dates[i]
        nddt_tst = all_dates[i+12]
        X_trn, y_trn = X[X.index<=nddt_trn], y[y.index<=nddt_trn]
        X_tst = X[(X.index>nddt_trn)&(X.index<=nddt_tst)]
        mod = LinearRegression().fit(X_trn,y_trn)
        y_pred += list(mod.predict(X_tst).flatten())
        year = nddt_tst.year
        
        if version == 'subsample':
            OLS_sub_dump[f'OLS_sub_{year}'] = mod
        else:
            OLS_dump[f'OLS_{year}'] = mod
    
    if version == 'subsample':
        save_variable(OLS_sub_dump, 'collection_OLS_subs')
    else:
        save_variable(OLS_dump, 'collection_OLSs')
            
    return y_pred


## ENet with expanding window
def expanding_ENet(data,fts,tgt,yrs_trn,yrs_vld,params):
    X = data[fts]
    y = data[[tgt]]
    y_pred = []
    ENet_dump = {}
    
    # deal with dates
    all_dates = pd.Series(data.index).drop_duplicates().sort_values().reset_index(drop=True)
    
    # train validate test split and prediction
    for i in np.arange(12*yrs_trn-1,len(all_dates[:-12*(yrs_vld+1)]),12):
        nddt_trn = all_dates[i]
        nddt_vld = all_dates[i+yrs_vld*12]
        nddt_tst = all_dates[i+(yrs_vld+1)*12]
        X_trn, y_trn = X[X.index<=nddt_trn], y[y.index<=nddt_trn]
        X_vld, y_vld = X[(X.index>nddt_trn)&(X.index<=nddt_vld)], y[(y.index>nddt_trn)&(y.index<=nddt_vld)]
        X_tst = X[(X.index>nddt_vld)&(X.index<=nddt_tst)]
        mod = val_fun(ElasticNet, params, X_trn, y_trn, X_vld, y_vld)
        y_pred += list(mod.predict(X_tst).flatten())
        year = nddt_tst.year
        ENet_dump[f'ENet_{year}'] = mod
    save_variable(ENet_dump, 'collection_ENets')

    return y_pred


## PLS with expanding window
def EXP_PLS(data,fts,tgt,yrs_trn,yrs_vld,params):
    X = data[fts]
    y = data[[tgt]]
    y_pred = []
    model_dump = {}
    
    # deal with dates
    all_dates = pd.Series(data.index).drop_duplicates().sort_values().reset_index(drop=True)
    
    # train validate test split and prediction
    for i in np.arange(12*yrs_trn-1,len(all_dates[:-12*(yrs_vld+1)]),12):
        nddt_trn = all_dates[i]
        nddt_vld = all_dates[i+yrs_vld*12]
        nddt_tst = all_dates[i+(yrs_vld+1)*12]
        X_trn, y_trn = X[X.index<=nddt_trn], y[y.index<=nddt_trn]
        X_vld, y_vld = X[(X.index>nddt_trn)&(X.index<=nddt_vld)], y[(y.index>nddt_trn)&(y.index<=nddt_vld)]
        X_tst = X[(X.index>nddt_vld)&(X.index<=nddt_tst)]
        mod = val_fun(PLSRegression, params, X_trn, y_trn, X_vld, y_vld)
        y_pred += list(mod.predict(X_tst).flatten())
        year = nddt_tst.year
        model_dump[f'pls_{year}'] = mod
    save_variable(model_dump, 'collection_PLSs')
        
    return y_pred



## PCR with expanding window
def EXP_PCR(data,fts,tgt,yrs_trn,yrs_vld,params):
    X = data[fts]
    y = data[[tgt]]
    y_pred = []
    model_dump = {}
    
    # deal with dates
    all_dates = pd.Series(data.index).drop_duplicates().sort_values().reset_index(drop=True)
    
    # train validate test split and prediction
    for i in np.arange(12*yrs_trn-1,len(all_dates[:-12*(yrs_vld+1)]),12):
        nddt_trn = all_dates[i]
        nddt_vld = all_dates[i+yrs_vld*12]
        nddt_tst = all_dates[i+(yrs_vld+1)*12]
        X_trn, y_trn = X[X.index<=nddt_trn], y[y.index<=nddt_trn]
        X_vld, y_vld = X[(X.index>nddt_trn)&(X.index<=nddt_vld)], y[(y.index>nddt_trn)&(y.index<=nddt_vld)]
        X_tst = X[(X.index>nddt_vld)&(X.index<=nddt_tst)]
        mod = val_fun(PCRegressor, params, X_trn, y_trn, X_vld, y_vld)
        y_pred += list(mod.predict(X_tst).flatten())
        year = nddt_tst.year
        model_dump[f'pcr_{year}'] = mod
    save_variable(model_dump, 'collection_PCRs')
        
    return y_pred



## PCR class
class PCRegressor:

    def __init__(self, n_PCs=1, loss='mse'):
        self.n_PCs = n_PCs
        if loss not in ['huber', 'mse']:
            raise AttributeError(
                f"The loss should be either 'huber' or 'mse', but {loss} is given"
            )
        else:
            self.loss = loss

    def set_params(self, **params):
        for param in params.keys():
            setattr(self, param, params[param])
        return self

    def fit(self, X, y):
        X = np.array(X, dtype=np.float32)
        N, K = X.shape
        y = np.array(y, dtype=np.float32).flatten()
        
        # Temporarily convert to float64 for mean and std calculation
        X_64 = X.astype(np.float64)
        self.mu = np.mean(X_64, axis=0).astype(np.float32).reshape((1, K))
        self.sigma = np.std(X_64, axis=0).astype(np.float32).reshape((1, K))

        # Check for NaNs or infinite values
        if np.any(np.isnan(self.sigma)) or np.any(np.isinf(self.sigma)):
            raise ValueError("Standard deviation calculation resulted in NaN or infinite values.")
        
        self.sigma = np.where(self.sigma == 0, 1, self.sigma)
        X = (X - self.mu) / self.sigma
        pca = PCA(n_components=self.n_PCs)
        X_pca = pca.fit_transform(X)
        self.Vk = pca.components_.T
        
        if self.loss == 'mse':
            self.model = LinearRegression().fit(X_pca, y)
        else:
            self.model = HuberRegressor().fit(X_pca, y)
        
        self.beta = self.model.coef_
        self.intercept_ = self.model.intercept_
        return self

    def predict(self, X):
        X = np.array(X, dtype=np.float32)
        X = (X - self.mu) / self.sigma
        X_pca = X @ self.Vk
        pred = X_pca @ self.beta + self.intercept_
        return pred


################################################################################
# Tree-Based Models
################################################################################
## Gradient Boosting
def expanding_GBM(data, fts, tgt, yrs_trn, yrs_vld, params):
    X = data[fts]
    y = data[[tgt]]
    y_pred = []
    model_dump = {}
    
    # deal with dates
    all_dates = pd.Series(data.index).drop_duplicates().sort_values().reset_index(drop=True)
    
    # train validate test split and prediction
    for i in np.arange(12*yrs_trn-1,len(all_dates[:-12*(yrs_vld+1)]),12):
        nddt_trn = all_dates[i]
        nddt_vld = all_dates[i+yrs_vld*12]
        nddt_tst = all_dates[i+(yrs_vld+1)*12]
        X_trn, y_trn = X[X.index<=nddt_trn], y[y.index<=nddt_trn]
        X_vld, y_vld = X[(X.index>nddt_trn)&(X.index<=nddt_vld)], y[(y.index>nddt_trn)&(y.index<=nddt_vld)]
        X_tst = X[(X.index>nddt_vld)&(X.index<=nddt_tst)]
        mod = val_fun(LGBMRegressor, params, X_trn, y_trn, X_vld, y_vld)
        y_pred += list(mod.predict(X_tst).flatten())
        year = nddt_tst.year
        model_dump[f'gbm_{year}'] = mod
    save_variable(model_dump, 'collection_GBMs')
    
    return y_pred


## Random Forest
def expanding_RF(data, fts, tgt, yrs_trn, yrs_vld, params):
    X = data[fts]
    y = data[[tgt]]
    y_pred = []
    model_dump = {}

    # deal with dates
    all_dates = pd.Series(data.index).drop_duplicates().sort_values().reset_index(drop=True)

    # train validate test split and prediction
    for i in np.arange(12*yrs_trn-1,len(all_dates[:-12*(yrs_vld+1)]),12):
        nddt_trn = all_dates[i]
        nddt_vld = all_dates[i+yrs_vld*12]
        nddt_tst = all_dates[i+(yrs_vld+1)*12]
        X_trn, y_trn = X[X.index<=nddt_trn], y[y.index<=nddt_trn]
        X_vld, y_vld = X[(X.index>nddt_trn)&(X.index<=nddt_vld)], y[(y.index>nddt_trn)&(y.index<=nddt_vld)]
        X_tst = X[(X.index>nddt_vld)&(X.index<=nddt_tst)]
        mod = val_fun(LGBMRegressor, params, X_trn, y_trn, X_vld, y_vld)
        y_pred += list(mod.predict(X_tst).flatten())
        year = nddt_tst.year
        model_dump[f'RF_{year}'] = mod
    save_variable(model_dump, 'collection_RFs')

    return y_pred





################################################################################
## lets start training
################################################################################
stdt, nddt = 19570101, 20191231
yrs_trn, yrs_vld = 18, 12
stdt_tst = str(stdt + (yrs_trn+yrs_vld)*10000)
stdt_tst = np.datetime64(stdt_tst[:4]+'-'+stdt_tst[4:6]+'-'+stdt_tst[6:])


## Top 1000 data
data = pd.read_pickle()
features = list(set(data.columns).difference({'permno','DATE','RET'}))
## df to store true and predicted returns for later on
res = data[['permno','RET']][data.index>=stdt_tst].copy()


### Linear Models ###
## OLS
print('OLS starts...')
st = time.time()
res['OLS_PRED'] = expanding_OLS(data,features,'RET',yrs_trn+yrs_vld)
## OLS with preselected size, bm, and momentum covariates
features_3 = ['mvel1','bm','mom1m']
res['OLS_3_PRED'] = expanding_OLS(data,features_3,'RET',yrs_trn+yrs_vld, 'subsample')
elapsed_time = time.time() - st
print(f'Both OLS finished!!\n Execution time: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')
gc.collect()
res.to_csv('results_5000.csv')

## ENet
print('ENet starts...')
st = time.time()
params_ENet = {'alpha':list(np.linspace(1e-4,1e-1,50))}
res['ENet_PRED'] = expanding_ENet(data,features,'RET',yrs_trn,yrs_vld,params_ENet)
elapsed_time = time.time() - st
print(f'ENet finished!!\n Execution time: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')
gc.collect()
res.to_csv('results_5000.csv')


## PLS
st = time.time()
params_PLS = {'n_components': [1, 5, 10, 50]}
res['PLS_PRED'] = EXP_PLS(data,features,'RET',yrs_trn,yrs_vld,params_PLS)
elapsed_time = time.time() - st
print(f'PLS finished!!!!!\n Execution time: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')
gc.collect()
res.to_csv('results_5000.csv')


## PCR
st = time.time()
params_PCR = {'n_PCs':[1,3,5,7,10,50],'loss':['mse','huber']}
res['PCR_PRED'] = EXP_PCR(data,features,'RET',yrs_trn,yrs_vld,params_PCR)
elapsed_time = time.time() - st
print(f'PCR finished!!!!!\n Execution time: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')
gc.collect()
res.to_csv('results_5000.csv')


### Tree-Based Models ###
## Gradient Boost Hist
print('GBM  starts...')
st = time.time()
params_GBM = {
    'objective':[None],
    'max_depth':[1,2],
    'n_estimators':[20,50,100,200, 300, 400, 500, 600, 1000],
    'random_state':[12308],
    'learning_rate':[.01,.1],
    'verbosity': [-1]
}
res['GBM_PRED'] = expanding_GBM(data,features,'RET',yrs_trn,yrs_vld,params_GBM)
gc.collect()
elapsed_time = time.time() - st
print(f'LGBM normal finished!!\n Execution time: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')
gc.collect()
res.to_csv('results_5000.csv')



## Random Forest LGBM package
print('RF starts training...')
st = time.time()
params_RF = {
    'n_estimators': [300],
    'max_depth': list(np.arange(1,7)),
    'feature_fraction': [0.1, 0.15, 0.2, 0.3, 0.5, 0.6, 0.8],
    'bagging_fraction': [0.6],
    'bagging_freq': [1],
    'random_state': [12308],
    'boosting_type': ['rf'],
    'verbosity': [-1]
}
res['RF_PRED'] = expanding_RF(data,features,'RET',yrs_trn,yrs_vld,params_RF)
gc.collect()
elapsed_time = time.time() - st
print(f'RF finished!!\\n Execution time: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')
gc.collect()
res.to_csv('results_5000.csv')


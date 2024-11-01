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
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, Dense, BatchNormalization
from keras.regularizers import L1L2
from keras.optimizers import Adam

warnings.filterwarnings('ignore')


################################################################################
# Definin our metrics  and validation function
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
# First, plain vanilla linear models
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
# Neural Networks
################################################################################

## NN with expanding window
def expanding_NN(data,fts,tgt,yrs_trn,yrs_vld,params, version):
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
        mod = val_fun(NN, params, X_trn, y_trn, X_vld, y_vld)
        y_pred += list(mod.predict(X_tst).flatten())
        year = nddt_tst.year
        model_dump[f'NNet_{version}_{year}'] = mod
    save_variable(model_dump, f'collection_NNets_{version}')
        
        
    return y_pred




# customized metrics
# out-of-sample r squared for keras
def R_oos_tf(y_true, y_pred):
    resid = tf.square(y_true-y_pred)
    denom = tf.square(y_true)
    return 1 - tf.divide(tf.reduce_mean(resid),tf.reduce_mean(denom))

# data standardization
# please standardize the data if BatchNormalization is not used
def standardize(X_trn, X_vld, X_tst):
    mu_trn = np.mean(np.array(X_trn),axis=0).reshape((1,X_trn.shape[1]))
    sigma_trn = np.std(np.array(X_trn),axis=0).reshape((1,X_trn.shape[1]))

    X_trn_std = (np.array(X_trn)-mu_trn)/sigma_trn
    X_vld_std = (np.array(X_vld)-mu_trn)/sigma_trn
    X_tst_std = (np.array(X_tst)-mu_trn)/sigma_trn
    return X_trn_std, X_vld_std, X_tst_std

# NN class
class NN:
    
    def __init__(
        self, n_layers=1, loss='mse', l1=1e-5, l2=0, learning_rate=.01, BatchNormalization=True, patience=5,
        epochs=100, batch_size=3000, verbose=1, random_state=12308, monitor='val_R_oos_tf', base_neurons=5
    ):
        self.n_layers = n_layers
        self.l1 = l1
        self.l2 = l2
        self.learning_rate = learning_rate
        self.BatchNormalization = BatchNormalization
        self.patience = patience
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.random_state = random_state
        self.monitor = monitor
        self.base_neurons = base_neurons

    def set_params(self, **params):
        for param in params.keys():
            setattr(self, param, params[param])
        return self
    
    def fit(self, X_trn, y_trn, X_vld, y_vld):
        # fix random seed for reproducibility
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        tf.random.set_seed(self.random_state)
        
        # model construction
        mod = Sequential()
        mod.add(Input(shape=(X_trn.shape[1],)))
        
        for i in np.arange(self.n_layers,0,-1):
            if self.n_layers>self.base_neurons:
                if self.n_layers == i:
                    mod.add(Dense(2**i, activation='relu'))
                else:
                    mod.add(Dense(2**i, activation='relu', kernel_regularizer=L1L2(self.l1,self.l2)))
            else:
                if self.n_layers == i:
                    mod.add(Dense(2**(self.base_neurons-(self.n_layers-i)), activation='relu'))
                else:
                    mod.add(Dense(2**(self.base_neurons-(self.n_layers-i)), 
                                  activation='relu', kernel_regularizer=L1L2(self.l1,self.l2)))
            if self.BatchNormalization:
                mod.add(BatchNormalization())
        
        mod.add(Dense(1, kernel_regularizer=L1L2(self.l1,self.l2)))
        
        # early stopping; i insert the mechanism to maximize the respective objective: max mode
        earlystop = tf.keras.callbacks.EarlyStopping(monitor=self.monitor, mode = 'max', patience=self.patience)

        # Adam solver
        opt = Adam(learning_rate=self.learning_rate)
        
        # compile the model
        mod.compile(loss=self.loss,
                    optimizer=opt,
                    metrics=[R_oos_tf])

        # fit the model
        mod.fit(X_trn, np.array(y_trn).reshape((len(y_trn),1)), epochs=self.epochs, batch_size=self.batch_size, 
                callbacks=[earlystop], verbose=self.verbose, 
                validation_data=(X_vld,np.array(y_vld).reshape((len(y_vld),1))))
        
        self.model = mod
        return self
    
    def predict(self, X):
        return self.model.predict(X, verbose=self.verbose)




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



# i have to convert booleans to numeric dummies to feed into the the neural net
for col in data.select_dtypes(include=['bool']).columns:
    data[col] = data[col].astype('float32')

### Neural Networks ###
## NN1
print('NNet1 starts training...')
st = time.time()
params_NN1 = {
    'n_layers': [1],
    'loss': ['mse'],
    'l1': [1e-5,1e-3],
    'learning_rate': [.001,.01],
    'batch_size': [10000],
    'epochs': [100],
    'random_state': [12308],
    'BatchNormalization': [True],
    'patience':[5],
    'verbose': [0],
    'monitor':['val_loss','val_R_oos_tf']
}
res['NN1_PRED'] = expanding_NN(data,features,'RET',yrs_trn,yrs_vld,params_NN1, 'layer1')
gc.collect()
elapsed_time = time.time() - st
print(f'NN1 finished!\n Execution time: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')


## NN2
print('NNet2 starts training...')
st = time.time()
params_NN2 = {
    'n_layers': [2],
    'loss': ['mse'],
    'l1': [1e-5,1e-3],
    'learning_rate': [.001,.01],
    'batch_size': [10000],
    'epochs': [100],
    'random_state': [12308],
    'BatchNormalization': [True],
    'patience':[5],
    'verbose': [0],
    'monitor':['val_loss','val_R_oos_tf']
}
res['NN2_PRED'] = expanding_NN(data,features,'RET',yrs_trn,yrs_vld,params_NN2, 'layer2')
gc.collect()
elapsed_time = time.time() - st
print(f'NN2 finished!\n Execution time: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')



## NN3
print('NNet3 starts training...')
st = time.time()
params_NN3 = {
    'n_layers': [3],
    'loss': ['mse'],
    'l1': [1e-5,1e-3],
    'learning_rate': [.001,.01],
    'batch_size': [10000],
    'epochs': [100],
    'random_state': [12308],
    'BatchNormalization': [True],
    'patience':[5],
    'verbose': [0],
    'monitor':['val_loss','val_R_oos_tf']
}
res['NN3_PRED'] = expanding_NN(data,features,'RET',yrs_trn,yrs_vld,params_NN3, 'layer3')
gc.collect()
elapsed_time = time.time() - st
print(f'NN3 finished!\n Execution time: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')



## NN4
print('NNet4 starts training...')
st = time.time()
params_NN4 = {
    'n_layers': [4],
    'loss': ['mse'],
    'l1': [1e-5,1e-3],
    'learning_rate': [.001,.01],
    'batch_size': [10000],
    'epochs': [100],
    'random_state': [12308],
    'BatchNormalization': [True],
    'patience':[5],
    'verbose': [0],
    'monitor':['val_loss','val_R_oos_tf']
}
res['NN4_PRED'] = expanding_NN(data,features,'RET',yrs_trn,yrs_vld,params_NN4, 'layer4')
gc.collect()
elapsed_time = time.time() - st
print(f'NN4 finished!\n Execution time: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')


## NN5
print('NNet5 starts training...')
st = time.time()
params_NN5 = {
    'n_layers': [5],
    'loss': ['mse'],
    'l1': [1e-5,1e-3],
    'learning_rate': [.001,.01],
    'batch_size': [10000],
    'epochs': [100],
    'random_state': [12308],
    'BatchNormalization': [True],
    'patience':[5],
    'verbose': [0],
    'monitor':['val_loss','val_R_oos_tf']
}
res['NN5_PRED'] = expanding_NN(data,features,'RET',yrs_trn,yrs_vld,params_NN5, 'layer5')
gc.collect()
elapsed_time = time.time() - st
print(f'NN5 finished!\n Execution time: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')
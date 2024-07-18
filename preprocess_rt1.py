import pandas as pd
import numpy as np
from sklearn.preprocessing import minmax_scale
import os
import time

os.chdir()

st = time.time()
data_ch = pd.read_pickle()
data_ch = data_ch[(data_ch['DATE']>=19570101)&(data_ch['DATE']<=20201231)].reset_index(drop=True)
data_ch['DATE'] = pd.to_datetime(data_ch['DATE'],format='%Y%m%d')+pd.offsets.MonthEnd(0)

characteristics = list(set(data_ch.columns).difference({'permno','DATE','SHROUT','mve0','sic2','RET','prc'}))

# filling missing values;
for ch in characteristics:
    data_ch[ch] = data_ch.groupby('DATE')[ch].transform(lambda x: x.fillna(x.median()))
for ch in characteristics:
    data_ch[ch] = data_ch[ch].fillna(0)

# minmax scaling to have standardized values [-1,1]
for ch in characteristics:
    data_ch[ch] = data_ch.groupby('DATE')[ch].transform(lambda x: minmax_scale(np.array(x).reshape((len(x),1)),(-1,1)).flatten())


# dummy variables for sic code
sic_dummies = pd.get_dummies(data_ch['sic2'].fillna(999).astype(int),prefix='sic').drop('sic_999',axis=1)
data_ch = pd.concat([data_ch,sic_dummies],axis=1)
data_ch.drop(['prc','SHROUT','mve0','sic2'],inplace=True,axis=1)

data_ch = data_ch.set_index('DATE')
data_ch = data_ch.sort_values(['DATE', 'permno'])

# since we want to run our models on rets at time t+1 i shift the column RET up and avoid building the shift in in the modelling code due to oveview reasons and syntax with used packages
data_ch['RET'] = data_ch.groupby('permno')['RET'].shift(-1)

# transform to float32
data_ch.dtypes.value_counts()
for col in data_ch.select_dtypes(include=['float64']).columns:
    data_ch[col] = data_ch[col].astype('float32')


# drop 2020 since not a whole year anymore:
data_ch = data_ch[~data_ch.index.year.isin([2020])]

# set nan as 0 since it means that no longer in 
num_missing_ret = data_ch['RET'].isna().sum()
num_missing_ret
data_ch['RET'] = data_ch['RET'].fillna(0)



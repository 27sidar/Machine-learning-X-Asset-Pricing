import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import lightgbm as lgb
import gc
import seaborn as sns
from sklearn.linear_model import LinearRegression, HuberRegressor
import os
import matplotlib.lines as mlines


os.chdir()
data = pd.read_pickle()
features = list(set(data.columns).difference({'permno','DATE','RET'}))

del data
gc.collect()


##### Discliamer
# for the sake of overview i avoided nested for loops which would iterate the analysis or plots over my multiple results.
# I describe each analysis and plot on its own to ensure the reader an easy line to follow, even without prior knowledge.



# define our functions for later
def load_variable(file_path):
    with open(file_path, 'rb') as file:
        loaded_variable = pickle.load(file)
    return loaded_variable

def R_oos(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted).flatten()
    predicted = np.where(predicted<0,0,predicted)
    return 1 - (np.dot((actual-predicted),(actual-predicted)))/(np.dot(actual,actual))




### R^2 oos (monthly vs annual)

# top 1000
results_1000 = pd.read_csv('results rt1 1000/results_1000.csv')
results_1000['DATE'] = pd.to_datetime(results_1000['DATE'])
results_1000.set_index('DATE', inplace = True)
results_1000.sort_index(inplace = True)

results_pcrpls = pd.read_csv('results rt1 1000/results_pcrpls.csv')
results_pcrpls['DATE'] = pd.to_datetime(results_pcrpls['DATE'])
results_pcrpls.set_index('DATE', inplace = True)
results_pcrpls.sort_index(inplace = True)
results_pcrpls

results_1000[['PCR_PRED', 'PLS_PRED']] = results_pcrpls[['PCR_PRED', 'PLS_PRED']]

models1 = list(set(results_1000.columns).difference({'DATE','RET', 'permno'}))
models1

month_roos_dump1 = {}

for model in models1:
    r_oos_value1 = R_oos(results_1000['RET'], results_1000[model]) * 100
    month_roos_dump1[model] = [r_oos_value1]
    
monthly_roos1 = pd.DataFrame(month_roos_dump1, index = ['Monthly R^2_oos 1000'])
monthly_roos1


# annual
annual_results1 = results_1000.groupby([results_1000.index.year, 'permno']).sum()
annual_results1

ann_roos_dump1 = {}

for model in models1:
    r_oos_value1 = R_oos(annual_results1['RET'], annual_results1[model]) * 100
    ann_roos_dump1[model] = [r_oos_value1]

annual_roos1 = pd.DataFrame(ann_roos_dump1, index = ['Annual R^2_oos 1000'])
annual_roos1




# top 5000
results_5000 = pd.read_csv('results rt1 5000/results_5000.csv')
results_5000['DATE'] = pd.to_datetime(results_5000['DATE'])
results_5000.set_index('DATE', inplace = True)
results_5000.sort_index(inplace = True)
models5 = list(set(results_5000.columns).difference({'DATE','RET', 'permno'}))

models5
month_roos_dump5 = {}

for model in models5:
    r_oos_value5 = R_oos(results_5000['RET'], results_5000[model]) * 100
    month_roos_dump5[model] = [r_oos_value5]
    
monthly_roos5 = pd.DataFrame(month_roos_dump5, index = ['Monthly R^2_oos 5000'])
monthly_roos5


# annual
annual_results5 = results_5000.groupby([results_5000.index.year, 'permno']).sum()
annual_results5

ann_roos_dump5 = {}

for model in models5:
    r_oos_value5 = R_oos(annual_results5['RET'], annual_results5[model]) * 100
    ann_roos_dump5[model] = [r_oos_value5]

annual_roos5 = pd.DataFrame(ann_roos_dump5, index = ['Annual R^2_oos 5000'])
annual_roos5


# append together for combined charts later on
monthly_roos = pd.concat([monthly_roos1, monthly_roos5], axis = 0)
annual_roos = pd.concat([annual_roos1, annual_roos5], axis = 0)







### Model complexity over time
## load in all the trained models

# trained on top 1000 firms
dump_ENets_1000 = load_variable('results rt1 1000/collection_ENets')
dump_OLS_1000 = load_variable('results rt1 1000/collection_OLSs')
dump_OLS3_1000 = load_variable('results rt1 1000/collection_OLS_subs')
dump_gbm_1000 = load_variable('results rt1 1000/collection_GBMs')
dump_rf_1000 = load_variable('results rt1 1000/collection_RFs')

# trained on top 5000 firms
dump_ENets_5000 = load_variable('results rt1 5000/collection_ENets')
dump_OLS_5000 = load_variable('results rt1 5000/collection_OLSs')
dump_OLS3_5000 = load_variable('results rt1 5000/collection_OLS_subs')
dump_gbm_5000 = load_variable('results rt1 5000/collection_GBMs')
dump_rf_5000 = load_variable('results rt1 5000/collection_RFs')








# ENet
# the authors report the number of features selected to have nonzero coefficients
# 1000
model_complexity_ENet1 = []
for model_name, model in dump_ENets_1000.items():
  year = int(model_name.split("_")[-1])
  complexity = np.count_nonzero(model.coef_)
  model_complexity_ENet1.append((year, complexity))
# sort by date
model_complexity_ENet1.sort(key=lambda x: x[0])
years = [data[0] for data in model_complexity_ENet1]
complexity_values1 = [data[1] for data in model_complexity_ENet1]

# 5000
model_complexity_ENet5 = []
for model_name, model in dump_ENets_5000.items():
  year = int(model_name.split("_")[-1])
  complexity = np.count_nonzero(model.coef_)
  model_complexity_ENet5.append((year, complexity))
# sort by date
model_complexity_ENet5.sort(key=lambda x: x[0])
years = [data[0] for data in model_complexity_ENet5]
complexity_values5 = [data[1] for data in model_complexity_ENet5]

#plot
sns.set_style("whitegrid")
fig, ax = plt.subplots(dpi = 300)
# Plot the data with specific line styles and markers
ax.plot(years, complexity_values1, marker='', markersize=4, linestyle='-', color='steelblue', linewidth=2, label='top 1000')
ax.plot(years, complexity_values5, marker='', linestyle='-', color='darkkhaki', linewidth=2, label='top 5000')
# Set the title and labels
ax.set_title('ElasticNet', fontsize=14, fontweight='bold')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('# of characteristics', fontsize=12)
ax.legend()
plt.show()




# gbm
# authors report the # of distinct characteristics entering into the trees; here i extract the number of trees / estimators at each refit
# 1000
model_complexity_gbm1 = []
for model_name, model in dump_gbm_1000.items():
  year = int(model_name.split("_")[-1])
  complexity1 = model.n_estimators_
  model_complexity_gbm1.append((year, complexity1))
# sort by date
model_complexity_gbm1.sort(key=lambda x: x[0])
years = [data[0] for data in model_complexity_gbm1]
complexity_values1 = [data[1] for data in model_complexity_gbm1]

# 5000
model_complexity_gbm5 = []
for model_name, model in dump_gbm_5000.items():
  year = int(model_name.split("_")[-1])
  complexity5 = model.n_estimators_
  model_complexity_gbm5.append((year, complexity5))
# sort by date
model_complexity_gbm5.sort(key=lambda x: x[0])
years = [data[0] for data in model_complexity_gbm5]
complexity_values5 = [data[1] for data in model_complexity_gbm5]

#plot
sns.set_style("whitegrid")
fig, ax = plt.subplots(dpi = 300)
# Plot the data with specific line styles and markers
ax.plot(years, complexity_values1, marker='', markersize=4, linestyle='-', color='steelblue', linewidth=2, label='top 1000')
ax.plot(years, complexity_values5, marker='', linestyle='-', color='darkkhaki', linewidth=2, label='top 5000')
# Set the title and labels
ax.set_title('Gradient Boosted Regression Tree', fontsize=14, fontweight='bold')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('# of distinct char.', fontsize=12)
ax.legend()
plt.show()






# RF
# authors report the tree depths over time of RF
# 1000
model_complexity_rf1 = []
for model_name, model in dump_rf_1000.items():
  year = int(model_name.split("_")[-1])
  complexity1 = model.max_depth
  model_complexity_rf1.append((year, complexity1))
# sort by date
model_complexity_rf1.sort(key=lambda x: x[0])
years = [data[0] for data in model_complexity_rf1]
complexity_values1 = [data[1] for data in model_complexity_rf1]

# 5000
model_complexity_rf5 = []
for model_name, model in dump_rf_5000.items():
  year = int(model_name.split("_")[-1])
  complexity5 = model.max_depth
  model_complexity_rf5.append((year, complexity5))
# sort by date
model_complexity_rf5.sort(key=lambda x: x[0])
years = [data[0] for data in model_complexity_rf5]
complexity_values5 = [data[1] for data in model_complexity_rf5]

#plot
sns.set_style("whitegrid")
fig, ax = plt.subplots(dpi = 300)
# Plot the data with specific line styles and markers
ax.plot(years, complexity_values1, marker='', markersize=4, linestyle='-', color='steelblue', linewidth=2, label='top 1000')
ax.plot(years, complexity_values5, marker='', linestyle='-', color='darkkhaki', linewidth=2, label='top 5000')
# Set the title and labels
ax.set_title('Random Forest', fontsize=14, fontweight='bold')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('max. Tree depth', fontsize=12)
ax.legend()
plt.show()


# RF fractions
# 1000
model_complexity_rf1 = []
for model_name, model in dump_rf_1000.items():
  year = int(model_name.split("_")[-1])
  complexity1 = model.feature_fraction
  model_complexity_rf1.append((year, complexity1))
# sort by date
model_complexity_rf1.sort(key=lambda x: x[0])
years = [data[0] for data in model_complexity_rf1]
complexity_values1 = [data[1] for data in model_complexity_rf1]

# 5000
model_complexity_rf5 = []
for model_name, model in dump_rf_5000.items():
  year = int(model_name.split("_")[-1])
  complexity5 = model.feature_fraction
  model_complexity_rf5.append((year, complexity5))
# sort by date
model_complexity_rf5.sort(key=lambda x: x[0])
years = [data[0] for data in model_complexity_rf5]
complexity_values5 = [data[1] for data in model_complexity_rf5]

#plot
sns.set_style("whitegrid")
fig, ax = plt.subplots(dpi = 300)
# Plot the data with specific line styles and markers
ax.plot(years, complexity_values1, marker='', markersize=4, linestyle='-', color='steelblue', linewidth=2, label='top 1000')
ax.plot(years, complexity_values5, marker='', linestyle='-', color='darkkhaki', linewidth=2, label='top 5000')
# Set the title and labels
ax.set_title('Random Forest', fontsize=14, fontweight='bold')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('chosen feature fraction', fontsize=12)
ax.legend()
plt.show()







### Variable importances
feature_names_original = features

# i created this function to aggreagate all indiviudal sic codes to an avg sic code predictor, to see how it affects predictions overall
def sicc(df):
    sic_mask = df['feature'].str.startswith('sic_')
    avg_sic_importance = df.loc[sic_mask, 'importance'].mean()
    df.loc[sic_mask, 'feature'] = 'sic_code'
    df.loc[sic_mask, 'importance'] = avg_sic_importance
    df = df.drop_duplicates().reset_index(drop=True)
    return df


## OLS
# 1000
all_coefs_OLS1 = []
for model in dump_OLS_1000.values():
  all_coefs_OLS1.append(model.coef_)

avg_coefs_OLS1 = np.abs(np.mean(all_coefs_OLS1, axis=0))
avg_coefs_OLS1 = avg_coefs_OLS1.flatten()
importance_df_OLS1 = pd.DataFrame({'feature': feature_names_original, 'importance': avg_coefs_OLS1})
#here i get the relative importance in pp
importance_df_OLS1['importance'] = importance_df_OLS1['importance'] / importance_df_OLS1['importance'].sum()
importance_df_OLS1 = sicc(importance_df_OLS1)
importance_df_OLS1p = importance_df_OLS1.sort_values('importance', ascending=False).head(15)
importance_df_OLS1p = importance_df_OLS1p.sort_values('importance', ascending=True)

plt.figure(figsize=(10, 9), dpi = 300)
plt.barh(importance_df_OLS1p['feature'], importance_df_OLS1p['importance'], color = 'steelblue')
plt.xlabel('Relative Importance')
plt.title('OLS Feature Importance top 1000')
plt.tight_layout()
plt.show()


# 5000
all_coefs_OLS5 = []
for model in dump_OLS_5000.values():
  all_coefs_OLS5.append(model.coef_)

avg_coefs_OLS5 = np.abs(np.mean(all_coefs_OLS5, axis=0))
avg_coefs_OLS5 = avg_coefs_OLS5.flatten()
importance_df_OLS5 = pd.DataFrame({'feature': feature_names_original, 'importance': avg_coefs_OLS5})
#here i get the relative importance in pp
importance_df_OLS5['importance'] = importance_df_OLS5['importance'] / importance_df_OLS5['importance'].sum()
importance_df_OLS5 = sicc(importance_df_OLS5)
importance_df_OLS5p = importance_df_OLS5.sort_values('importance', ascending=False).head(15)
importance_df_OLS5p = importance_df_OLS5p.sort_values('importance', ascending=True)

plt.figure(figsize=(10, 9))
plt.barh(importance_df_OLS5p['feature'], importance_df_OLS5p['importance'], color = 'darkkhaki')
plt.xlabel('Relative Importance')
plt.title('OLS Feature Importance top 5000')
plt.tight_layout()
plt.show()





## OLS with 3 variables 
features_3 = ['mvel1','bm','mom1m']
# 1000
all_coefs_OLS31 = []
for model in dump_OLS3_1000.values():
  all_coefs_OLS31.append(model.coef_)

avg_coefs_OLS31 = np.abs(np.mean(all_coefs_OLS31, axis=0))
avg_coefs_OLS31 = avg_coefs_OLS31.flatten()
importance_df_OLS31 = pd.DataFrame({'feature': features_3, 'importance': avg_coefs_OLS31})
#here i get the relative importance in pp
importance_df_OLS31['importance'] = importance_df_OLS31['importance'] / importance_df_OLS31['importance'].sum()
importance_df_OLS31 = importance_df_OLS31.sort_values('importance', ascending=False).head(15)
importance_df_OLS31 = importance_df_OLS31.sort_values('importance', ascending=True)

plt.figure(figsize=(10, 9))
plt.barh(importance_df_OLS31['feature'], importance_df_OLS31['importance'], color = 'steelblue')
plt.xlabel('Relative Importance')
plt.title('OLS3 Feature Importance top 1000')
plt.tight_layout()
plt.show()


# 5000
all_coefs_OLS35 = []
for model in dump_OLS3_5000.values():
  all_coefs_OLS35.append(model.coef_)

avg_coefs_OLS35 = np.abs(np.mean(all_coefs_OLS35, axis=0))
avg_coefs_OLS35 = avg_coefs_OLS35.flatten()
importance_df_OLS35 = pd.DataFrame({'feature': features_3, 'importance': avg_coefs_OLS35})
#here i get the relative importance in pp
importance_df_OLS35['importance'] = importance_df_OLS35['importance'] / importance_df_OLS35['importance'].sum()
importance_df_OLS35 = importance_df_OLS35.sort_values('importance', ascending=False).head(15)
importance_df_OLS35 = importance_df_OLS35.sort_values('importance', ascending=True)

plt.figure(figsize=(10, 9))
plt.barh(importance_df_OLS35['feature'], importance_df_OLS35['importance'], color = 'darkkhaki')
plt.xlabel('Relative Importance')
plt.title('OLS3 Feature Importance top 5000')
plt.tight_layout()
plt.show()







## Elastic Nets
# 1000
all_coefs_ENet1 = []
for model in dump_ENets_1000.values():
  all_coefs_ENet1.append(model.coef_)

avg_coefs_ENet1 = np.mean(all_coefs_ENet1, axis=0)
importance_df_ENet1 = pd.DataFrame({'feature': feature_names_original, 'importance': avg_coefs_ENet1**2 + np.abs(avg_coefs_ENet1)})
importance_df_ENet1['importance'] = importance_df_ENet1['importance'] / importance_df_ENet1['importance'].sum()
importance_df_ENet1 = sicc(importance_df_ENet1)
importance_df_ENet1p = importance_df_ENet1.sort_values('importance', ascending=False).head(15)
importance_df_ENet1p = importance_df_ENet1p.sort_values('importance', ascending=True)
# Plot
plt.figure(figsize=(10, 9), dpi = 300)
plt.barh(importance_df_ENet1p['feature'], importance_df_ENet1p['importance'], color = 'steelblue')
plt.xlabel('Relative Importance', fontsize = 15)
plt.tight_layout()
plt.title('Elastic Net importance top 1,000', fontsize=23, weight='bold')
plt.xticks(fontsize=15)  
plt.yticks(fontsize=15) 
plt.show()


# 5000
all_coefs_ENet5 = []
for model in dump_ENets_5000.values():
  all_coefs_ENet5.append(model.coef_)

avg_coefs_ENet5 = np.mean(all_coefs_ENet5, axis=0)
importance_df_ENet5 = pd.DataFrame({'feature': feature_names_original, 'importance': avg_coefs_ENet5**2 + np.abs(avg_coefs_ENet5)})
importance_df_ENet5['importance'] = importance_df_ENet5['importance'] / importance_df_ENet5['importance'].sum()
importance_df_ENet5 = sicc(importance_df_ENet5)
importance_df_ENet5p = importance_df_ENet5.sort_values('importance', ascending=False).head(15)
importance_df_ENet5p = importance_df_ENet5p.sort_values('importance', ascending=True)
# Plot
plt.figure(figsize=(10, 9), dpi = 300)
plt.barh(importance_df_ENet5p['feature'], importance_df_ENet5p['importance'], color = 'darkkhaki')
plt.xlabel('Relative Importance', fontsize = 15)
plt.title('Elastic Net importance top 5,000', fontsize=23, weight='bold')
plt.xticks(fontsize=15)  
plt.yticks(fontsize=15) 
plt.tight_layout()
plt.show()





## gbm
# 1000
all_importances_gbm1 = []
for model in dump_gbm_1000.values():
    all_importances_gbm1.append(model.feature_importances_)

avg_importance_gbm1 = np.mean(all_importances_gbm1, axis=0)
feature_names_gbm1 = dump_gbm_1000[list(dump_gbm_1000.keys())[0]].feature_name_
importance_df_gbm1 = pd.DataFrame({'feature': feature_names_gbm1, 'importance': avg_importance_gbm1})
# here i transform to get relative importance as in the paper
importance_df_gbm1['importance'] = importance_df_gbm1['importance'] / importance_df_gbm1['importance'].sum()
importance_df_gbm1 = sicc(importance_df_gbm1)
importance_df_gbm1p = importance_df_gbm1.sort_values('importance', ascending=False).head(15)
importance_df_gbm1p = importance_df_gbm1p.sort_values('importance', ascending=True)
# Plot
plt.figure(figsize=(10, 9), dpi = 300)
plt.barh(importance_df_gbm1p['feature'], importance_df_gbm1p['importance'], color = 'steelblue')
plt.xlabel('Relative Importance', fontsize = 15)
plt.title('Gradient Boost importance top 1,000', fontsize=23, weight='bold')
plt.xticks(fontsize=15)  
plt.yticks(fontsize=15) 
plt.tight_layout()
plt.show()


# 5000
all_importances_gbm5 = []
for model in dump_gbm_5000.values():
    all_importances_gbm5.append(model.feature_importances_)

avg_importance_gbm5 = np.mean(all_importances_gbm5, axis=0)
feature_names_gbm5 = dump_gbm_5000[list(dump_gbm_5000.keys())[0]].feature_name_
importance_df_gbm5 = pd.DataFrame({'feature': feature_names_gbm5, 'importance': avg_importance_gbm5})
# here i transform to get relative importance as in the paper
importance_df_gbm5['importance'] = importance_df_gbm5['importance'] / importance_df_gbm5['importance'].sum()
importance_df_gbm5 = sicc(importance_df_gbm5)
importance_df_gbm5p = importance_df_gbm5.sort_values('importance', ascending=False).head(15)
importance_df_gbm5p = importance_df_gbm5p.sort_values('importance', ascending=True)
# Plot
plt.figure(figsize=(10, 9), dpi = 300)
plt.barh(importance_df_gbm5p['feature'], importance_df_gbm5p['importance'], color = 'darkkhaki')
plt.xlabel('Relative Importance', fontsize = 15)
plt.title('Gradient Boost importance top 5,000', fontsize=23, weight='bold')
plt.xticks(fontsize=15)  
plt.yticks(fontsize=15) 
plt.tight_layout()
plt.show()




# RF
# 1000
all_importances_rf1 = []
for model in dump_rf_1000.values():
    all_importances_rf1.append(model.feature_importances_)

avg_importance_rf1 = np.mean(all_importances_rf1, axis=0)
feature_names_rf1 = dump_rf_1000[list(dump_rf_1000.keys())[0]].feature_name_
importance_df_rf1 = pd.DataFrame({'feature': feature_names_rf1, 'importance': avg_importance_rf1})
# here i transform to get relative importance as in the paper
importance_df_rf1['importance'] = importance_df_rf1['importance'] / importance_df_rf1['importance'].sum()
importance_df_rf1 = sicc(importance_df_rf1)
importance_df_rf1p = importance_df_rf1.sort_values('importance', ascending=False).head(15)
importance_df_rf1p = importance_df_rf1p.sort_values('importance', ascending=True)
# Plot
plt.figure(figsize=(10, 9), dpi = 300)
plt.barh(importance_df_rf1p['feature'], importance_df_rf1p['importance'], color = 'steelblue')
plt.xlabel('Relative Importance', fontsize = 15)
plt.title('Random Forest importance top 1,000', fontsize=23, weight='bold')
plt.xticks(fontsize=15)  
plt.yticks(fontsize=15) 
plt.tight_layout()
plt.show()


# 5000
all_importances_rf5 = []
for model in dump_rf_5000.values():
    all_importances_rf5.append(model.feature_importances_)

avg_importance_rf5 = np.mean(all_importances_rf5, axis=0)
feature_names_rf5 = dump_rf_5000[list(dump_rf_5000.keys())[0]].feature_name_
importance_df_rf5 = pd.DataFrame({'feature': feature_names_rf5, 'importance': avg_importance_rf5})
# here i transform to get relative importance as in the paper
importance_df_rf5['importance'] = importance_df_rf5['importance'] / importance_df_rf5['importance'].sum()
importance_df_rf5 = sicc(importance_df_rf5)
importance_df_rf5p = importance_df_rf5.sort_values('importance', ascending=False).head(15)
importance_df_rf5p = importance_df_rf5p.sort_values('importance', ascending=True)
# Plot
plt.figure(figsize=(10, 9), dpi = 300)
plt.barh(importance_df_rf5p['feature'], importance_df_rf5p['importance'], color = 'darkkhaki')
plt.xlabel('Relative Importance', fontsize = 15)
plt.title('Random Forest importance top 5,000', fontsize=23, weight='bold')
plt.xticks(fontsize=15)  
plt.yticks(fontsize=15) 
plt.tight_layout()
plt.show()





# plot for overall importances across each model
# weighted for all (so both 1000 and 5000 ø)

# List of models
models = ['gbm', 'rf', 'ENet']

# 5000
avg_importance_list = []

# Loop over each model
for model in models:
    # Load the importance dataframes
    df5 = globals()[f'importance_df_{model}5']
    
    df_avg = (df5.set_index('feature'))

    # Add the model name as a new column
    df_avg['model'] = model

    # Append to the overall dataframe
    avg_importance_list.append(df_avg.reset_index())


# Concatenate all dataframes in the list
avg_importance_df = pd.concat(avg_importance_list)
avg_importance_df
# Pivot the dataframe for the heatmap
heatmap_data = avg_importance_df.pivot(index='feature', columns='model', values='importance')

# Sort by the maximum value in each row
heatmap_data = heatmap_data.loc[heatmap_data.max(axis=1).sort_values(ascending=False).index]

# Create the heatmap
plt.figure(figsize=(6, 10), dpi = 300)
cmap = sns.light_palette("steelblue", as_cmap=True)
sns.heatmap(heatmap_data, cmap=cmap, linewidths = 0.002, linecolor = 'white', cbar_kws = {"aspect": 40})
plt.title('Average Feature Importance Across Models')
plt.show()






# average over both models
# Initialize an empty dataframe for storing average importances
avg_importance_list = []

# Loop over each model
for model in models:
    # Load the importance dataframes
    df1 = globals()[f'importance_df_{model}1']
    df5 = globals()[f'importance_df_{model}5']

    # Calculate the average importance
    df_avg = (df1.set_index('feature') + df5.set_index('feature')) / 2

    # Add the model name as a new column
    df_avg['model'] = model

    # Append to the overall dataframe
    avg_importance_list.append(df_avg.reset_index())


# Concatenate all dataframes in the list
avg_importance_df = pd.concat(avg_importance_list)
avg_importance_df
# Pivot the dataframe for the heatmap
heatmap_data = avg_importance_df.pivot(index='feature', columns='model', values='importance')

# Sort by the maximum value in each row
heatmap_data = heatmap_data.loc[heatmap_data.max(axis=1).sort_values(ascending=False).index]

# Create the heatmap
plt.figure(figsize=(6, 10))
cmap = sns.light_palette("steelblue", as_cmap=True)
sns.heatmap(heatmap_data, cmap=cmap, linewidths = 0.002, linecolor = 'white', cbar_kws = {"aspect": 40})
plt.title('Average Feature Importance Across Models')
plt.show()










## performance of ML portfolios
# At the end of month T , we calculate one-month-ahead out-of-sample stock return predictions for each method. We then sort stocks into deciles based on each model’s forecasts. And we buy the highest expected return stocks (decile 10) and sells the lowest (decile 1). At the end of month T + 1, we can calculate the realized returns of the portfolios (buy side and sell side respectively)
# at the end we want for each model and each decile: predicted monthly return, ø realized monthly return, standard deviation of realized returns & thereof SR
# Note that SR for our top data will be lower since a lot is driven my microcaps (see respective table without microcaps)
pf1 = results_1000.copy()
pf5 = results_5000.copy()
del results_1000, results_5000

def create_deciles(group, model_columns):
    for col in model_columns:
        decile_col = col + '_DECILE'
        # i handle non-unique bin edges by using duplicates='drop'
        group[decile_col] = pd.qcut(group[col], q=10, labels=False, duplicates='drop') + 1
    return group

pf_with_deciles1 = pf1.groupby(pf1.index).apply(lambda x: create_deciles(x, models1))
pf_with_deciles5 = pf5.groupby(pf5.index).apply(lambda x: create_deciles(x, models5))

# check why i get nans for ENet & GBM prediction / deciles
#pf_with_deciles.isna().sum()
# seems like same predicitons over multiple; deciles can not be formed for these
# Elastic Net has coefficents = 0 from like 2008 on, so give the same predictions for every single one; have to fix this

# check for decile distribution
#values1 = pf_with_deciles1['GBM_PRED_DECILE'].value_counts()




# equal weighted
# 1000
results1 = pd.DataFrame()
results1

results_list1 = []

# Calculate metrics for each model and decile
for model in models1:
    decile_col = model + '_DECILE'
    grouped = pf_with_deciles1.groupby([pf_with_deciles1.index, decile_col])
    
    # Calculate average predicted return for each decile
    avg_pred = grouped[model].mean().reset_index()
    avg_pred.columns = ['DATE', 'Decile', 'Pred']
    
    # Calculate average realized return for each decile
    avg_realized = grouped['RET'].mean().reset_index()
    avg_realized.columns = ['DATE', 'Decile', 'Avg']
    
    # Calculate standard deviation of realized returns for each decile
    std_realized = grouped['RET'].std().reset_index()
    std_realized.columns = ['DATE', 'Decile', 'Std']
    
    # Merge the results
    merged = avg_pred.merge(avg_realized, on=['DATE', 'Decile']).merge(std_realized, on=['DATE', 'Decile'])
    
    # Calculate Sharpe Ratio (assuming risk-free rate is 0)
    merged['SR'] = ((1 + merged['Avg']) ** 12 -1) / (merged['Std'] * np.sqrt(12))
    
    # Add the model name to identify in the final DataFrame
    merged['Model'] = model
    
    # Append to the results list
    results_list1.append(merged)

# Concatenate all the results
results1 = pd.concat(results_list1, ignore_index=True)
results1


HML_dict1 = {}
for model in models1:
    # create the pivot table for the format of dec 1-10
    pf = results1[results1['Model'] == model].pivot_table(values = ['Pred', 'Avg', 'Std', 'SR'], index = 'Decile')
    # now inserting the HML row
    pf.loc['H-L', 'Avg'] = pf.loc[10.0, 'Avg'] - pf.loc[1.0, 'Avg']
    pf.loc['H-L', 'Pred'] = pf.loc[10.0, 'Pred'] - pf.loc[1.0, 'Pred']
    
    # now i calculate the std of HML
    filt = results1[results1['Model'] == model]
    filt = filt[['DATE', 'Decile', 'Avg', 'Model']]
    filt_dec10 = filt[filt['Decile'] == 10]
    filt_dec1 = filt[filt['Decile'] == 1]
    filt_dec10
    filt_dec1
    merged_filt = pd.merge(filt_dec10, filt_dec1, on=['DATE', 'Model'], suffixes=('_10', '_1'))
    merged_filt['HML_Return'] = merged_filt['Avg_10'] - merged_filt['Avg_1']
    pf_std = merged_filt['HML_Return'].std()
    pf.loc['H-L', 'Std'] = pf_std
    
    # and last the annualized Sharpe Ratio
    pf.loc['H-L', 'SR'] = ((1+ pf.loc['H-L', 'Avg']) ** 12 - 1) / (pf.loc['H-L', 'Std'] * np.sqrt(12))    
    
    pf = pf.reindex(columns = ['Pred', 'Avg', 'Std', 'SR'])
    HML_dict1[model] = pf

HML_dict1

with pd.ExcelWriter('HML1000.xlsx') as writer:
    for model, df in HML_dict1.items():
        df.to_excel(writer, sheet_name=model, index=True)





# 5000
results5 = pd.DataFrame()
results5

results_list5 = []

# Calculate metrics for each model and decile
for model in models5:
    decile_col = model + '_DECILE'
    grouped = pf_with_deciles5.groupby([pf_with_deciles5.index, decile_col])
    
    # Calculate average predicted return for each decile
    avg_pred = grouped[model].mean().reset_index()
    avg_pred.columns = ['DATE', 'Decile', 'Pred']
    
    # Calculate average realized return for each decile
    avg_realized = grouped['RET'].mean().reset_index()
    avg_realized.columns = ['DATE', 'Decile', 'Avg']
    
    # Calculate standard deviation of realized returns for each decile
    std_realized = grouped['RET'].std().reset_index()
    std_realized.columns = ['DATE', 'Decile', 'Std']
    
    # Merge the results
    merged = avg_pred.merge(avg_realized, on=['DATE', 'Decile']).merge(std_realized, on=['DATE', 'Decile'])
    
    # Calculate Sharpe Ratio (assuming risk-free rate is 0)
    merged['SR'] = ((1 + merged['Avg']) ** 12 -1) / (merged['Std'] * np.sqrt(12))
    
    # Add the model name to identify in the final DataFrame
    merged['Model'] = model
    
    # Append to the results list
    results_list5.append(merged)

# Concatenate all the results
results5 = pd.concat(results_list5, ignore_index=True)



HML_dict5 = {}
for model in models5:
    # create the pivot table for the format of dec 1-10
    pf = results5[results5['Model'] == model].pivot_table(values = ['Pred', 'Avg', 'Std', 'SR'], index = 'Decile')
    # now inserting the HML row
    pf.loc['H-L', 'Avg'] = pf.loc[10.0, 'Avg'] - pf.loc[1.0, 'Avg']
    pf.loc['H-L', 'Pred'] = pf.loc[10.0, 'Pred'] - pf.loc[1.0, 'Pred']
    
    # now i calculate the std of HML
    filt = results5[results5['Model'] == model]
    filt = filt[['DATE', 'Decile', 'Avg', 'Model']]
    filt_dec10 = filt[filt['Decile'] == 10]
    filt_dec1 = filt[filt['Decile'] == 1]
    filt_dec10
    filt_dec1
    merged_filt = pd.merge(filt_dec10, filt_dec1, on=['DATE', 'Model'], suffixes=('_10', '_1'))
    merged_filt['HML_Return'] = merged_filt['Avg_10'] - merged_filt['Avg_1']
    pf_std = merged_filt['HML_Return'].std()
    pf.loc['H-L', 'Std'] = pf_std
    
    # and last the annualized Sharpe Ratio
    pf.loc['H-L', 'SR'] = ((1+ pf.loc['H-L', 'Avg']) ** 12 - 1) / (pf.loc['H-L', 'Std'] * np.sqrt(12))    
    
    pf = pf.reindex(columns = ['Pred', 'Avg', 'Std', 'SR'])
    HML_dict5[model] = pf

HML_dict5

with pd.ExcelWriter('HML5000.xlsx') as writer:
    for model, df in HML_dict5.items():
        df.to_excel(writer, sheet_name=model, index=True)


















## cumulative returns as final plot
## to do: model separation for each results (1000 vs 5000 vs predicted 5k)

# 1000
rets1 = results1[['DATE', 'Avg', 'Decile', 'Model']]
rets1 = rets1[rets1['Model'].isin(['ENet_PRED', 'GBM_PRED', 'RF_PRED', 'PCR_PRED', 'PLS_PRED'])]
rets1 = rets1[rets1['Decile'].isin([10,1])]
rets1['DATE'] = rets1['DATE'].apply(lambda x: x[0])
rets1['DATE'] = pd.to_datetime(rets1['DATE'])
rets1.set_index('DATE', inplace = True)
rets1.sort_index(inplace = True)

decile101 = rets1[rets1['Decile'] == 10]
decile11 = rets1[rets1['Decile'] == 1]

cumulative_log_returns_101 = decile101.groupby('Model')['Avg'].apply(lambda x: np.log(1 + x).cumsum())
cumulative_log_returns_11 = decile11.groupby('Model')['Avg'].apply(lambda x: np.log(1 + x).cumsum())

colors = plt.cm.get_cmap('tab10', len(rets1['Model'].unique()))

fig, ax = plt.subplots(figsize=(12, 8), dpi = 300)

# Plotting each model's cumulative log returns for Deciles 10 and 1
for i, model in enumerate(rets1['Model'].unique()):
    ax.plot(cumulative_log_returns_101[model].index, cumulative_log_returns_101[model], color=colors(i), linestyle='-')
    ax.plot(cumulative_log_returns_11[model].index, cumulative_log_returns_11[model], color=colors(i), linestyle='--')

# Add recession shading
recession_periods = [
    ('1990-07-01', '1991-03-31'),
    ('2001-03-01', '2001-11-30'),
    ('2007-12-01', '2009-06-30')
]

for start, end in recession_periods:
    ax.axvspan(pd.to_datetime(start), pd.to_datetime(end), color='gray', alpha=0.3)

# Customize the plot
ax.set_title('Cumulative log Returns Over Time (solid = decile 10; dashed = decile 1) - base model top 1,000')
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative log Returns')
ax.grid(True, linestyle='--', alpha=0.5)

# Move legend above the plot
handles = [mlines.Line2D([], [], color=colors(i), linestyle='-', label=model) for i, model in enumerate(rets1['Model'].unique())]
ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)

# Set y-axis limits similar to the image
ax.set_ylim(-1, 4)

# Show the plot
plt.tight_layout()
plt.show()








# 5000
rets5 = results5[['DATE', 'Avg', 'Decile', 'Model']]
rets5 = rets5[rets5['Model'].isin(['ENet_PRED', 'GBM_PRED', 'RF_PRED', 'PCR_PRED', 'PLS_PRED'])]
rets5 = rets5[rets5['Decile'].isin([10,1])]
rets5['DATE'] = rets5['DATE'].apply(lambda x: x[0])
rets5['DATE'] = pd.to_datetime(rets5['DATE'])
rets5.set_index('DATE', inplace = True)
rets5.sort_index(inplace = True)

decile105 = rets5[rets5['Decile'] == 10]
decile15 = rets5[rets5['Decile'] == 1]

cumulative_log_returns_105 = decile105.groupby('Model')['Avg'].apply(lambda x: np.log(1 + x).cumsum())
cumulative_log_returns_15 = decile15.groupby('Model')['Avg'].apply(lambda x: np.log(1 + x).cumsum())

colors = plt.cm.get_cmap('tab10', len(rets5['Model'].unique()))

fig, ax = plt.subplots(figsize=(12, 8))

# Plotting each model's cumulative log returns for Deciles 10 and 1
for i, model in enumerate(rets1['Model'].unique()):
    ax.plot(cumulative_log_returns_105[model].index, cumulative_log_returns_105[model], color=colors(i), linestyle='-')
    ax.plot(cumulative_log_returns_15[model].index, cumulative_log_returns_15[model], color=colors(i), linestyle='--')

# Add recession shading
recession_periods = [
    ('1990-07-01', '1991-03-31'),
    ('2001-03-01', '2001-11-30'),
    ('2007-12-01', '2009-06-30')
]

for start, end in recession_periods:
    ax.axvspan(pd.to_datetime(start), pd.to_datetime(end), color='gray', alpha=0.3)

# Customize the plot
ax.set_title('Cumulative log Returns Over Time (solid = decile 10; dashed = decile 1) - base model top 5,000')
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative log Returns')
ax.grid(True, linestyle='--', alpha=0.5)

# Move legend above the plot
handles = [mlines.Line2D([], [], color=colors(i), linestyle='-', label=model) for i, model in enumerate(rets1['Model'].unique())]
ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)

# Set y-axis limits similar to the image
ax.set_ylim(-1, 4)

# Show the plot
plt.tight_layout()
plt.show()














## playground for predicting 1000 with 5000 model

# we now need our dataset to then feed into the trained 5000 model. With this trained model we can then predict and do the same analysis as for the above
# top 1000 data 
data = pd.read_pickle()
data['YEAR'] = data.index.year
data = data.reset_index()
res = data[['DATE', 'permno','RET']].copy()

features = list(set(data.columns).difference({'permno','DATE','RET', 'YEAR'}))
features


res['ENET'] = None
# Iterate through each year and make predictions
for year in range(1987, 2020):
    data_year = data[data['YEAR'] == year]
    X_year = data_year[features]
    # get the corresponding model for the current year
    model = dump_ENets_5000[f'ENet_{year}']
    predictions = model.predict(X_year)
    res.loc[data_year.index, 'ENET'] = predictions


res['GBM'] = None
for year in range(1987, 2020):
    data_year = data[data['YEAR'] == year]
    X_year = data_year[features]
    model = dump_gbm_5000[f'gbm_{year}']
    predictions = model.predict(X_year)
    res.loc[data_year.index, 'GBM'] = predictions


res['RF'] = None
for year in range(1987, 2020):
    data_year = data[data['YEAR'] == year]
    X_year = data_year[features]
    model = dump_rf_5000[f'RF_{year}']
    predictions = model.predict(X_year)
    res.loc[data_year.index, 'RF'] = predictions


res['OLS3'] = None
for year in range(1987, 2020):
    data_year = data[data['YEAR'] == year]
    X_year = data_year[features_3]
    model = dump_OLS3_5000[f'OLS_sub_{year}']
    predictions = model.predict(X_year)
    res.loc[data_year.index, 'OLS3'] = predictions




res.set_index('DATE', inplace = True)




# R ^2 oos
modelsx = list(set(res.columns).difference({'DATE','RET', 'permno'}))
month_roos_dumpx = {}

for model in modelsx:
    r_oos_valuex = R_oos(res['RET'], res[model]) * 100
    month_roos_dumpx[model] = [r_oos_valuex]
    
monthly_roosx = pd.DataFrame(month_roos_dumpx, index = ['Monthly R^2_oos mix'])
monthly_roosx


# annual
annual_resultsx = res.groupby([res.index.year, 'permno']).sum()
annual_resultsx

ann_roos_dumpx = {}

for model in modelsx:
    r_oos_valuex = R_oos(annual_resultsx['RET'], annual_resultsx[model]) * 100
    ann_roos_dumpx[model] = [r_oos_valuex]

annual_roosx = pd.DataFrame(ann_roos_dumpx, index = ['Annual R^2_oos mix'])
annual_roosx



# plots
new_columns = {
    'PCR_PRED': 'PCR',
    'ENet_PRED': 'ENET',
    'OLS_PRED': 'OLS',
    'OLS_3_PRED': 'OLS3',
    'GBM_PRED': 'GBM',
    'RF_PRED': 'RF',
    'PLS_PRED': 'PLS'
}


# monthly plot

monthly_roos.rename(columns=new_columns, inplace=True)
monthly_roos_all = pd.concat([monthly_roos, monthly_roosx], axis=0)

monthly_roos_all = monthly_roos_all[['OLS', 'OLS3', 'ENET', 'GBM', 'RF', 'PCR', 'PLS']]
monthly_roos_all
# Separate the DataFrame
monthly_main = monthly_roos_all.iloc[:2]
monthly_mix = monthly_roos_all.iloc[2:]

### plot with OLS white since too large negative
fig, ax = plt.subplots(figsize=(10, 6), dpi = 300)
bar_width = 0.4
index = np.arange(len(monthly_main.columns))
# Plot the main bars
plt.bar(index, monthly_main.iloc[0], bar_width, label='top 1000', color = 'steelblue')
# Plot the bars for top 5000, but make the OLS bar white
for i in range(len(index)):
    if monthly_main.columns[i] == 'OLS':
        plt.bar(index[i] + bar_width, monthly_main.iloc[1][i], bar_width, color='white')
    else:
        plt.bar(index[i] + bar_width, monthly_main.iloc[1][i], bar_width, label='top 5000' if i == 0 else "", color='darkkhaki')

for i, col in enumerate(monthly_main.columns):
    if not pd.isna(monthly_mix[col].values[0]):
        plt.axhline(y=monthly_mix[col].values[0], xmin=(i/(len(index))), xmax=((i + 1)/(len(index))), color='red', linestyle='--', label='predicted top' if i == 0 else "")
plt.xlabel('Models')
plt.ylabel('R²oos (%)')
plt.title('Monthly out-of-sample stock-level prediction performance (percentage R²oos)')
plt.xticks(index + bar_width / 2, monthly_main.columns, rotation=45, ha='right')
# here i add labels
for i in range(len(index)):
    plt.text(i, monthly_main.iloc[0][i], f'{monthly_main.iloc[0][i]:.2f}%', ha='right', va='bottom')
    if monthly_main.columns[i] != 'OLS':
        plt.text(i + bar_width, monthly_main.iloc[1][i], f'{monthly_main.iloc[1][i]:.2f}%', ha='center', va='bottom')
    if not pd.isna(monthly_mix.iloc[0][i]):
        plt.text(i + 0.2, monthly_mix.iloc[0][i], f'{monthly_mix.iloc[0][i]:.2f}%', ha='center', va='bottom')

from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='steelblue', lw=4),
                Line2D([0], [0], color='darkkhaki', lw=4),
                Line2D([0], [0], color='red', lw=2, linestyle='--')]
plt.legend(custom_lines, ['top 1,000', 'top 5,000', 'predicted top 1,000'])

plt.ylim([-0.6, 1.4])
plt.tight_layout()
plt.show()








# annual plot
annual_roos.rename(columns=new_columns, inplace=True)
annual_roos_all = pd.concat([annual_roos, annual_roosx], axis=0)

#reorder for the plot
annual_roos_all = annual_roos_all[['OLS', 'OLS3', 'ENET', 'GBM', 'RF', 'PCR', 'PLS']]

# Separate the DataFrame
annual_main = annual_roos_all.iloc[:2]
annual_mix = annual_roos_all.iloc[2:]

fig, ax = plt.subplots(figsize=(10, 6), dpi = 300)

# Bar width and index setup
bar_width = 0.4
index = np.arange(len(annual_main.columns))

# Plot the main bars
plt.bar(index, annual_main.iloc[0], bar_width, label='top 1000', color='steelblue')
plt.bar(index + bar_width, annual_main.iloc[1], bar_width, label='top 5000', color='darkkhaki')

for i, col in enumerate(annual_main.columns):
    if not pd.isna(annual_mix[col].values[0]):
        plt.axhline(y=annual_mix[col].values[0], xmin=(i/(len(index))), xmax=((i + 1)/(len(index))), color='red', linestyle='--', label='predicted top' if i == 0 else "")


# Set labels and title
plt.xlabel('Models')
plt.ylabel('R²oos (%)')
plt.title('Annual out-of-sample stock-level prediction performance (percentage R²oos)')
plt.xticks(index + bar_width / 2, annual_main.columns, rotation=45, ha='right')

# Add data labels
for i in range(len(index)):
    plt.text(i, annual_main.iloc[0][i] + 0.2, f'{annual_main.iloc[0][i]:.2f}%', ha='center', va='bottom')
    plt.text(i + bar_width, annual_main.iloc[1][i], f'{annual_main.iloc[1][i]:.2f}%', ha='center', va='bottom')
    if not pd.isna(annual_mix.iloc[0][i]):
        plt.text(i + 0.2, annual_mix.iloc[0][i], f'{annual_mix.iloc[0][i]:.2f}%', ha='center', va='bottom')

from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='steelblue', lw=4),
                Line2D([0], [0], color='darkkhaki', lw=4),
                Line2D([0], [0], color='red', lw=2, linestyle='--')]
plt.legend(custom_lines, ['top 1,000', 'top 5,000', 'predicted top 1,000'])
plt.tight_layout()
plt.show()














## portfolios
pf = res.copy()
pf_with_decilesx = pf.groupby(pf.index).apply(lambda x: create_deciles(x, modelsx))
pf_with_decilesx


resultsx = pd.DataFrame()
resultsx

results_listx = []

# Calculate metrics for each model and decile
for model in modelsx:
    decile_col = model + '_DECILE'
    grouped = pf_with_decilesx.groupby([pf_with_decilesx.index, decile_col])
    
    # Calculate average predicted return for each decile
    avg_pred = grouped[model].mean().reset_index()
    avg_pred.columns = ['DATE', 'Decile', 'Pred']
    
    # Calculate average realized return for each decile
    avg_realized = grouped['RET'].mean().reset_index()
    avg_realized.columns = ['DATE', 'Decile', 'Avg']
    
    # Calculate standard deviation of realized returns for each decile
    std_realized = grouped['RET'].std().reset_index()
    std_realized.columns = ['DATE', 'Decile', 'Std']
    
    # Merge the results
    merged = avg_pred.merge(avg_realized, on=['DATE', 'Decile']).merge(std_realized, on=['DATE', 'Decile'])
    
    # Calculate Sharpe Ratio (assuming risk-free rate is 0)
    merged['SR'] = ((1 + merged['Avg']) ** 12 -1) / (merged['Std'] * np.sqrt(12))
    
    # Add the model name to identify in the final DataFrame
    merged['Model'] = model
    
    # Append to the results list
    results_listx.append(merged)

# Concatenate all the results
resultsx = pd.concat(results_listx, ignore_index=True)

resultsx

modelsx = list(set(res.columns).difference({'DATE','RET', 'permno', 'GBM'}))


HML_dictx = {}
for model in modelsx:
    # create the pivot table for the format of dec 1-10
    pf = resultsx[resultsx['Model'] == model].pivot_table(values = ['Pred', 'Avg', 'Std', 'SR'], index = 'Decile')
    # now inserting the HML row
    pf.loc['H-L', 'Avg'] = pf.loc[10.0, 'Avg'] - pf.loc[1.0, 'Avg']
    pf.loc['H-L', 'Pred'] = pf.loc[10.0, 'Pred'] - pf.loc[1.0, 'Pred']
    
    # now i calculate the std of HML
    filt = resultsx[resultsx['Model'] == model]
    filt = filt[['DATE', 'Decile', 'Avg', 'Model']]
    filt_dec10 = filt[filt['Decile'] == 10]
    filt_dec1 = filt[filt['Decile'] == 1]
    filt_dec10
    filt_dec1
    merged_filt = pd.merge(filt_dec10, filt_dec1, on=['DATE', 'Model'], suffixes=('_10', '_1'))
    merged_filt['HML_Return'] = merged_filt['Avg_10'] - merged_filt['Avg_1']
    pf_std = merged_filt['HML_Return'].std()
    pf.loc['H-L', 'Std'] = pf_std
    
    # and last the annualized Sharpe Ratio
    pf.loc['H-L', 'SR'] = ((1+ pf.loc['H-L', 'Avg']) ** 12 - 1) / (pf.loc['H-L', 'Std'] * np.sqrt(12))    
    
    pf = pf.reindex(columns = ['Pred', 'Avg', 'Std', 'SR'])
    HML_dictx[model] = pf

HML_dictx










resultsx = resultsx[resultsx['Model'].isin(['ENET', 'OLS3', 'RF'])]

# log cumulative rets plot
retsx = resultsx[['DATE', 'Avg', 'Decile', 'Model']]
retsx = retsx[retsx['Decile'].isin([10,1])]
retsx['DATE'] = retsx['DATE'].apply(lambda x: x[0])
retsx['DATE'] = pd.to_datetime(retsx['DATE'])
retsx.set_index('DATE', inplace = True)
retsx.sort_index(inplace = True)

decile10x = retsx[retsx['Decile'] == 10]
decile1x = retsx[retsx['Decile'] == 1]

cumulative_log_returns_10x = decile10x.groupby('Model')['Avg'].apply(lambda x: np.log(1 + x).cumsum())
cumulative_log_returns_1x = decile1x.groupby('Model')['Avg'].apply(lambda x: np.log(1 + x).cumsum())

colors = plt.cm.get_cmap('tab10', len(retsx['Model'].unique()))

fig, ax = plt.subplots(figsize=(12, 8))

# Plotting each model's cumulative log returns for Deciles 10 and 1
for i, model in enumerate(retsx['Model'].unique()):
    ax.plot(cumulative_log_returns_10x[model].index, cumulative_log_returns_10x[model], color=colors(i), linestyle='-')
    ax.plot(cumulative_log_returns_1x[model].index, cumulative_log_returns_1x[model], color=colors(i), linestyle='--')

# Add recession shading
recession_periods = [
    ('1990-07-01', '1991-03-31'),
    ('2001-03-01', '2001-11-30'),
    ('2007-12-01', '2009-06-30')
]

for start, end in recession_periods:
    ax.axvspan(pd.to_datetime(start), pd.to_datetime(end), color='gray', alpha=0.3)

# Customize the plot
ax.set_title('Cumulative log Returns Over Time (solid = decile 10; dashed = decile 1)')
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative log Returns')
ax.grid(True, linestyle='--', alpha=0.7)

# Move legend above the plot
handles = [mlines.Line2D([], [], color=colors(i), linestyle='-', label=model) for i, model in enumerate(retsx['Model'].unique())]
ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)

# Set y-axis limits similar to the image
ax.set_ylim(-1, 4)

# Show the plot
plt.tight_layout()
plt.show()










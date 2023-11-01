import numpy as np
import pandas as pd
#import os
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder
import pickle
from sklearn.preprocessing import LabelEncoder

def inference_pipeline_dec_tree(dat, le):
    ## using decision Tree as it seems to have best performance
    #le = LabelEncoder()
    dat["Race_code"] = le.fit_transform(dat[["Race"]])
    dat["Gender_code"] = le.fit_transform(dat[["Gender"]])
    dat["college_code"] = le.fit_transform(dat[["college"]])
    print(dat["college"])
    #make predictions
    print(dat["college_code"])
    dat = dat[['SAT Score', 'ACT Score', 'Number of AP Courses',
               'Sports or Other Activities', 'Financial aid needed',
               'GPA', 'Race_code', 'Gender_code', 'college_code']]
    print(dat)
    ypred = dec_tree.predict(scaler.fit_transform(dat))    
    print("Your chance of admission at ",le.inverse_transform(dat.college_code),"is: ",ypred)     
    return ypred


# Below is the new cleaned up model - not working due to memory limits I guess
data_model = pd.read_csv("/data/model_data_clean_2.csv", engine='python', sep=',', quotechar='"', on_bad_lines='warn')
print(data_model)

data_model.head()
data_model.rename(columns={"newScore":"Chance of Admit"},inplace=True)

dict = {'SATScore': 'SAT Score',
        'ACTScore': 'ACT Score',
        'NumAP': 'Number of AP Courses',
         'Sports':'Sports or Other Activities',
       'finaid':'Financial aid needed'}

data_model.rename(columns=dict,
          inplace=True)
data_model.head()
le = LabelEncoder()
data_model["Race_code"] = le.fit_transform(data_model[["Race"]])
data_model["Gender_code"] = le.fit_transform(data_model[["Gender"]])
data_model["college_code"] = le.fit_transform(data_model[["college"]])

col_re = ['SAT Score', 'ACT Score', 'Number of AP Courses','college_code',
       'Financial aid needed', 'Sports or Other Activities', 'GPA',
          'Race_code', 'Gender_code','Chance of Admit']
data_model=data_model[col_re]

data_model.head()

targets = data_model['Chance of Admit']
features = data_model.drop(columns = {'Chance of Admit'})

X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

X_test_original = X_test.copy()
y_test_original = y_test.copy()

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

linreg = LinearRegression()
linreg.fit(X_train, y_train)
# save the model to disk
filename = 'LinearRegression_model.sav'
pickle.dump(linreg, open(filename, 'wb'))
y_predict = linreg.predict(X_test)
linreg_score = (linreg.score(X_test, y_test))*100
print(linreg_score)

dec_tree = DecisionTreeRegressor(random_state=0, max_depth=6)
dec_tree.fit(X_train, y_train)
filename = 'DecisionTreeRegressor_model.sav'
pickle.dump(dec_tree, open(filename, 'wb'))
y_predict = dec_tree.predict(X_test)
dec_tree_score = (dec_tree.score(X_test, y_test))*100
print(dec_tree_score)

forest = RandomForestRegressor(n_estimators=100,max_depth=4,random_state=0)
forest.fit(X_train, y_train)
filename = 'RandomForestRegressor_model.sav'
# pickle.dump(forest, open(filename, 'wb'))
y_predict = forest.predict(X_test)
forest_score = (forest.score(X_test, y_test))*100
print(forest_score)

new_test_data_2 = pd.read_csv("/data/userInput.csv")
print(new_test_data_2)
new_test_data_2.rename(columns={"newScore":"Chance of Admit"},inplace=True)
## remove totscore

#new_test_data_2.drop(['totscore','Chance of Admit'], axis=1, inplace=True)
dict = {'SATScore': 'SAT Score',
        'ACTScore': 'ACT Score',
        'NumAP': 'Number of AP Courses',
         'Sports':'Sports or Other Activities',
       'finaid':'Financial aid needed'}

new_test_data_2.rename(columns=dict,
          inplace=True)
new_test_data_2.head()
coaList = inference_pipeline_dec_tree(new_test_data_2, le)
print("COA is : {0}".format(coaList[0]))




    



'''
# df_model.drop(['college'], axis=1, inplace=True)
ord_enc = OrdinalEncoder()
df_model["Race_code"] = ord_enc.fit_transform(df_model[["Race"]])
df_model["Gender_code"] = ord_enc.fit_transform(df_model[["Gender"]])
df_model["college"] = ord_enc.fit_transform(df_model[["college"]])

df_model.drop(['Race','Gender'], axis=1, inplace=True)

col_re = ['SAT Score', 'ACT Score', 'Number of AP Courses', 'college',
       'Financial aid needed', 'Sports or Other Activities', 'GPA',
          'Race_code', 'Gender_code','Chance of Admit']
df_model=df_model[col_re]
df_model.head()
print(df_model.describe())

targets = df_model['Chance of Admit']
features = df_model.drop(columns = {'Chance of Admit'})

X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

X_test_original = X_test.copy()
y_test_original = y_test.copy()

scaler = StandardScaler()
scale_fit = scaler.fit(X_train.values) #save the mean and std. dev computed for your data.
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


print(X_test_original.columns)

linreg = LinearRegression()
linreg.fit(X_train, y_train)
# save the model to disk
filename = 'finalized_LR_model.sav'
pickle.dump(linreg, open(filename, 'wb'))
 
y_predict = linreg.predict(X_test)
linreg_score = (linreg.score(X_test, y_test))*100
print(linreg_score)

dec_tree = DecisionTreeRegressor(random_state=0, max_depth=6)
dec_tree.fit(X_train, y_train)
filename = 'finalized_DT_model.sav'
pickle.dump(linreg, open(filename, 'wb'))
y_predict = dec_tree.predict(X_test)
dec_tree_score = (dec_tree.score(X_test, y_test))*100
print(dec_tree_score)

# forest = RandomForestRegressor(n_estimators=100,max_depth=4,random_state=0)
# forest.fit(X_train, y_train)
# filename = 'finalized_RF_model.sav'
# pickle.dump(forest, open(filename, 'wb'))
# y_predict = forest.predict(X_test)
# forest_score = (forest.score(X_test, y_test))*100
# print(forest_score)



X_new = [[1600,0,16,"Carnegie Mellon University",1,1,9.65,5,0]]
X_new = scale_fit.transform(X_new) #use the above saved 
print(X_new)
y_new_predict = dec_tree.predict(X_new)
print(y_new_predict)
'''
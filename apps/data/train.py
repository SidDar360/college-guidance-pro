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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn_pandas import DataFrameMapper
import warnings
warnings.filterwarnings("ignore")


def feature_preprocess(df, num_vars, cat_vars, dep_var):
    ### 1.5 Imputation and scaling
    preprocessor = DataFrameMapper(
        [ ([c], [LabelEncoder()]) for c in cat_vars] +
        [([c], [SimpleImputer(strategy = 'mean'), StandardScaler()]) for c in num_vars],
        df_out = True
    )

    preprocessor.fit(df)
    X_it = preprocessor.transform(df)
    Y_it = df[dep_var]

    return X_it, Y_it, preprocessor

def inference_pipeline(dat, preprocessor, model):
    X_future = preprocessor.transform(dat)
    ypred = model.predict(X_future)    
    print("Your chance of admission at ",dat.college,"is: ",ypred)     
    return ypred


# Below is the new cleaned up model - not working due to memory limits I guess
data_model = pd.read_csv("/data/model_data_clean_3.csv", engine='python', sep=',', quotechar='"', on_bad_lines='warn')

col_name_dict = { 'newScore': 'Chance of Admit',
        'SATScore': 'SAT Score',
        'ACTScore': 'ACT Score',
        'NumAP': 'Number of AP Courses',
         'Sports':'Sports or Other Activities',
       'finaid':'Financial aid needed'}

data_model.rename(columns=col_name_dict,
          inplace=True)

data_model.head()
cat_vars = ['Race', 'Gender','college']
num_vars = ['SAT Score', 'ACT Score', 'Number of AP Courses',  'Financial aid needed', 'Sports or Other Activities', 'GPA']
dep_var = 'Chance of Admit'

X_it, Y_it, preprocessor = feature_preprocess(data_model, num_vars, cat_vars, dep_var)
X_train, X_test, y_train, y_test = train_test_split(X_it, Y_it, test_size=0.2, random_state=42)


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
pickle.dump(forest, open(filename, 'wb'))
y_predict = forest.predict(X_test)
forest_score = (forest.score(X_test, y_test))*100
print(forest_score)

new_test_data_2 = pd.read_csv("/data/userInput.csv")
new_test_data_2.rename(columns=col_name_dict,inplace=True)
print(new_test_data_2.head())
coaList = inference_pipeline(new_test_data_2, preprocessor, dec_tree)
print("COA is : {0}".format(coaList[0]))


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




# data_model = pd.read_csv("/data/model_data_clean_2.csv", engine='python', sep=',', quotechar='"', on_bad_lines='warn')
# print(data_model)
# data_model.head()
# data_model.rename(columns={"newScore":"Chance of Admit"},inplace=True)

# dict = {'SATScore': 'SAT Score',
#     'ACTScore': 'ACT Score',
#     'NumAP': 'Number of AP Courses',
#         'Sports':'Sports or Other Activities',
#     'finaid':'Financial aid needed'}

# data_model.rename(columns=dict,
#         inplace=True)
# data_model.head()
# le = LabelEncoder()
# data_model["Race_code"] = le.fit_transform(data_model[["Race"]])
# data_model["Gender_code"] = le.fit_transform(data_model[["Gender"]])
# data_model["college_code"] = le.fit_transform(data_model[["college"]])

# col_re = ['SAT Score', 'ACT Score', 'Number of AP Courses','college_code',
#     'Financial aid needed', 'Sports or Other Activities', 'GPA',
#         'Race_code', 'Gender_code','Chance of Admit']
# data_model=data_model[col_re]

# data_model.head()

# targets = data_model['Chance of Admit']
# features = data_model.drop(columns = {'Chance of Admit'})

# X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.fit_transform(X_test)


def inference_pipeline_dec_tree(dat, loaded_model, myscaler):
    ## using decision Tree as it seems to have best performance
    le = LabelEncoder()
    dat["Race_code"] = le.fit_transform(dat[["Race"]])
    dat["Gender_code"] = le.fit_transform(dat[["Gender"]])
    dat["college_code"] = le.fit_transform(dat[["college"]])
    #make predictions
    #print(dat)
    dat = dat[['SAT Score', 'ACT Score', 'Number of AP Courses',
               'Sports or Other Activities', 'Financial aid needed',
               'GPA', 'Race_code', 'Gender_code', 'college_code']]
    print(dat)
    ypred = loaded_model.predict(myscaler.fit_transform(dat))    
    print("Your chance of admission at ",le.inverse_transform(dat.college_code),"is: ",ypred)     
    return ypred

def initializeModel():
    data_model = pd.read_csv("/data/model_data_clean_2.csv", engine='python', sep=',', quotechar='"', on_bad_lines='warn')
    print(data_model)

    data_model.head()
    data_model.rename(columns={"newScore":"Chance of Admit"},inplace=True)

    # data_model.drop(['totscore'], axis=1, inplace=True)
    # data_model.head()

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
    return scaler


scaler = initializeModel()
firstLine = '"college","SATScore","ACTScore","Race","Gender","NumAP","Sports","finaid","GPA","newScore"'
secondLine= '"{0}",{1},{2},"{3}","{4}",{5},{6},{7},{8},{9}'.format("Alabama A & M University",322,12,"White","Male",5,0,1,3.21,0.0733333333333333)

f = open("/data/tmp.csv", "w")
f.writelines([firstLine,"\n", secondLine, "\n"])
f.close()


new_test_data_2 = pd.read_csv("/data/tmp.csv")
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

filename = '/data/DecisionTreeRegressor_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
coaList = inference_pipeline_dec_tree(new_test_data_2, loaded_model, scaler)

#print("Your chance of admission at ",le.inverse_transform(college_code),"is: ",ypred)    

print("COA is : {0}".format(coaList[0]))












# X_new = [["Alabama A & M University",322,12,"White","Male",5,0,1,3.21,0.0733333333333333]]
# # X_new = scale_fit.transform(X_new) #use the above saved 
# print(X_new)

# new_test_data_2 = pd.read_csv("/data/userInput.csv")
# new_test_data_2.rename(columns={"newScore":"Chance of Admit"},inplace=True)
# ## remove totscore

# #new_test_data_2.drop(['totscore','Chance of Admit'], axis=1, inplace=True)
# dict = {'SATScore': 'SAT Score',
#         'ACTScore': 'ACT Score',
#         'NumAP': 'Number of AP Courses',
#          'Sports':'Sports or Other Activities',
#        'finaid':'Financial aid needed'}

# new_test_data_2.rename(columns=dict,
#           inplace=True)
# new_test_data_2.head()

# le = LabelEncoder()
# new_test_data_2["Race_code"] = le.fit_transform(new_test_data_2[["Race"]])
# new_test_data_2["Gender_code"] = le.fit_transform(new_test_data_2[["Gender"]])
# new_test_data_2["college_code"] = le.fit_transform(new_test_data_2[["college"]])

# new_test_data_2 = new_test_data_2[['SAT Score', 'ACT Score', 'Number of AP Courses',
#             'Sports or Other Activities', 'Financial aid needed',
#             'GPA', 'Race_code', 'Gender_code', 'college_code']]
# print(new_test_data_2)

# X_new = scaler.transform(new_test_data_2) #use the above saved 
# print(X_new)



 


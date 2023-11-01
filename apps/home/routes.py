# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from apps.home import blueprint
from flask import render_template, request, jsonify
from flask_login import login_required
from jinja2 import TemplateNotFound
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
from sklearn.preprocessing import LabelEncoder
import pickle

@blueprint.route('/index', methods=['GET', 'POST'])
@login_required
def index():
    colleges = []
    result_COA = ()
    with open('/data/colleges.txt') as f:
        colleges = f.read().splitlines()


    return render_template('home/index.html', segment='index', colleges = colleges, result_COA=result_COA)

@blueprint.route('/process_coa', methods=['POST', 'GET'])
def process_qt_calculation():
  print("Its a Call")

  if request.method == "POST":
    print("Its a POST Call")

    coa_data = request.get_json()
    colleges = []
    result_COA = ()
    results = calculate_COA(coa_data)
    print(jsonify(results))
    return jsonify(results)


def calculate_COA(coa_data):
    univ = coa_data[0]['univSelect']
    satScore = coa_data[1]['satScore']
    actScore = coa_data[2]['actScore']
    gpa = coa_data[3]['gpa']
    numAPCourses = coa_data[4]['numAPCourses']
    sportExtra = coa_data[5]['sportExtra']
    finAsstNeeded = coa_data[6]['finAsstNeeded']
    sex = coa_data[7]['sex']
    race = coa_data[8]['race']
    predictiveModel = coa_data[9]['predictiveModel']

    print("Printing values in calculate_COA")
    print(coa_data)

    ## Call the ML code to calculate 
    scaler = initializeTrainedModel()

    firstLine = '"college","SATScore","ACTScore","Race","Gender","NumAP","Sports","finaid","GPA","newScore"'
    secondLine= '"{0}",{1},{2},"{3}","{4}",{5},{6},{7},{8},0'.format(univ,satScore,actScore,race,sex,numAPCourses,sportExtra,finAsstNeeded,gpa)
    f = open("/data/tmp.csv", "w")
    f.writelines([firstLine,"\n", secondLine, "\n"])
    f.close()

    new_test_data_2 = pd.read_csv("/data/tmp.csv")
    new_test_data_2.rename(columns={"newScore":"Chance of Admit"},inplace=True)
    dict = {'SATScore': 'SAT Score',
            'ACTScore': 'ACT Score',
            'NumAP': 'Number of AP Courses',
            'Sports':'Sports or Other Activities',
        'finaid':'Financial aid needed'}

    new_test_data_2.rename(columns=dict,
            inplace=True)
    new_test_data_2.head()

    filename = "/data/{0}_model.sav".format(predictiveModel)
    loaded_model = pickle.load(open(filename, 'rb'))
    coaList = inference_pipeline(new_test_data_2, loaded_model, scaler)

    print("COA is : {0}".format(coaList[0]))
    y_new_predict = coaList[0]*100

    if(y_new_predict > 100):
        y_new_predict = 100
    chanceOfAdmission = round(y_new_predict, 2)
    results = {'selectedUniversity': univ, "coa": chanceOfAdmission}

    return results #(univ, chanceOfAdmission)

def inference_pipeline(dat, loaded_model, myscaler):
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

def initializeTrainedModel():
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

def initializeTrainedModelData():
    df_model = pd.read_csv("/data/UG_Agmission_Data.csv")
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

    targets = df_model['Chance of Admit']
    features = df_model.drop(columns = {'Chance of Admit'})

    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    scale_fit = scaler.fit(X_train.values) #save the mean and std. dev computed for your data.
    return scale_fit


@blueprint.route('/<template>')
@login_required
def route_template(template):

    try:

        if not template.endswith('.html'):
            template += '.html'

        # Detect the current page
        segment = get_segment(request)

        # Serve the file (if exists) from app/templates/home/FILE.html
        return render_template("home/" + template, segment=segment)

    except TemplateNotFound:
        return render_template('home/page-404.html'), 404

    except:
        return render_template('home/page-500.html'), 500


# Helper - Extract current page name from request
def get_segment(request):

    try:

        segment = request.path.split('/')[-1]

        if segment == '':
            segment = 'index'

        return segment

    except:
        return None

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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn_pandas import DataFrameMapper
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
    #results = {'selectedUniversity': "SAMPLE UNIVERSITY", "coa": "95.67"}
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


    firstLine = '"college","SATScore","ACTScore","Race","Gender","NumAP","Sports","finaid","GPA","newScore"'
    secondLine= '"{0}",{1},{2},"{3}","{4}",{5},{6},{7},{8},0'.format(univ,satScore,actScore,race,sex,numAPCourses,sportExtra,finAsstNeeded,gpa)
    f = open("/data/tmp.csv", "w")
    f.writelines([firstLine,"\n", secondLine, "\n"])
    f.close()

    new_test_data_2 = pd.read_csv("/data/tmp.csv")
    new_test_data_2.rename(columns=col_name_dict,inplace=True)
    print(new_test_data_2.head())

    filename = "/data/{0}_model.sav".format(predictiveModel)
    loaded_model = pickle.load(open(filename, 'rb'))
    coaList = inference_pipeline(new_test_data_2, preprocessor, loaded_model)
    print("COA is : {0}".format(coaList[0]))

    y_new_predict = coaList[0]*100

    if(y_new_predict > 100):
        y_new_predict = 100
    chanceOfAdmission = round(y_new_predict, 2)
    results = {'selectedUniversity': univ, "coa": chanceOfAdmission}

    return results #(univ, chanceOfAdmission)

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

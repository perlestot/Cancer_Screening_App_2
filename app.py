#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 19:57:40 2020

@author: PLoTAir
"""

import numpy as np
import flask
import pickle
# import tensorflow as tf
# from tensorflow import keras

# app
app = flask.Flask(__name__)

# load model
NN_Cancer_Model = pickle.load(open("NN_Cancer_Screen.pkl","rb"))
# global graph
# graph = tf.get_default_graph()
# NN_Cancer_Model._make_predict_function()

# routes
@app.route("/")
def home():
    return """
           <body> 
           <h1 align=center>Cancer Screening by Invitrace (Well Project)<h1>
           </body>
           <form action="http://localhost:4000/page_CC"">
           <input type="submit" value="Get Start!!" style="margin:0px auto; display:block;"/>
           </form>
           """

@app.route("/predict", methods=["GET"])
def predict():
    thal = flask.request.args["thal"]
    cp = flask.request.args["cp"]
    slope = flask.request.args["slope"]
    exang = flask.request.args["exang"]
    ca = flask.request.args["ca"]
    
    fmap = {"normal": [1, 0, 0],
            "fixed defect": [0, 1, 0],
            "reversable defect": [0, 0, 1],
            "typical angina": [1, 0, 0, 0],
            "atypical angina": [0, 1, 0, 0],
            "non anginal pain": [0, 0, 1, 0],
            "asymptomatic": [0, 0, 0, 1],
            "upsloping": [1, 0, 0],
            "flat": [0, 1, 0],
            "downsloping": [0, 0, 1]}
    
    # X_new = fmap[thal] + fmap[cp] + fmap[slope]
    
    X_new = np.array(fmap[thal] + fmap[cp] + fmap[slope] + [int(exang)] + [int(ca)]).reshape(1, -1)
    yhat = heart.predict(X_new)
    if yhat[0] == 1:
        outcome = "heart disease"
    else:
        outcome = "normal"
    prob = heart.predict_proba(X_new)
    
    return "This patient is diagnosed as " + outcome + " with probability " + str(round(prob[0][1], 2))

@app.route("/page_Demographic")
def page_Demographic():
   with open("page_Demographic.html", 'r') as viz_file:
       return viz_file.read()

@app.route("/page_CC")
def page_CC():
   with open("page_CC.html", 'r') as viz_file:
       return viz_file.read()

@app.route("/result", methods=["GET", "POST"])
def result():
    """Gets prediction using the HTML form"""
    if flask.request.method == "POST":
        inputs = flask.request.form
        AGEDIAG = [int(inputs["Age"])]
        isMale = [int(inputs["Gender"])]
        Fever = [int(inputs["Fever"])]
        N_V = [int(inputs["N_V"])]
        Cough = [int(inputs["Cough"])]
        Anorexia = [int(inputs["Anorexia"])]
        Chest_Pain = [int(inputs["Chest_Pain"])]
        Fatigue = [int(inputs["Fatigue"])]
        Diarrhea = [int(inputs["Diarrhea"])]
        Constipation = [int(inputs["Constipation"])]
        Occult_Blood_Faeces = [int(inputs["Occult_Blood_Faeces"])]
        Abdominal_Pain = [int(inputs["Abdominal_Pain"])]
        Back_Pain = [int(inputs["Back_Pain"])]
        Abnormal_Vaginal_Bleeding = [int(inputs["Abnormal_Vaginal_Bleeding"])]
        Lump = [int(inputs["Lump"])]
        Breast_Skin_Change = [int(inputs["Breast_Skin_Change"])]
        Discharge = [int(inputs["Discharge"])]
        Hematuria = [int(inputs["Hematuria"])]
        Dysuria = [int(inputs["Dysuria"])]
        Hoarseness = [int(inputs["Hoarseness"])]
        Dysphagia = [int(inputs["Dysphagia"])]
        Hemoptysis = [int(inputs["Hemoptysis"])]
        Headache = [int(inputs["Headache"])]
        Dizziness = [int(inputs["Dizziness"])]
        Hip_Pain = [int(inputs["Hip_Pain"])]
        Peri_anal_Symptoms = [int(inputs["Peri_anal_Symptoms"])]
        Breast_Pain = [int(inputs["Breast_Pain"])]
        Jaundice = [int(inputs["Jaundice"])]
        Dyspnea = [int(inputs["Dyspnea"])]
        Amber_Urine = [int(inputs["Amber_Urine"])]
        Paresis = [int(inputs["Paresis"])]


        unk_CC = [0]
        Breast_0 = [0]
        Breast_1 = [0]
        Breast_2 = [0]
        GI_Upper_1 = [0]
        GI_Upper_2 = [0]
        GI_Liver_1 = [0]
        GI_Colorectal_1 = [0]
        GI_Colorectal_2 = [0]
        GI_Colorectal_3 = [0]
        Gynae_Cervical_1 = [0]
        Lung_1 = [0]
        Lung_2 = [0]
        Lung_3 = [0]
    
    X_new = np.array(AGEDIAG +isMale +Fever +N_V +Cough +Anorexia +Chest_Pain +Fatigue +Diarrhea +Constipation +\
                 Occult_Blood_Faeces +Abdominal_Pain +Back_Pain +Abnormal_Vaginal_Bleeding +Lump +\
                 Breast_Skin_Change +Discharge +Hematuria +Dysuria +Hoarseness +Dysphagia +Hemoptysis +\
                 Headache +Dizziness +Hip_Pain +Peri_anal_Symptoms +Breast_Pain +Jaundice +Dyspnea +\
                 Amber_Urine +Paresis +unk_CC +Breast_0 +Breast_1 +Breast_2 +GI_Upper_1 +GI_Upper_2 +\
                 GI_Liver_1 +GI_Colorectal_1 +GI_Colorectal_2 +GI_Colorectal_3 +Gynae_Cervical_1 +\
                 Lung_1 +Lung_2 +Lung_3).reshape(1, -1)

    X_new = X_new.astype(int)
    Class_label = ['Breast', 'Cervical', 'Colorectal', 'Liver', 'Lung']
    # with graph.as_default():
    y_predict_val = NN_Cancer_Model.predict(X_new)

    y_predict_prob = list(map(lambda x: round(x,3),y_predict_val[0]))
    cancer_over_threshold = []
    for prob,class_name in zip(y_predict_prob,Class_label):
        if prob > 0.5:
            cancer_over_threshold.append(class_name)
        if len(cancer_over_threshold):
            text_output = "1"
        else:
            text_output = "0"

              
    return str(X_new) #text_output

if __name__ == '__main__':
    """Connect to Server"""
    HOST = "127.0.0.1"
    PORT = "4000"
    app.run(HOST, PORT, debug=False)

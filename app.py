#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 19:57:40 2020

@author: PLoTAir
"""

import numpy as np
import flask
import pickle
from sklearn.linear_model import LogisticRegression

# app
app = flask.Flask(__name__)

# load model
LogReg_clf = pickle.load(open("LogReg_Cancer_Screen.pkl","rb"))

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
    y_predict_prob = LogReg_clf.predict_proba(X_new).ravel()
    Class_label = LogReg_clf.classes_
    cancer_over_threshold = []
    for prob,class_name in zip(y_predict_prob,Class_label):
        if prob > 0.5:
            cancer_over_threshold.append(class_name)
    if len(cancer_over_threshold):
        text_output = "You have a risk of", *cancer_over_threshold, 'cancer \nWe suggest consulting a doctor immediately.'
    else:
        text_output = "Congratulations, you are not at risk of getting cancer in Top 5 Cancer."

              
    return text_output

if __name__ == '__main__':
    """Connect to Server"""
    HOST = "127.0.0.1"
    PORT = "4000"
    app.run(HOST, PORT, debug=False)

from os import name
import numpy as np
from flask import Flask, request, jsonify, render_template
import urllib.request
from Depression_Detection import app
from tensorflow import keras
from math import expm1
import pandas as pd
import random
app = Flask(__name__)

model = keras.models.load_model('network.h5')


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/story')
def home1():
    return render_template('Story.html')

@app.route('/story2')
def home2():
    return render_template('Story-2.html')

@app.route('/story3')
def home3():
    return render_template('Story-3.html')

@app.route('/story4')
def home4():
    return render_template('Story-4.html')

@app.route('/predict',methods=['POST'])
def predict():
    relationScore = 0
    financeScore = 0
    studyPressureScore = 0
    academicResultScore = 0
    livingPlaceScore = 0
    supportAcademicLifeScore = 0
    socialMediaScore = 0
    mealScore = 0
    healthIssueScore = 0
    hobbyScore = 0
    sleepScore = 0
    predictValues = []
    if request.method == "POST":
        relation = request.form.get('question1')
        
        if relation == 'single':
            relationScore = 0
        elif relation == 'relationship':
            relationScore == 1

        financial = request.form.get('question2')

        if financial == 'yes':
            financeScore = 0
        elif financial == 'no':
            financeScore = 1

        studyPressure = request.form.get('question3')

        if studyPressure == 'yes':
            studyPressureScore = 2
        elif studyPressure == 'no':
            studyPressureScore = 0
        elif studyPressure == 'Not Applicable':
            studyPressureScore = 1


        academicResult = request.form.get('question4')

        if academicResult == 'yes':
            academicResultScore = 2
        elif studyPressure == 'no':
            academicResultScore = 0
        elif studyPressure == 'Not Applicable':
            academicResultScore = 1

        livingPlace = request.form.get('question5')

        if livingPlace == 'yes':
            livingPlaceScore = 2
        elif livingPlace == 'no':
            livingPlaceScore = 0
        elif livingPlace == 'Not Applicable':
            livingPlaceScore = 1

        supportAcademicLife = request.form.get('question6')

        if supportAcademicLife == 'Friends':
            supportAcademicLifeScore = 1
        elif supportAcademicLife == 'Family':
            supportAcademicLifeScore = 0
        elif supportAcademicLife == 'Not Applicable':
            supportAcademicLifeScore = 3
        elif supportAcademicLife == 'No One':
            supportAcademicLifeScore = 2

        socialMedia = request.form.get('question7')

        if socialMedia == 'yes':
            socialMediaScore = 2
        elif socialMedia == 'no':
            socialMediaScore = 0
        elif socialMedia == 'Not Applicable':
            socialMediaScore = 1

        meal = request.form.get('question8')

        if meal == 'yes':
            mealScore = 2
        elif meal == 'no':
            mealScore = 1
        elif meal == 'Neutral':
            mealScore = 0

        healthIssue = request.form.get('question9')

        if healthIssue == 'yes':
            healthIssueScore = 1
        elif healthIssue == 'no':
            healthIssueScore = 0
        

        hobby = request.form.get('question10')

        if hobby == 'yes':
            hobbyScore = 1
        elif hobby == 'no':
            hobbyScore = 0
        

        sleep = request.form.get('sleep')
        sleepScore = sleep

        predictValues.extend([int(sleepScore), relationScore, studyPressureScore, financeScore, academicResultScore, livingPlaceScore, supportAcademicLifeScore, socialMediaScore, mealScore, healthIssueScore, hobbyScore])

        input_array = np.array(predictValues)
        input_array_for_prediction = np.expand_dims(input_array,axis=0)
        answer = model.predict_classes(input_array_for_prediction)
        result = answer[0][0]
        finalAnswer = 0
        if result == 1:
            finalAnswer = random.randrange(70, 90)
        elif result == 0:
            finalAnswer = random.randrange(20,50)

       

       
    return render_template("Suggestions.html", value=finalAnswer)



if __name__=='__main__':
   app.run(debug=True)
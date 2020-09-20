#API file
from flask import Flask,request,render_template,jsonify
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


app=Flask(__name__)

model=pickle.load(open('timble3.pkl','rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction[0]

    return render_template('index.html', prediction_text='The Flower type belogs to {}'.format(output))


if __name__=='__main__':
    app.run(debug=True)
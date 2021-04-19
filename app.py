import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
from model import SentimentRecommentation

app = Flask(__name__)
recommder = SentimentRecommentation()


@app.route('/', methods=['GET', 'POST'])
def home():
    flag = False
    features_data = ""
    if request.method == 'POST':
        flag = True
        user = request.form["user_id"]
        print(user)
        features_data = recommder.recommenderSystem(user)
    return render_template('index.html',data=features_data,flag=flag)


if __name__ =='__main':
    app.run(debug=True)
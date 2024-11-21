from flask import Flask,request,render_template, jsonify
from flask_cors import CORS,cross_origin
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)
app=application

@app.route('/')
@cross_origin()
def index(): 
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
@cross_origin()
def predict_datapoint():
    if request.method=='GET':
        return render_template('index.html')
    else:
        data=CustomData(
            longitude=float(request.form.get('longitude')),
            latitude=float(request.form.get('latitude')),
            housing_median_age=request.form.get('housing_median_age'),
            total_rooms=request.form.get('total_rooms'),
            total_bedrooms=request.form.get('total_bedrooms'),
            population=request.form.get('population'),
            households=request.form.get('households'),
            median_income=float(request.form.get('median_income')),
            ocean_proximity=request.form.get('ocean_proximity'),
        ) 
        pred_df=data.get_data_as_data_frame()
        print(pred_df)        

        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)
        return render_template('index.html',results=results[0])

@app.route('/predictAPI',methods=['POST'])
@cross_origin()
def predict_api():
    if request.method=='GET':
        return render_template('index.html')
    else:
        data=CustomData(
            longitude=float(request.json('longitude')),
            latitude=float(request.json('latitude')),
            housing_median_age=request.json('housing_median_age'),
            total_rooms=request.json('total_rooms'),
            total_bedrooms=request.json('total_bedrooms'),
            population=request.json('population'),
            households=request.json('households'),
            median_income=float(request.json('median_income')),
            ocean_proximity=request.json('ocean_proximity'),
        ) 

        pred_df=data.get_data_as_data_frame()      
        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)

        dct = {'median_house_value':round(results[0],2)}
        return jsonify(dct)   

if __name__=="__main__":
    # app.run(host="0.0.0.0", debug=True) 
    app.run(host="0.0.0.0") 






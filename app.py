from flask import Flask, request, jsonify , render_template
import os
import sys
import pandas as pd
import numpy as np
from src.laptop_price_prediction.pipeline.prediction_pipeline import PredictionPipeline

app = Flask(__name__)

@app.route('/',methods = ['GET'])
def homepage():
    return render_template('index.html')

@app.route('/train',methods=['GET'])  # route to train the pipeline
def training():
    os.system("python main.py")
    return "Training Successful!" 

@app.route("/predict", methods = ['POST','GET'])
def predict():
    if request.method == 'POST':
        try:
            Company = str(request.form.get('Company'))
            TypeName = str(request.form.get('TypeName'))
            Inches = str(request.form.get('Inches'))
            ScreenResolution = str(request.form.get('ScreenResolution'))
            Cpu = str(request.form.get('Cpu'))
            Ram = str(request.form.get('Ram'))
            Memory = str(request.form.get('Memory'))
            Gpu = str(request.form.get('Gpu'))
            OpSys = str(request.form.get('OpSys'))
            Weight = str(request.form.get('Weight'))
            
            data = pd.DataFrame([{"Company": Company,"TypeName": TypeName,"Inches": Inches,
                                  "ScreenResolution": ScreenResolution,"Cpu": Cpu,"Ram": Ram,"Memory": Memory,"Gpu": Gpu,"OpSys": OpSys,"Weight": Weight}])

            pipeline = PredictionPipeline()

            prediction = pipeline.predict(data)

            output = round(prediction[0], 2)

            return render_template("index.html",prediction_text=f"Predicted Laptop Price: ₹ {output}")
            
        except Exception as e:
            return render_template("index.html",prediction_text=f"Error: {str(e)}")   
            
if __name__ == "__main__":
    app.run(debug=True)

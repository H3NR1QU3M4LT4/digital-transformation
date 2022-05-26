"""
This script runs the application using a development server.
It contains the definition of routes and views for the application.
"""

import os
import pickle
import numpy as np
from flask import Flask, flash, request, redirect, url_for, send_from_directory, jsonify
from flask_cors import CORS

PATH_MODEL_WINE_QUALITY = "models/wine_quality_model.sav"
PATH_MODEL_VINES = "models/vines_model.sav"

app = Flask(__name__)
app.config["DEBUG"] = True
CORS(app)

# Make the WSGI interface available at the top level so wfastcgi can get it.
wsgi_app = app.wsgi_app

@app.route("/", methods=["GET"])
def success():
    return "welcome"

@app.route("/predict_wine_quality", methods=["GET", "POST"])
def predict_wine_quality():
    if request.method == "POST":
        auc_dict = {}
        # fixed_acidity 	volatile_acidity 	citric_acid 	residual_sugar 	chlorides 	free_sulfur_dioxide 	total_sulfur_dioxide 	density pH 	sulphates 	alcohol
        model_wine_quality = pickle.load(open(PATH_MODEL_WINE_QUALITY, "rb"))
        data = request.form.to_dict()

        # change the input data to a numpy array
        input_data_as_numpy_array = np.asarray(list(data.values()))
        numeric_array = np.asarray(input_data_as_numpy_array, dtype=float)
        # reshape the numpy array as we are predicting for only on instance
        input_data_reshaped = numeric_array.reshape(1, -1)

        prediction = model_wine_quality.predict(input_data_reshaped)

        return jsonify(int(prediction[0]))

    elif request.method == "GET":
        return "Wine Quality Predictions!"

@app.route("/predict_vines_quality", methods=["GET", "POST"])
def predict_vines_quality():
    if request.method == "POST":
        # temperatura, humidade, intensidade_chuva, intervalo_chuva, total_chuva, velocidade_vento, radiação_solar, sulfur_solo, ph_solo, sulphates_solo, vine_zones
        model_vines_quality = pickle.load(open(PATH_MODEL_VINES, "rb"))
        data = request.form.to_dict()

        # change the input data to a numpy array
        input_data_as_numpy_array = np.asarray(list(data.values()))
        numeric_array = np.asarray(input_data_as_numpy_array, dtype=float)
        # reshape the numpy array as we are predicting for only on instance
        input_data_reshaped = numeric_array.reshape(1, -1)

        prediction = model_vines_quality.predict(input_data_reshaped)

        return jsonify(int(prediction[0]))

    elif request.method == "GET":
        return "Vines Quality Predictions!"

if __name__ == "__main__":
    HOST = os.environ.get("SERVER_HOST", "localhost")
    try:
        PORT = int(os.environ.get("SERVER_PORT", "5555"))
    except ValueError:
        PORT = 5555

    app.run(HOST, PORT)

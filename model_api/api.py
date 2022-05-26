"""
This script runs the application using a development server.
It contains the definition of routes and views for the application.
"""

import os
import random
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

from predicitons import predicitons
from sensors_records import csv_to_json

PATH_MODEL_WINE_QUALITY = "models/wine_quality_model.sav"
PATH_MODEL_VINES = "models/vines_model.sav"

app = Flask(__name__)
CORS(app)

vines_dataset_json, wine_quality_dataset = csv_to_json()


@app.route("/", methods=["GET"])
def success():
    return "welcome"


@app.route("/get_sensor_wine_quality_records", methods=["GET"])
def get_sensor_wine_quality_records():
    return jsonify(random.choices(wine_quality_dataset, k=3))


@app.route("/get_sensor_vines_records", methods=["GET"])
def get_sensor_vines_records():
    return jsonify(random.choices(vines_dataset_json, k=3))


@app.route("/predict_wine_quality", methods=["GET", "POST"])
def predict_wine_quality():
    if request.method == "POST":
        data = request.form.to_dict()

        # fixed_acidity 	volatile_acidity 	citric_acid 	residual_sugar 	chlorides 	free_sulfur_dioxide 	total_sulfur_dioxide 	density pH 	sulphates 	alcohol
        aux_dict = predicitons(PATH_MODEL_WINE_QUALITY, data)

        return jsonify(aux_dict)

    elif request.method == "GET":
        return "Wine Quality Predictions!"
    
@app.route("/simple_post", methods=["GET", "POST"])
@cross_origin()
def simnple_post():
    if request.method == "POST":
        data = request.form['Name']
        print("data", data)
        return data

    elif request.method == "GET":
        return "simple post"


@app.route("/predict_vines_quality", methods=["GET", "POST"])
def predict_vines_quality():
    if request.method == "POST":
        data = request.form.to_dict()
        # temperatura, humidade, intensidade_chuva, intervalo_chuva, total_chuva, velocidade_vento, radiação_solar, sulfur_solo, ph_solo, sulphates_solo, vine_zones
        aux_dict = predicitons(PATH_MODEL_VINES, data)

        return jsonify(aux_dict)

    elif request.method == "GET":
        return "Vines Quality Predictions!"


if __name__ == "__main__":
    HOST = os.environ.get("SERVER_HOST", "localhost")
    try:
        PORT = int(os.environ.get("SERVER_PORT", "5555"))
    except ValueError:
        PORT = 5555

    app.run(HOST, PORT)

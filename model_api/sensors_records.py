import os
import fnmatch
import pandas as pd

PATH_CSVS = "data/"


def csv_to_json():
    for file in os.listdir(PATH_CSVS):
        if fnmatch.fnmatch(file, "*.csv"):
            dir_list = os.listdir(PATH_CSVS)

    csv_files = list(filter(lambda f: f.endswith(".csv"), dir_list))

    data = []
    for csv in csv_files:
        data_store = pd.read_csv(PATH_CSVS + csv, sep=",")
        data.append(pd.DataFrame(data_store))

    vines, wine = data
    vines_dataset_json = vines.to_dict(orient="records")
    wine_quality_dataset = wine.to_dict(orient="records")

    return vines_dataset_json, wine_quality_dataset

import json
import os
import logging

from models.DataInstance import DataInstance

"""
reads json from data file
returns a list containing DataInstance objects
"""


def read_train_json_file(filename):
    logging.info("read_train_json_file :: running")
    if os.path.exists(filename):
        intents = []

        with open(filename, "r") as json_file:
            data = json.load(json_file)

            for k in data.keys():
                intent = data[k]["intent"]
                positions = data[k]["positions"]
                slots = data[k]["slots"]
                text = data[k]["text"]

                temp = DataInstance(k, intent, positions, slots, text)
                intents.append(temp)

        logging.info("read_train_json_file :: complete")

        return intents
    else:
        logging.error("FileNotFoundError :: No file found with that path!")
        raise FileNotFoundError()


"""
read dev.json file and returns an array containing texts
"""


def read_dev_data(file):
    logging.info("read_dev_data :: running")
    texts = []
    with open(file, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)

        for k in data.keys():
            text = data[k]["text"]
            texts.append(text)

    logging.info("read_dev_data :: complete")
    return texts

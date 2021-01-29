import calendar
import time
import json
import logging


# generate an innocent timestamp for file generation
def get_time_stamp():
    ts = calendar.timegm(time.gmtime())
    return ts


# gets slot name from its values
def get_slot_from_word(word, slot_dict):
    for slot_label, value in slot_dict.items():
        if word in value.split():
            return slot_label
    return None


# creates the json file for submission
def create_prediction_file(results):
    logging.info("create_prediction_file :: running")
    results_dict = dict()
    for idx, res in enumerate(results):
        results_dict[str(idx)] = res

    # write json file
    timestamp = get_time_stamp()
    with open(f"../predictions/prediction_{timestamp}.json", "w") as f:
        json.dump(results_dict, f, indent=2)

    logging.info(f"create_prediction_file :: prediction_{timestamp}.json")
    logging.info("create_prediction_file :: complete")

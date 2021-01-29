import json

"""
class DataInstance
Holds the following properties from the train dataset

id : str - id in the json file
intent : str,
positions : object,
slots : object,
text : str
"""


class DataInstance(object):
    def __init__(self, id, intent, positions, slots, text):
        self.id = id
        self.intent = intent
        self.positions = positions
        self.slots = slots
        self.text = text

    def __repr__(self):
        return str(json.dumps(self.__dict__, indent=2))

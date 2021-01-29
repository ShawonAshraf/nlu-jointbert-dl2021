from abc import ABC

import tensorflow as tf
from transformers import TFBertModel
from tensorflow.keras.layers import Dropout, Dense

"""
Classifier model for Joint intent classification and slot filling
default dropout = 0.1
"""


# ABC to get rid of pylint errors
class JointIntentAndSlotFillingModel(tf.keras.Model, ABC):
    def __init__(self, model_name, intent_num_labels, slot_num_labels, dropout_prob=0.1):
        super().__init__(name="joint_intent_slot_filling")
        self.bert = TFBertModel.from_pretrained(model_name)
        self.dropout = Dropout(dropout_prob)
        self.intent_classifier = Dense(intent_num_labels,
                                       name="intent_classifier")
        self.slot_classifier = Dense(slot_num_labels,
                                     name="slot_classifier")

    def call(self, inputs, **kwargs):
        trained_bert = self.bert(inputs, **kwargs)
        pooled_output = trained_bert.pooler_output
        sequence_output = trained_bert.last_hidden_state

        # sequence_output will be used for slot_filling / classification
        sequence_output = self.dropout(sequence_output,
                                       training=kwargs.get("training", False))
        slot_logits = self.slot_classifier(sequence_output)

        # pooled_output for intent classification
        pooled_output = self.dropout(pooled_output,
                                     training=kwargs.get("training", False))
        intent_logits = self.intent_classifier(pooled_output)

        return slot_logits, intent_logits

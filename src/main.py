import tensorflow as tf

from data_loader import read_train_json_file, read_dev_data
from utils import create_prediction_file
from classifier import JointIntentAndSlotFillingModel
from preprocessing import Preprocessor
from model_helper import compile_train_model
from nlu import run_inference

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("gpu_id", help="select the gpu id or enter 0 for default operation")
args = parser.parse_args()

gpu_id = args.gpu_id

# take gpu_id as an arg to run on gpus
# mention a specific gpu_id (starting from 0) for multi gpu setups
# like those at IMS, or pass nothing to use default config

if gpu_id:
    # set gpu
    gpu_id = int(gpu_id)
    gpus = tf.config.experimental.list_physical_devices("GPU")
    tf.config.experimental.set_visible_devices(gpus[gpu_id], "GPU")
    tf.config.experimental.set_memory_growth(gpus[gpu_id], True)
else:
    # do nothing
    pass

# read data
train_data = read_train_json_file("../data/nlu_traindev/train.json")

# pre processing
model_name = "bert-base-cased"  # pretrained bert model to load
preprocessor = Preprocessor(data=train_data, model_name=model_name)

# run preprocessor
preprocessor.run_preprocessor()

encoded_texts = preprocessor.encoded_texts
encoded_intents = preprocessor.encoded_intents
encoded_slots = preprocessor.encoded_slots

intent_map = preprocessor.intent_map
slot_map = preprocessor.slot_map

intent_names = preprocessor.intent_names
slot_names = preprocessor.slot_names

# model
joint_model = JointIntentAndSlotFillingModel(model_name=model_name,
                                             intent_num_labels=len(intent_map),
                                             slot_num_labels=len(slot_map))

# train
x = {"input_ids": encoded_texts["input_ids"],
     "token_type_ids": encoded_texts["token_type_ids"],
     "attention_mask": encoded_texts["attention_mask"]}

# for 2 epochs with batch size 32
# shuffling is set to true in the function
model, history = compile_train_model(model=joint_model, x=x, y=(encoded_slots, encoded_intents),
                                     epochs=2, batch_size=32)

# run inference

# read dev file and predict
dev_texts = read_dev_data("../data/nlu_traindev/dev.json")

# process results
results = run_inference(dev_texts, preprocessor.tokenizer, joint_model, intent_names, slot_names)
create_prediction_file(results)

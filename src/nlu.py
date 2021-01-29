import tensorflow as tf
from tqdm import tqdm
import logging

"""
nlu()
runs inference on a single text instance and returns a dictionary containing the
intent and slots

tokenizer : bert tokenizer
model : trained model
intent_names : list of unique intent names
slot_names: list of unique slot names
"""


def nlu(text, tokenizer, model, intent_names, slot_names):
    inputs = tf.constant(tokenizer.encode(text))[None, :]  # batch_size = 1
    outputs = model(inputs)
    slot_logits, intent_logits = outputs

    # outputs with the highest score
    slot_ids = slot_logits.numpy().argmax(axis=-1)[0, :]
    intent_id = intent_logits.numpy().argmax(axis=-1)[0]

    # return object, a dictionary
    info = {"intent": intent_names[intent_id], "slots": {}}

    # temporary dict for slots
    out_dict = {}
    # get all slot names and add to out_dict as keys, avoid nil
    predicted_slots = set([slot_names[s] for s in slot_ids if s != 0])
    for ps in predicted_slots:
        out_dict[ps] = []

    # check if the text starts with a small letter
    # since the cased model is used, Add and add will generate different tokens
    # and will change the alignment with slot_ids
    if text[0].islower():
        tokens = tokenizer.tokenize(text, add_special_tokens=True)
    else:
        tokens = tokenizer.tokenize(text)

    # align tokens with slot_ids
    for token, slot_id in zip(tokens, slot_ids):
        # add all to out_dict
        slot_name = slot_names[slot_id]

        if slot_name == "<PAD>":
            continue

        # collect tokens
        collected_tokens = [token]
        idx = tokens.index(token)

        # see if the token starts with ##
        # then it belongs to the previous token
        if token.startswith("##"):
            # check if the token already exists or not
            if tokens[idx - 1] not in out_dict[slot_name]:
                collected_tokens.insert(0, tokens[idx - 1])

        # add collected tokens to slots
        out_dict[slot_name].extend(collected_tokens)

    # process out_dict
    for slot_name in out_dict:
        tokens = out_dict[slot_name]
        slot_value = tokenizer.convert_tokens_to_string(tokens)

        # get rid of any lingering empty space
        info["slots"][slot_name] = slot_value.strip()

    return info


"""
calls nlu() on a list of texts, returns a list

tokenizer : bert tokenizer
model : trained model
intent_names : list of unique intent names
slot_names: list of unique slot names
"""


def run_inference(all_texts, tokenizer, model, intent_names, slot_names):
    logging.info("run_inference :: running")
    results = []
    for i in tqdm(range(len(all_texts)), desc="run_inference"):
        text = all_texts[i]
        res = nlu(text, tokenizer, model, intent_names, slot_names)
        results.append(res)
    logging.info("run_inference :: complete")
    return results

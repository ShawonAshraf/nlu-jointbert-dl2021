# nlu-jointbert-dl2021
Repository for the course project of Deep Learning for Speech and Natural Language Processing at Universität Stuttgart. Winter 2020-2021.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ShawonAshraf/nlu-jointbert-dl2021/blob/main/notebooks/nlu_jointbert_dl21.ipynb)


## Task
Natural Language Understanding

## Data
Custom dataset provided by the instructors.

### Dataset properties
````json5
{
  "text": "",
  "positions": [{}],
  "slots": [{}],
  "intent": ""
}
````

## Model Description
__Inputs:__ text

__Labels:__ slots, intents

__Pretrained model in use:__ `bert-base-cased` from Huggingface Transformers.

## ENV Setup
````bash
pip install -r requirements.txt
````

## Sample prediction output

### Input
```
add kansas city, missouri to Stress Relief
```

### Output
```json5
{
  "intent": "AddToPlaylist",
  "slots": {
    "playlist": "Stress Relief",
    "entity_name": "kansas city, missouri"
  }
}
```

### Run
```bash
python main.py <gpu_id>
# use gpu_id from nvidia-smi for multi-gpu systems
# for single gpu, use 0
```

## Reference
1. Chen et al. (2019), BERT for Joint Intent Classification and Slot Filling.
https://arxiv.org/abs/1902.10909

2. https://github.com/monologg/JointBERT


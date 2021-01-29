from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy

import logging

"""
trains the model
the idea is to feed x into the pre trained bert model 
and then fine tune

default learning rate = 3e-5
epsilon = 1e-08
"""


def compile_train_model(model, x, y,
                        epochs, batch_size,
                        learning_rate=3e-5, epsilon=1e-08):
    logging.info("compile_train_model :: running")

    # optimizer -> Adam
    opt = Adam(learning_rate=learning_rate, epsilon=epsilon)

    # two outputs, one for slots, another for intents
    losses = [SparseCategoricalCrossentropy(from_logits=True),
              SparseCategoricalCrossentropy(from_logits=True)]

    # metrics -> accuracy -> requirement for the project
    metrics = [SparseCategoricalAccuracy('accuracy')]

    # compile model
    model.compile(optimizer=opt, loss=losses, metrics=metrics)

    history = model.fit(x, y, epochs=epochs,
                        batch_size=batch_size, shuffle=True,
                        verbose=1)

    logging.info("compile_train_model :: complete")

    return model, history

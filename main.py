import tensorflow as tf
from tensorflow import keras
from PIDOptimizer.pidoptimizer import PIDOptimizer
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import math
import numpy as np
import os


x_train = np.array([1, 2, 3, 4])
x_train = np.expand_dims(x_train, 1)
y_train = np.array([60, 70, 80, 90])
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(1)
keras.backend.clear_session()
np.random.seed(777)
tf.random.set_seed(777)
model = keras.models.Sequential([keras.layers.Dense(1, input_shape=(None, 1))])

optimizer = PIDOptimizer(learning_rate=0.01, kd=0.01)
train_loss = tf.keras.metrics.Mean()


def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x)
        loss = tf.keras.losses.mean_squared_error(y, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    loss = train_loss(loss)

    return loss


EPOCHS = 500
for epoch in range(EPOCHS):

    for x, y in train_ds:
        loss = train_step(x, y)
    if epoch % 50 == 0:
        print("Loss : {0}".format(loss.numpy().item()))

    train_loss.reset_states()


y_pred = model.predict([5])
print("IF x == 5 : {0}".format(y_pred.item()))

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend_config
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops

from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import math
# Common imports
import numpy as np
import os

class PIDOptimizer(keras.optimizers.Optimizer):
    def __init__(self, learning_rate=0.001, momentum= 0.0,
                 kd=None, use_locking=False,
                 name='PIDOptimizer', **kwargs):
        super().__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate)) # handle lr=learning_rate

        if kd is None:
            kd = 0.25 * learning_rate + 0.5 +\
                (1 + math.pi ** 2 * 16 / 9) / learning_rate

        self._lr = learning_rate
        self._momentum = momentum
        self._kd = kd

        self._lr_t = tf.convert_to_tensor(self._lr)
        self._momentum_t = tf.convert_to_tensor(self._momentum)
        self._kd_t = tf.convert_to_tensor(self._kd)
        
    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        for v in var_list:
            self.add_slot(v, "V")
        for v in var_list:
            self.add_slot(v, "D")
        for v in var_list:
            self.add_slot(v, "grad_buf")

    @tf.function
    def _resource_apply_dense(self, grad, var):
        V = self.get_slot(var, 'V')
        D = self.get_slot(var, 'D')
        grad_buf = self.get_slot(var, 'grad_buf')

        V_update = V.assign(self._momentum_t * V - self._lr_t * grad,
                            use_locking=self._use_locking)
        D_update = D.assign(self._momentum_t * D - (1 - self._momentum_t) *
                            (grad - grad_buf), use_locking=self._use_locking)
        grad_buf_update = grad_buf.assign(grad, use_locking=self._use_locking)

        var_update = var.assign_add(V_update + self._kd_t * D_update,use_locking=self._use_locking)

    def _resource_apply_sparse(self, grad, var):
        self._apply_dense(grad, var)

    def get_config(self):
        config = super().get_config()
        config.update({
            **base_config,
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "momentum": self._serialize_hyperparameter("momentum"),
            "kd": self._serialize_hyperparameter("kd"),
            "use_locking": self._serialize_hyperparameter("use_locking"),
        })
        return config

x_train = np.array([1, 2, 3, 4])
x_train = np.expand_dims(x_train, 1)
y_train = np.array([60, 70, 80, 90])
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(4)
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

    for x,y in train_ds:
        loss = train_step(x, y)
    if epoch % 50 == 0:
        print("Loss : {0}".format(loss.numpy().item()))

    train_loss.reset_states()


y_pred = model.predict([5])
print("IF x == 5 : {0}".format(y_pred.item()))

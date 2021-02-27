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
# PID-Optimizer-TF version 2
**[A PID Controller Approach for Stochastic Optimization of Deep Networks](http://www4.comp.polyu.edu.hk/~cslzhang/paper/CVPR18_PID.pdf) For Tensorflow v2**


## Other Implementation
[PyTorch](https://github.com/tensorboy/PIDOptimizer) / [TensorFlow v1](https://github.com/machida-mn/tensorflow-pid)

Pull requests are encouraged and always welcome!

## Example

```python
# pidoptimizer.py

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

    for x,y in train_ds:
        loss = train_step(x, y)
    if epoch % 50 == 0:
        print("Loss : {0}".format(loss.numpy().item()))

    train_loss.reset_states()


y_pred = model.predict([5])
print("IF x == 5 : {0}".format(y_pred.item()))
```

```
Loss : 3055.43115234375
Loss : 90.16002655029297
Loss : 25.60433006286621
Loss : 7.271267414093018
Loss : 2.0649285316467285
Loss : 0.5864140391349792
Loss : 0.1665322184562683
Loss : 0.047291629016399384
Loss : 0.013429916463792324
Loss : 0.0038131256587803364
IF x == 5 : 100.03143310546875
```

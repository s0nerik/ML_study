import tensorflow as tf
import numpy as np

# Hyperparams
learning_rate = 0.001
num_iterations = 10000

# Model parameters
W = tf.Variable([.0], dtype=tf.float32)
b = tf.Variable([.0], dtype=tf.float32)
# Model input and output
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
linear_model = x * W + b
# training data
points = np.genfromtxt("data.csv", delimiter=",")
x_train = [i[0] for i in points]
y_train = [i[1] for i in points]
# loss
error = tf.reduce_sum(((y - linear_model) / len(points)) ** 2)  # sum of the squares
# optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(error)
# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)  # reset values to wrong
for i in range(num_iterations):
    sess.run(optimizer, {x: x_train, y: y_train})

# evaluate training accuracy
curr_W, curr_b, curr_loss = sess.run([W, b, error], {x: x_train, y: y_train})
print("W: %s b: %s loss: %s" % (curr_W, curr_b, curr_loss))

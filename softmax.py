import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

sess = tf.InteractiveSession()

#Begin building the computation graph
#Start with defining nodes for our inputs and outputs

# a 2d tensor of floating points numbers with a shape
# of [None, 784] -- where 784 is the size of a single input
# and None says the number of input examples is unbounded
x = tf.placeholder(tf.float32, shape = [None, 784])

# also a 2d tensor of floating point numbers but with a
# shape of [None, 10] -- where the output is a one-hot
# 10 dimensional vector
y_ = tf.placeholder(tf.float32, shape = [None, 10])

# Next we define our weights `W` and biases `b` for our model.
# We initalize our variables as tensors full of zeros
# W is a 784x10 matrix as we have 784 input features and 10 outputs
# b is a 10-demensional vector because we have 10 output classes
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Tensorflow requires any variables be initalized before use.
sess.run(tf.initialize_all_variables())

# We can now implement out regression model.
# In this example, we'll multiply the vectorized input `x` by
# the weight matrix `W` and the bias `b` and compute the softmax
# probabilities that are assigned to each class.
y = tf.nn.softmax(tf.matmul(x,W) + b)

# Next we can define the cost-function to be minimized during
# training.  In this case we will use the cross-entropy between
# the target and the model's prediction.
j = -tf.reduce_sum(y_*tf.log(y))

# Now we can begin training our model.
# Tensorflow knows the entire computation graph and as such
# it can use automatic differentiation to find the gradients
# of the cost with respect to each of the variables.
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(j)

for i in range(1000):
  batch = mnist.train.next_batch(50)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})

# How did our model do?
# `tf.argmax` gives us the index of the highest entry in a tensor
# along some axis.
prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

# Evaluate the model (should be around 91%)
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

# Notes:
# Softmax - if you want to assign probabilities to an object
# being one of several different things, this is the way to go.


# Making our model better
# This simple example used a softmax regression model with a single linear
# layer.  We can do significantly better by using softmax regression
# with a multilayer convolutional network.

# In this improved model, we'll be creating lots of ReLU neurons with
# lots of weights and biases.  ReLU neurons should be initalized with slightly
# positive values to avoid 'dead' neurons.

# generates random values that follow a normal distribution with a
# specified mean and standard deviation
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# creates a matrix with the specified value that conforms to the specified
# shape.  ex: (0.1, shape=[1,3]) => [0.1, 0.1, 0.1]
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

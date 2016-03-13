import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

sess = tf.InteractiveSession()

n_input = 784
n_hidden = 256
n_output = 10

# helper functions
def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

def mlp_model(input, n_input, n_hidden, n_output):
    # define the input layer weights and bias
    w_h = weight_variable([n_input, n_hidden])
    b_h = bias_variable([n_hidden])

    # define the hidden layer weights and bias
    w_o = weight_variable([n_hidden, n_output])
    b_o = bias_variable([n_output])

    # define mlp model
    step1 = tf.nn.relu(tf.matmul(input, w_h) + b_h)
    step2 = tf.nn.softmax(tf.matmul(step1, w_o) + b_o)
    return step2

def results(y, y_):
    prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
    print(accuracy.eval(
        feed_dict={x: mnist.test.images, y_: mnist.test.labels})
    )

# define input layer
x = tf.placeholder(tf.float32, shape = [None, n_input])

# define output layer
y_ = tf.placeholder(tf.float32, shape = [None, n_output])

# define the model
y = mlp_model(x, n_input, n_hidden, n_output)

# initalize variables
sess.run(tf.initialize_all_variables())

# define the cost
j = -tf.reduce_sum(y_*tf.log(y))

# define optimizer
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(j)

for i in range(1000):
  batch = mnist.train.next_batch(50)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})

results(y, y_)

sess.close()

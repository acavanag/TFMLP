import tensorflow as tf

# Generate XOR training data

train_inputs = [[0,0],[0,1],[1,0],[1,1]]
train_outputs = [[0],[1],[1],[0]]

# Build learning model

model_path = "/Users/andrewcavanagh/Documents/Projects/TensorFlow/xor.ckpt"
sess = tf.InteractiveSession()

n_input = 2
n_hidden = 4
n_output = 1

def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

def mlp_model(input, n_input, n_hidden, n_output):
    w_h = weight_variable([n_input, n_hidden])
    b_h = bias_variable([n_hidden])
    w_o = weight_variable([n_hidden, n_output])
    b_o = bias_variable([n_output])
    step1 = tf.nn.tanh(tf.matmul(input, w_h) + b_h)
    step2 = tf.nn.tanh(tf.matmul(step1, w_o) + b_o)
    return step2

x = tf.placeholder(tf.float32, shape = [None, n_input])
y_ = tf.placeholder(tf.float32, shape = [None, n_output])
y = mlp_model(x, n_input, n_hidden, n_output)

sess.run(tf.initialize_all_variables())

j = tf.reduce_mean(( (y_ * tf.log(y)) + ((1 - y_) * tf.log(1.0 - y)) ) * -1)
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(j)

saver = tf.train.Saver()

# Train model
# for i in range(100000):
#     train_step.run(feed_dict={x: train_inputs, y_: train_outputs})
#
# save_path = saver.save(sess, model_path)
# print("Model saved in file: %s" % save_path)

saver.restore(sess, model_path)
print sess.run(y, feed_dict={x: [[1,0],[0,1],[1,1],[0,0]]})

sess.close()

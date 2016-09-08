from tensorflow.exampls.tutorials.minst import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
'''
Each MNIST data point has two parts, an image and a label. Images are 
"x" and labels "y."

Training images will be "mnist.train.images" and labels will be 
"mnist.train.labels"

Each image is 28x28 pixels, represented by a matrix of numbers from 0 to
1. This is strung out to a vector for analysis.

Softmax Regression - If you want to assign probabilities to an object 
being one of several different things, softmax is the thing to do. 
Softmax will give a list of values between 0 and 1 which sum to 1. 
'''

import tensorflow as tf
'''
Create symbolic variable x, a 2D tensor of floating point numbers.
It will take any number of images. It has a shape of (any amount by 
784)
'''
x = tf.placeholder(tf.float32, [None, 784])

'''
Create the variables that can be used and modified by the computation.
These will be the weights and biases. W's shape is 784x10 so it can be 
multiplied by the 784x1 image vector, resulting in a 10x1 vector. b is 
a 10x1 vector so it can simply be added as an offset (bias). 
'''
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

'''
Implementation of the model. 
'''
y = tf.nn.softmax(tf.matmul(x, W) + b)

'''
Implementation of cross-entropy.
'''
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

'''
Training step. Asking tensorflow to minimize cross_entropy using the 
gradient descent algo with a learning rate of 0.5. This simply shifts 
the variable a bit in the direction that reduces the entropy. Other 
optimization algorithms could be swapped into this line.
'''
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)
'''
For each step of the loop we are taking a batch of 100 random data 
points from the training set. We run train_step to feed in the batches 
data to replace the placeholders. Small batches like this is called 
stochastic training. Specifically in this case "Stochastic gradient 
descent." 
'''
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

'''
Evaluation model
'''
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

import tensorflow as tf

import numpy as np
from PIL import Image

n_output = 10
output_layer = tf.matmul(layer_3, weights['out']) + biases['out']

biases = {
    'out': tf.Variable(tf.constant(0.1, shape=[n_output]))
}

sess = tf.Session()
new_saver = tf.train.import_meta_graph('tensorflow-model-1000.meta')
new_saver.restore(sess, tf.train.latest_checkpoint('./'))

img = np.invert(Image.open("./images/mnist_first_digit.png").convert('L')).ravel()

prediction = sess.run(tf.argmax(output_layer,1), feed_dict={X: [img]})
print ("Prediction for test image:", np.squeeze(prediction))

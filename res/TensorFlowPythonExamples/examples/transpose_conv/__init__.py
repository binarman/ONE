import tensorflow as tf
import numpy as np

in_ = tf.compat.v1.placeholder(tf.float32, shape=(1, 8, 8, 1), name="Hole")

filters = tf.compat.v1.constant(np.random.uniform(low = -1., high = 1, size=(2,3,4)), dtype=tf.float32)
output_shape = [1,5,5,2]
strides = (2, 2)

op_ = tf.compat.v1.nn.conv2d_transpose(in_, filters, output_shape, strides, "VALID")

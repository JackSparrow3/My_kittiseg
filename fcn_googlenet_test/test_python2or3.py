import numpy as np
import tensorflow as tf


# [batch, height, width, depth]
x_image = tf.placeholder(tf.float32,shape=[3,2])
x = tf.reshape(x_image,[1,3,2,1])

#Filter: W  [kernel_height, kernel_width, output_depth, input_depth]
# W_cpu = np.array([[1,1,1,1]],dtype=np.float32)
W_cpu = np.random.rand(4)
W_cpu = np.float32(W_cpu)

W = tf.Variable(W_cpu)
W = tf.reshape(W, [1,4,1,1])

strides=[1, 1, 1, 1]
padding='VALID'

y = tf.nn.conv2d_transpose(x, W, [1,4,3,1],strides, padding)
# y_ = tf.nn.conv2d(x,W,strides=strides,padding=padding)
x_data = np.array([[1,-1],[2,2],[1,2]],dtype=np.float64)
with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)

    x = (sess.run(x, feed_dict={x_image: x_data}))
    W = (sess.run(W, feed_dict={x_image: x_data}))
    y = (sess.run(y, feed_dict={x_image: x_data}))
    # y_=(sess.run(y_,feed_dict={x_image: x_data}))
    print "The shape of x:\t", x.shape, ",\t and the x.reshape(3,2) is :"
    print x.reshape(3,2)
    print ""

    print "The shape of x:\t", W.shape, ",\t and the W.reshape(1,2) is :"
    print W.reshape(1,2)
    print ""

    print "The shape of y:\t", y.shape, ",\t and the y.reshape(3,3) is :"
    print y.reshape(3,3)
    print ""

    # print "The shape of y:\t", y_.shape, ",\t and the y.reshape(3,3) is :"

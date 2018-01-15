import tensorflow as tf
import tensorflow.contrib.slim as slim
wei=tf.get_variable('fuse_1_conv_0_weight',[3,3,3,3],dtype=tf.float32,)
x=tf.initializer
x1=tf.constant(1.0,shape=[1,3,3,1])
kernel=tf.constant(1.0,shape=[2,2,3,1])
kernel1=tf.constant(1.0,shape=[2,2,3,3])
x2=tf.constant(1.0,shape=[1,6,6,3])
x3=tf.constant(1.0,shape=[1,154,154,3])

y2=tf.nn.max_pool(x3,[1,2,2,1],[1,2,2,1],'SAME')
# y2=tf.nn.conv2d(x3,kernel,strides=[1,2,2,1],padding='VALID')
y3=tf.nn.conv2d_transpose(y2,kernel1,output_shape=[1,154,154,3],strides=[1,2,2,1],padding='SAME')
y4=tf.nn.conv2d(x2,kernel,strides=[1,2,2,1],padding='SAME')
with tf.Session() as sess:
	init=tf.global_variables_initializer()
	sess.run(init)
	# sess.run(y3)
	a=y2.get_shape()
	print (a)

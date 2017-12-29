import tensorflow as tf

path = '/home/yu/projects/KittiSeg/inception/inception_v3.ckpt'

# sess = tf.Session()
# init = tf.global_variables_initializer()
# sess.run(init)

with tf.Session() as sess:
# saver=tf.train.Saver()
    saver = tf.train.import_meta_graph(sess,path)
#     saver.restore(sess,path)
    print (sess.run(tf.get_default_graph().get_operations()))

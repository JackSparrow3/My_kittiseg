import tensorflow as tf
import cv2
import inception.inception_v3
import numpy as np


img=cv2.imread('/home/yu/projects/KittiSeg/DATA/data_road/training/image_2/um_000000.png')
img=cv2.resize(img,(1248,384))
img=np.expand_dims(img,0)
# print img.shape
# cv2.imshow('1',img)
# cv2.waitKey(0)
img_input=tf.placeholder(tf.float32,shape=[1,None,None,3])
_,logits,end_point=inception.inception_v3.inception_v3_fcn(img_input)

sess=tf.Session()
init=tf.global_variables_initializer()
sess.run(init)
saver=tf.train.Saver()
saver.restore(sess,"/home/yu/projects/KittiSeg/RUNS/InceptionV3_dropout=0.8_ap=93.5/model.ckpt-92000")
output,endpoint=sess.run([logits,end_point],feed_dict={img_input:img})
for i in endpoint:
    shape=endpoint[i].shape
    print (i)
    print (shape)
print img.shape

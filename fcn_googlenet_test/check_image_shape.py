import scipy as scp
import os
import cv2
import tensorflow as tf
from inception import inception_v3_same
#TODO check the image shape in the train set
# image_dir='/home/yu/projects/KittiSeg/DATA/data_road'
# image_dic={}
#
# with open('/home/yu/projects/KittiSeg/DATA/data_road/train3.txt') as file:
#     for i, datum in enumerate(file):
#         datum = datum.rstrip()
#         image_file, gt_file = datum.split(" ")
#         image_file = os.path.join(image_dir, image_file)
#         gt_file = os.path.join(image_dir, gt_file)
#         image = cv2.imread(image_file)
#         shape=str(image.shape)
#         if shape in image_dic:
#             image_dic[shape] += 1
#
#         elif shape  not in image_dic:
#             image_dic[shape] = 1
#
#             print shape,image_file
#         elif image_dic == None:
#             image_dic[shape] = 1
#         else:
#             break
#
#
#     print image_dic

image_input=tf.placeholder(tf.float32,[1,375,1242,3])
image=cv2.imread('/home/yu/projects/KittiSeg/DATA/data_road/training/image_2/um_000000.png')
sess = tf.Session()
_,net,end_points = inception_v3_same.inception_v3_fcn(inputs=image_input)
init = tf.global_variables_initializer()
sess.run(init)

saver=tf.train.Saver()
saver.restore(sess,'/home/yu/projects/KittiSeg/inception/inception_v3.ckpt')

sess.run([net,end_points],feed_dict={image_input:image})
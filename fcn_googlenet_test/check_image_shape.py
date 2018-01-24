import scipy as scp
import os
import sys
import cv2
import tensorflow as tf
import inception.inception_v4 as inceptionv4
import ResNet.resnet_v2 as resnet
import tensorflow.contrib.slim as slim

import incl.tensorflow_fcn.fcn8_vgg as vgg
# sys.path.append(sys.path.append('/home/yu/projects/FCN_GoogLeNet'))
# import inception_v3_fcn
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


image_input=tf.placeholder(tf.float32,[1,384,1248,3])
image=cv2.imread('/home/yu/projects/KittiSeg/DATA/data_road/training/image_2/um_000000.png')
# gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
# vgg_fcn = vgg.FCN8VGG(vgg16_npy_path='/home/yu/projects/KittiSeg/DATA/vgg16.npy')
# vgg_fcn.build(image_input, train=False, num_classes=2, random_init_fc8=True)

with slim.arg_scope(inceptionv4.inception_arg_scope()):
    net,end_points = inceptionv4.inception_v4(image_input,2,False)
# with slim.arg_scope(resnet.resnet_arg_scope()):
#     net,end_points = resnet.resnet_v2_101(image_input, 2, False, False)
for i in end_points:
    print i
init = tf.global_variables_initializer()
sess.run(init)

saver=tf.train.Saver()
saver.restore(sess,'/home/yu/projects/KittiSeg/inception/inception_v3.ckpt')

sess.run([net,end_points],feed_dict={image_input:image})
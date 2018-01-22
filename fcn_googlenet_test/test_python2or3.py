import tensorflow as tf
import cv2
img=cv2.imread('../RUNS/InceptionV3_u_net_bn_f1=96.2182000.png')
shape=img.shape
print shape
cv2.imshow('1',img)
cv2.waitKey(1)
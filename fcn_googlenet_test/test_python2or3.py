import cv2
import tensorflow as tf
img=cv2.imread('/home/yu/projects/KittiSeg/RUNS/V3_refinev1_f1=96.7182000.png',-1)
print img.shape
img=cv2.resize(img,(1240,384))
print img.shape
tf.sequence_mask()
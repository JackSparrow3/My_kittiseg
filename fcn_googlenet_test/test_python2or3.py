import cv2
import tensorflow as tf
img=cv2.imread('/home/yu/projects/KittiSeg/DATA/data_road/gt_image_3/um_road_000060.png',-1)
print img.shape
# img=cv2.resize(img,(1240,384))
# print img.shape
# cv2.imshow('1',img)
# cv2.waitKey(1000)
color=cv2.applyColorMap(img,cv2.COLORMAP_JET)
cv2.imshow('2',color)
cv2.imwrite('score_umm_000060.png',color)
cv2.waitKey(0)
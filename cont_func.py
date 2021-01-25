import numpy as np
import cv2


def cont_func(ker_size,orig_pic,cur_bw):
    kernel = np.ones((ker_size+7, ker_size+7), np.uint8)
    erosion = cv2.erode(cur_bw.astype('uint8'), kernel, iterations=1)
    kernel = np.ones((ker_size+9, ker_size+9), np.uint8)
    dilated_head = cv2.dilate(erosion, kernel, iterations=1)
    kernel_er = np.ones((3, 3), np.uint8)
    for i in range(3):
        dilated_head = cv2.dilate(dilated_head, kernel_er, iterations=1)*cur_bw
    kernel_dil = np.ones((6, 6), np.uint8)
    cur_tail = cur_bw - dilated_head
    cur_tail = cv2.erode(cur_tail, kernel_er, iterations=1)     #to get rid of some extras from the head
    cur_tail = cv2.dilate(cur_tail, kernel_dil, iterations=2)   #to go back to having an overlappingwithhead tail
    if np.mean(cur_tail)<1e-10:
        return [dilated_head]
    return cur_tail,dilated_head




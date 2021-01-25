import numpy as np
import cv2
import matplotlib.pyplot as plt

def find_ker_size(bw_pic,show=False):
    ker_size = 6
    diffs = {}
    if show:
        plt.figure()
        plt.imshow(255*bw_pic)
        plt.title('bw')
        plt.figure()
    for i in range(10):
        ker = np.ones((ker_size,ker_size),np.uint8)
        img_opened = cv2.morphologyEx(bw_pic.astype('uint8'), cv2.MORPH_OPEN, ker)
        diffs[ker_size]=np.mean(img_opened==bw_pic)
        ker_size+=2
        if show:
            plt.subplot(2,5,i+1)
            plt.imshow(255*img_opened)
            plt.title(ker_size-2)
    all_diffs = np.array(list(diffs.values()))
    all_diffs_moved_over = np.zeros_like(all_diffs)
    all_diffs_moved_over[:-1]=all_diffs[1:]
    grad_diffs = abs(all_diffs-all_diffs_moved_over)
    loc = np.argmax(grad_diffs[:-1])
    if show:
        plt.show()
    return 8 + loc*2
import cv2
def preprocess(pic,size=9):
    proc_pic = cv2.GaussianBlur(pic,(size,size),0)
    return proc_pic
import numpy as np
import cv2
import math

def calc_dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def most_dist_points(xs, ys,return_centroid=False):
    x0 = np.mean(xs)
    y0 = np.mean(ys)
    biggest_dist = 0
    for i in range(len(xs)):
        dist = calc_dist([x0, y0], [xs[i], ys[i]])
        if dist > biggest_dist:
            biggest_dist = dist
            pt = [xs[i], ys[i]]

    biggest_dist = 0
    for i in range(len(xs)):
        dist = calc_dist(pt, [xs[i], ys[i]])
        if dist > biggest_dist:
            biggest_dist = dist
            pt_2 = [xs[i], ys[i]]
    if return_centroid:
        centroid = [x0,y0]
        return pt,pt_2,dist,centroid
    return pt, pt_2, dist

def finding_largest_region(bw_img):
    c_output = cv2.connectedComponentsWithStats(bw_img.astype('uint8'))
    stats = c_output[2]
    biggest_label = np.argwhere(stats[:, 4] == np.unique(stats[:, 4])[-2])[0][0]
    return c_output[1] == biggest_label

def centeroidnp(arr):
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return sum_x/length, sum_y/length

def calc_dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def calc_most_dist_pt(pt_0,pts):
    num_pts = np.shape(pts)[1]
    pt_0 = np.hstack([pt_0] * num_pts)
    # pt_0 = np.tile(pt_0, (num_pts, 1))

    dist_calc = (pts-pt_0)**2
    dist = np.sum(dist_calc,axis=0)
    loc_pt = np.argmax(dist)
    return pts[:,loc_pt]

def angle_to_rotate(centr,centc,pt1r,pt1c):
    dx = centc-pt1c
    dy = -(centr-pt1r)
    alpha = 180*(np.arctan(dy/dx))/np.pi
    if dx>0:
        return 180-alpha
    else:
        return -alpha

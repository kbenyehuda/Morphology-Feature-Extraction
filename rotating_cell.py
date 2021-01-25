import copy
from PIL import Image, ImageDraw
from ex_funcs import *
from preprocess import *


def leave_pt_of_tail(pt_0,pt_1,cur_pic):
    [y_max,x_max] = np.where(cur_pic==np.max(cur_pic))
    y_max = y_max[0]
    x_max = x_max[0]
    d0 = calc_dist([x_max,y_max],pt_0)
    d1 = calc_dist([x_max,y_max],pt_1)
    if d1>d0:
        [cent_c,cent_r]=pt_1
        [pt_1_c,pt_1_r]=pt_0
    else:
        [cent_c,cent_r] = pt_0
        [pt_1_c,pt_1_r] = pt_1

    return [cent_c,cent_r],[pt_1_c,pt_1_r]


def rotating_cell_notail(cur_pic,opened_bw,return_head_only=True,draw_pts=False):
    [ys,xs] = np.where(opened_bw==1)
    pt_0,pt_1,dist,cent = most_dist_points(xs,ys,return_centroid=True)
    [cent_c,cent_r],[pt_1_c,pt_1_r] = leave_pt_of_tail(pt_0,pt_1,cur_pic)

    kernel = np.ones((3,3),np.uint8)
    for i in range(20):
        opened_bw = cv2.morphologyEx(opened_bw.astype('uint8'), cv2.MORPH_DILATE, kernel,iterations=1)
    head_pic = cur_pic * opened_bw

    # rotating image
    if return_head_only:
        pil_img = Image.fromarray(head_pic.astype('uint8'))
    else:
        pil_img = Image.fromarray(cur_pic.astype('uint8'))
    pil_img = pil_img.convert('RGB')
    pil_img_clean = copy.copy(pil_img)
    if draw_pts:
        draw = ImageDraw.Draw(pil_img)
        draw.rectangle([cent_c - 3, (cent_r) - 3, cent_c + 3, (cent_r) + 3], fill='red', outline=1, width=1)
        draw.rectangle([pt_1_c - 3, (pt_1_r) - 3, pt_1_c + 3, (pt_1_r) + 3], fill='green', outline=1, width=1)
    angle = angle_to_rotate(cent_r, cent_c, pt_1_r, pt_1_c)
    pil_img = pil_img.rotate(angle, expand=1)
    pil_img_clean = pil_img_clean.rotate(angle, expand=1)
    pil_img_clean_array = np.array(pil_img_clean)
    pil_img_array = np.array(pil_img)

    # moving the cell to the left of the image
    [red_ys,red_xs] = np.where(pil_img_array[:,:,0]==255)
    [ys, xs] = np.where(pil_img_array[:,:,0] != 0)
    left_x = np.min(red_xs) + 10
    right_x = np.max(xs)
    dist_x = right_x-left_x
    up_y = np.min(ys)
    down_y = np.max(ys)
    dist_y = down_y-up_y
    half_dist_y = int(dist_y/2)
    new_img = np.zeros_like(pil_img_array)
    mid_y = int(pil_img_array.shape[0]/2)
    new_img[mid_y-half_dist_y:mid_y+half_dist_y,:dist_x]=pil_img_clean_array[up_y:up_y + 2*half_dist_y,left_x:right_x]
    new_img = preprocess(new_img,5)
    new_img = new_img[mid_y-75:mid_y+75,:150]
    return new_img


def rotating_cell(cur_tail,cur_pic,opened_bw,return_head_only = True,draw_pts=False):
    tail_plus_head = opened_bw + cur_tail
    pts_in_both = np.where(tail_plus_head == 2)


    cent_r, cent_c = centeroidnp(np.transpose(pts_in_both))
    cent_r = round(cent_r)
    cent_c = round(cent_c)

    # get other dis pt
    pt_0 = np.array([cent_r, cent_c])[:, np.newaxis]
    all_head_pts_r, all_head_pts_c = np.where(opened_bw == 1)
    all_head_pts = np.vstack((all_head_pts_r, all_head_pts_c))

    pt_1_r, pt_1_c = calc_most_dist_pt(pt_0, all_head_pts)

    kernel = np.ones((3, 3), np.uint8)
    for i in range(20):
        opened_bw = cv2.morphologyEx(opened_bw.astype('uint8'), cv2.MORPH_DILATE, kernel, iterations=1)
    head_pic = cur_pic * opened_bw

    # rotating image
    if return_head_only:
        pil_img = Image.fromarray(head_pic.astype('uint8'))
    else:
        pil_img = Image.fromarray(cur_pic.astype('uint8'))
    pil_img = pil_img.convert('RGB')
    pil_img_clean = copy.copy(pil_img)
    if draw_pts:
        draw = ImageDraw.Draw(pil_img)
        draw.rectangle([cent_c - 3, (cent_r) - 3, cent_c + 3, (cent_r) + 3], fill='red', outline=1, width=1)
        draw.rectangle([pt_1_c - 3, (pt_1_r) - 3, pt_1_c + 3, (pt_1_r) + 3], fill='green', outline=1, width=1)
    angle = angle_to_rotate(cent_r, cent_c, pt_1_r, pt_1_c)
    pil_img = pil_img.rotate(angle, expand=1)
    pil_img_array = np.array(pil_img)

    pil_img_clean = pil_img_clean.rotate(angle,expand=1)
    pil_img_clean_array = np.array(pil_img_clean)

    [red_ys, red_xs] = np.where(pil_img_array[:,:,0] == 255)
    left_x = np.min(red_xs) + 10
    all_head_pts_r, all_head_pts_c = np.where(pil_img_array[:, :, 0] != 0)
    right_x = np.max(all_head_pts_c)
    dist_x = right_x-left_x
    up_y = np.min(all_head_pts_r)
    down_y = np.max(all_head_pts_r)
    dist_y = down_y-up_y
    half_dist_y = int(dist_y/2)
    new_img = np.zeros_like(pil_img_array)
    mid_y = int(pil_img_array.shape[0]/2)
    new_img[mid_y-half_dist_y:mid_y+half_dist_y,:dist_x]=pil_img_clean_array[up_y:up_y+2*half_dist_y,left_x:right_x]
    new_img = preprocess(new_img,5)
    new_img = new_img[mid_y - 75:mid_y + 75, :150]

    return new_img
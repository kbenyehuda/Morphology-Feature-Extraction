from find_ker_size import *
from cont_func import *
from nucleus_acrosome import *
from rotating_cell import *


def get_head_nuc_acr_max1(pic):
    pic = np.squeeze(pic[:, :, 0])
    pic = preprocess(pic)
    pic = 255 * (pic / np.max(pic))
    bw_pic = pic > 30

    # taking just biggest region
    bw_pic = finding_largest_region(bw_pic)
    kernel = np.ones((7, 7))
    bw_pic = cv2.morphologyEx(bw_pic.astype('uint8'), cv2.MORPH_DILATE, kernel)

    # finding kernel size for getting rid of tail
    ker_size = find_ker_size(bw_pic)

    # getting either tail and head or just head
    cont_func_output = cont_func(ker_size, pic, bw_pic)

    # getting final roated head in correct position
    if len(cont_func_output) < 2:
        dilated_head = cont_func_output[0]
        rotated_cell = rotating_cell_notail(pic, dilated_head, draw_pts=True)

    else:
        cur_tail = cont_func_output[0]
        try:
            cur_tail = finding_largest_region(cur_tail)
        except:
            print('got error')
            print('avg cur_tail = ', np.mean(cur_tail))
        dilated_head = cont_func_output[1]
        rotated_cell = rotating_cell(cur_tail, pic, dilated_head, draw_pts=True)

    # filter by head area
    head_area_in_pixels = np.sum(rotated_cell > 35)
    if head_area_in_pixels > 7000 and head_area_in_pixels < 13000:
        nuc, acr, nuc_acr_pic = find_nucleus_and_acrosome(rotated_cell, show=False, orig_pic=pic, return_pic=True)
        acr = np.array(acr) * 1 - np.array(nuc) * 1 > 0
        [ys_nuc, xs_nuc] = np.where(nuc > 0)

        # deleting some more tail if it exists
        new_rotated_head = np.zeros_like(rotated_cell)
        left_nuc_x = np.min(xs_nuc) - 10
        if left_nuc_x > 0:
            rest_x = rotated_cell.shape[1] - left_nuc_x
            new_rotated_head[:, :rest_x] = rotated_cell[:, left_nuc_x:left_nuc_x + rest_x]
            rotated_cell = new_rotated_head

    return rotated_cell, nuc,acr
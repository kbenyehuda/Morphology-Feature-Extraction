import pickle
import pandas as pd
import numpy as np


def get_beta(curfilename):

    don_ind = curfilename.index('donor ')
    don_ind2 = curfilename.index('video_')
    don_num = curfilename[don_ind+6:don_ind2-1]

    vid_ind2 = curfilename.index('C_')
    vid_num = curfilename[don_ind2+6:vid_ind2-1]

    cut_filename = curfilename[vid_ind2:]
    frame_ind = cut_filename.index('\\')
    frame_ind2 = cut_filename.index('.png')
    frame_num = cut_filename[frame_ind+1:frame_ind2]

    mat_struct_filename = 'struct_d_'+don_num+'.xlsx'
    cur_key = 'd'+don_num+'v'+vid_num+'f'+frame_num

    cur_max_value = get_value(mat_struct_filename,cur_key)

    return cur_max_value/255,cur_key,don_num,vid_num
    # return cur_max_value*max1/(255**2),cur_key,don_num


def get_value(xl_filename,frame_name):
    try:
        data = pd.read_excel(xl_filename)
        return float(data[frame_name])
    except KeyError:
        print('key ',frame_name,' not in ',xl_filename)
        return np.nan

from getting_beta import get_beta
from features_funcs import get_all_feats
from get_head_acr_nuc import *
from glob import glob
from numba import prange
from TC import *



base_dir = 'D:/Users/Keren/Documents/university/Year 2/DNA Fragmentation/Acridine Orange/Motility/new_videos/final'

donors = [302,305,306,309,310,311,312,314]
cell_names_to_write = []
data_to_write = []
for d_i in prange(len(donors)):
    cell_names_to_write.append([])
    data_to_write.append([])
    cur_donor = donors[d_i]
    video_folders = glob(base_dir+'/donor '+str(cur_donor)+'/video_*')
    for v_i in prange(len(video_folders)):
        vid_folder = video_folders[v_i]
        vid_ind = vid_folder.index('video_')
        vid_num = int(vid_folder[vid_ind+6:])

        cells_folders = glob(vid_folder+'/C_*')
        for c_i in prange(len(cells_folders)):
            cell_folder = cells_folders[c_i]
            print(cell_folder)
            cell_ind = cell_folder.index('C_')
            cell_num = int(cell_folder[cell_ind+2:])

            cell_pics = glob(cell_folder+'/*.png')
            for f_i in prange(len(cell_pics)):
                curfilename = cell_pics[f_i]
                frame_ind = curfilename.index(str(cell_num))
                frame_num = curfilename[frame_ind+len(str(cell_num))+1:-4]

                try:
                    cell_pic = cv2.imread(curfilename)
                    cell_head, nuc, acr = get_head_nuc_acr_max1(cell_pic)
                    cur_beta, cur_key, don_num, vid_num = get_beta(curfilename)
                    c_area, nuc_area, acr_area, mean_post_ant_diff, mean_opd, var_opd, drymass = get_all_feats(
                        np.squeeze(cell_head[:, :, 0]), nuc, acr, cur_beta)

                    cell_ind1 = curfilename.index('C_')
                    cut_filename = curfilename[cell_ind1:]
                    cell_ind2 = cut_filename.index('\\')
                    cell_name = cut_filename[:cell_ind2]

                    cur_ind = donors.index(int(don_num))
                    final_cell_name = cur_key + cell_name

                    cell_names_to_write[cur_ind].append(final_cell_name)
                    cur_list = [c_area, nuc_area, acr_area, mean_post_ant_diff, mean_opd, var_opd, drymass]
                    data_to_write[cur_ind].append(cur_list)
                    # print(cur_list)
                except:
                    print('didnt create a row for ', curfilename[-40:])
                    continue

            # with pd.ExcelWriter('final_exp_3.xlsx') as writer:
            #     columns = ['Total Head Area','Nucleus Area','Acrosome Area','Mean Post Ant Diff','Mean OPD','Var OPD','Drymass']
            #     for i in range(len(donors)):
            #         if len(cell_names_to_write[i])>0:
            #             print(i)
            #             cur_data = np.array(data_to_write[i])
            #             cur_names = cell_names_to_write[i]
            #             df = pd.DataFrame(data=cur_data,index=cur_names,columns=columns)
            #             sheet_name = 'donor '+str(donors[i])
            #             df.to_excel(writer, sheet_name=sheet_name)





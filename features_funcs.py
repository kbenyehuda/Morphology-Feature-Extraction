import numpy as np

def get_total_cell_pic_area(cell_pic,cell_pic_th=35):
    # return area in um^2
    # 2.4 um pixel / 39 system magnification
    # cell_pic is not binary
    return np.sum(cell_pic>cell_pic_th)*((2.4/39)**2)

def get_nucleus_or_acrosome_area(nuc_or_acr,beta):
    # return area in um^2
    # 2.4 um pixel / 39 system magnification
    # nuc_or_acr is binary
    return np.sum(nuc_or_acr)*((2.4/39)**2)

def get_mean_post_ant_diff(cell_pic,beta,bg_th=35):
    [ys,xs] = np.where(cell_pic>bg_th)
    left_x = np.min(xs)
    right_x = np.max(xs)
    mid_x = int(np.mean([left_x,right_x]))
    posterior = cell_pic[:,:mid_x]
    anterior = cell_pic[:,mid_x:]
    post_mean_opd = get_mean_opd(posterior,beta,bg_th)
    ant_mean_opd = get_mean_opd(anterior,beta,bg_th)
    return post_mean_opd - ant_mean_opd



def get_mean_opd(cell_pic,beta,bg_th=35):
    #pm
    # area_of_pix = ((2.4*1000)**2)/39
    return (beta*np.sum(cell_pic[cell_pic>bg_th])/np.sum(cell_pic>bg_th))

def get_variance_opd(cell_pic,beta,bg_th=35):
    #/1000
    return ((beta*np.std(cell_pic[cell_pic>bg_th]))**2)/1000

def get_drymass(cell_pic,cell_pic_area,beta):
    #cell_pic_area in um^2
    #cell_pic is the OPD of the cell_pic in nm
    #return in pm
    pixel_area = (2.4**2)/39
    alpha = 1.8*(10**5) #[um^3/ug] constant based on research
    opd_um = cell_pic/1000
    drymass = np.sum(opd_um*pixel_area/alpha)
    return drymass*1000

def get_all_feats(cell_pic,nuc,acr,beta):
    c_area = get_total_cell_pic_area(cell_pic)
    nuc_area = get_nucleus_or_acrosome_area(nuc,beta)
    acr_area = get_nucleus_or_acrosome_area(acr,beta)
    mean_post_ant_diff = get_mean_post_ant_diff(cell_pic,beta)
    mean_opd = get_mean_opd(cell_pic,beta)
    var_opd = get_variance_opd(cell_pic,beta)
    drymass = get_drymass(cell_pic,c_area,beta)

    return c_area,nuc_area,acr_area,mean_post_ant_diff,mean_opd,var_opd,drymass


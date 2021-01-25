import matplotlib.pyplot as plt
import copy
from ex_funcs import *
import cv2


def find_right_region(blobs):
    num_regions, regions, stats, centroids = cv2.connectedComponentsWithStats(blobs.astype(np.uint8))
    right_blob_val = np.argmax(centroids[1:,0])+1
    return regions==right_blob_val,np.max(centroids[1:,0])

def creating_color_image(color,img):
  r = color[0]
  r_img = r*img
  g = color[1]
  g_img = g*img
  b = color[2]
  b_img = b*img
  rgb_color = np.concatenate((r_img[:,:,np.newaxis],g_img[:,:,np.newaxis],b_img[:,:,np.newaxis]),axis=2)
  return rgb_color

def draw_perimeter(output_img,object,color=[255, 0, 0]):

    kernel = np.ones((3, 3), np.uint8)
    obj_eroded = cv2.morphologyEx(object.astype('uint8'), cv2.MORPH_ERODE, kernel, iterations=1)
    obj_outline_2d = object - obj_eroded
    obj_outline_rgb = creating_color_image(color, obj_outline_2d)
    output_img[obj_outline_rgb > 0] = obj_outline_rgb[obj_outline_rgb > 0]
    return output_img

def find_nucleus_and_acrosome(cell,nuc_color=[255,0,0],acr_color = [0,255,0],show=False,bg_th=35 ,orig_pic = None, return_pic = False):
    unique_labels = np.unique(cell)
    t_nuc = 0.6*np.mean(unique_labels[-5:])
    nucleus = np.zeros((cell.shape[0],cell.shape[1]))
    nucleus[cell[:,:,0] > t_nuc] = 1
    nucleus = finding_largest_region(nucleus)

    acrosome = np.zeros_like(nucleus)
    acrosome[np.logical_and(cell[:,:,0]<t_nuc , cell[:,:,0]>bg_th)]=1

    ker_size = 11
    kernel = np.ones((ker_size, ker_size), np.uint8)
    acr_eroded = cv2.morphologyEx(acrosome.astype('uint8'), cv2.MORPH_ERODE, kernel, iterations=1)
    try:
        acr_eroded,left_x = find_right_region(acr_eroded)
    except:
        [ys,xs] = np.where(acr_eroded!=0)
        left_x = np.min(xs)

    dil_kernel = np.ones((3, 3), np.uint8)
    acr_dilated = cv2.morphologyEx(acr_eroded.astype('uint8'), cv2.MORPH_DILATE, dil_kernel, iterations=1)
    left_most_x = left_x

    for i in range(3):
        acr_dilated = cv2.morphologyEx(acr_dilated.astype('uint8'), cv2.MORPH_DILATE, dil_kernel, iterations=1)
        acr_times_cell = acr_dilated * (np.array(cell[:,:,0]>bg_th*1))

    acr_minus_nuc = np.array(acr_times_cell)*1-np.array(nucleus)*1
    acrosome_final = acr_minus_nuc>0

    for i in range(5):
        nucleus = cv2.morphologyEx(nucleus.astype('uint8'), cv2.MORPH_DILATE, dil_kernel, iterations=1) * (cell[:,:,0]>bg_th)
    nucleus = np.array(nucleus)*1 - np.array(acrosome_final)*1
    nucleus = nucleus > 0

    if show or return_pic:
        output_img = copy.copy(cell)
        output_img = draw_perimeter(output_img,nucleus,nuc_color)
        output_img = draw_perimeter(output_img,acrosome_final,acr_color)
        if show:
            plt.subplot(1,2,1)
            plt.imshow(orig_pic,cmap='gray')
            plt.title('orig cropped cell')
            plt.subplot(1,2,2)
            plt.imshow(output_img)
            plt.title('rotated with nucleus and acrosome')
            plt.show()
        else:
            return nucleus,acrosome_final,output_img
    return nucleus,acrosome_final

def finding_threshold(cls_1,show=False,num_ths=1,initial_guesses=None,max_its=None):
  combined_array = cls_1
  percentile = 100*(1/(num_ths+1))
  percentiles = [int(np.ceil(percentile*(i))) for i in range(num_ths+2)]
  old_threshs = [np.percentile(combined_array,i) for i in percentiles]
  if initial_guesses:
    old_threshs = initial_guesses

  my_eps = 0.001
  eps=200
  i=0
  if not max_its:
    max_its = 100000
  while eps>my_eps and i<max_its:
    i+=1
    inds = [np.where(np.logical_and(combined_array>old_threshs[i],combined_array<old_threshs[i+1])) for i in range(num_ths+1)]
    sums = [np.shape(inds[i])[1] for i in range(len(inds))]
    avgs = [np.mean(combined_array[inds[i]]) for i in range(len(inds))]
    try:
      new_threshs = [np.average([avgs[i],avgs[i+1]],weights=[sums[i],sums[i+1]]) for i in range(num_ths)]
    except ZeroDivisionError:
      return old_threshs
    new_threshs.insert(0,np.min(combined_array))
    new_threshs.append(np.max(combined_array))
    eps = np.mean(abs(np.array(new_threshs)-np.array(old_threshs)))
    old_threshs = new_threshs
  if show:
    plt.imshow(segment_pic(cls_1,new_thresh))
    plt.show()
  return new_threshs


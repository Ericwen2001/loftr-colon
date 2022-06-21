import cv2
import random
import numpy as np
from numpy.linalg import inv
from matplotlib import pyplot as plt
import os
import imageio
import pdb



### This function is provided by Mez Gebre's repository "deep_homography_estimation"
#   https://github.com/mez/deep_homography_estimation
#   Dataset_Generation_Visualization.ipynb

def create_gif(image_list, gif_name, duration=0.35):
    frames = []
    for image_name in image_list:
        frames.append(image_name)
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
    return

def ImagePreProcessing(image, path):
    img = cv2.imread(path + '/%s' % image, 0)
    img = cv2.resize(img, (700, 700))

    rho = 50
    patch_size = 500
    top_point = (50, 50)
    left_point = (patch_size + 50, 50)
    bottom_point = (patch_size + 50, patch_size + 50)
    right_point = (50, patch_size + 50)
    test_image = img.copy()
    four_points = [top_point, left_point, bottom_point, right_point]

    perturbed_four_points = []
    for point in four_points:
        perturbed_four_points.append((point[0] + random.randint(-rho, rho), point[1] + random.randint(-rho, rho)))

    H = cv2.getPerspectiveTransform(np.float32(four_points), np.float32(perturbed_four_points))
    H_inverse = inv(H)

    warped_image = cv2.warpPerspective(img, H_inverse, (700,700))

    annotated_warp_image = warped_image.copy()

    Ip1 = test_image[top_point[1]:bottom_point[1], top_point[0]:bottom_point[0]]
    Ip2 = warped_image[top_point[1]:bottom_point[1], top_point[0]:bottom_point[0]]
    #x_warp = cv2.warpPerspective(Ip1, H_inverse, (Ip2.shape[1], Ip2.shape[0]))
    image_list = (img, warped_image)
    duration = 0.35
    #cv2.imwrite('C:/Users/PanTianbo/Desktop/img1.png', img)
    #cv2.imwrite('C:/Users/PanTianbo/Desktop/img2.png', warped_image)
    #gif_name = 'C:/Users/PanTianbo/Desktop/hhhhhh.gif'
    #create_gif(image_list, gif_name, duration)
    training_image = np.dstack((Ip1, Ip2))
    H_four_points = np.subtract(np.array(perturbed_four_points), np.array(four_points))
    #datum = (training_image, H_four_points)
    data ={
        'image0': Ip1,
        'Homography0': H,
        'image1': Ip2,
        'Homography1': H_inverse

    }
    return img, warped_image, H_inverse


# save .npy files
def savedata(path):
    lst = os.listdir(path)
    os.makedirs(path + '_processed/')
    new_path = path + '_processed/'
    #pdb.set_trace()
    for i in lst:
        image, warped_image, H_inverse = ImagePreProcessing(i, path)
        os.makedirs(path + '_processed/' + '%s' % i.split('.')[0])
        save_path = path + '_processed/' + '%s' % i.split('.')[0]
        cv2.imwrite(save_path + '/img.png', image)
        cv2.imwrite(save_path + '/warped_img.png', warped_image)
        np.savetxt(save_path + '/gt_homo.txt', H_inverse.reshape(9,1))

if __name__ == '__main__':
    train_path = 'C:/datasets/synapse/syntheticColon_I/train'
    savedata(train_path)


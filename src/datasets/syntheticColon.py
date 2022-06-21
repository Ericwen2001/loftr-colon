from PIL import Image, ImageOps
from os import path as osp
from typing import Dict
from unicodedata import name
import cv2
import os
import numpy as np
import torch
import torch.utils as utils
import pdb
from numpy.linalg import inv
from src.utils.dataset import (
    read_syntheticColon_depth
)

import numpy as np
from scipy.spatial.transform import Rotation as R


def get_poses(scene, root):
    """
    :param scene: Index of trajectory
    :param root: Root folder of dataset
    :return: all camera poses as quaternion vector and 4x4 projection matrix
    """
    locations = []
    rotations = []
    loc_reader = open(root + 'SavedPosition_' + scene + '.txt', 'r')
    rot_reader = open(root + 'SavedRotationQuaternion_' + scene + '.txt', 'r')
    for line in loc_reader:
        locations.append(list(map(float, line.split())))

    for line in rot_reader:
        rotations.append(list(map(float, line.split())))

    locations = np.array(locations)
    rotations = np.array(rotations)
    poses = np.concatenate([locations, rotations], 1)

    r = R.from_quat(rotations).as_matrix()

    TM = np.eye(4)
    TM[1, 1] = -1

    poses_mat = []
    for i in range(locations.shape[0]):
        ri = r[i]
        Pi = np.concatenate((ri, locations[i].reshape((3, 1))), 1)
        Pi = np.concatenate((Pi, np.array([0.0, 0.0, 0.0, 1.0]).reshape((1, 4))), 0)
        Pi_left = TM @ Pi @ TM   # Translate between left and right handed systems
        poses_mat.append(Pi_left)

    return poses, np.array(poses_mat)


def get_relative_pose(pose_t0, pose_t1):

    """
    :param pose_tx: 4x4 camera pose describing camera to world frame projection of camera x.
    :return: Position of camera 1's origin in camera 0's frame.
    """
    return np.matmul(np.linalg.inv(pose_t0), pose_t1)

class ColonDataset(utils.data.Dataset):
    def __init__(self,
                 root_dir,
                 scene,
                 length,
                 mode='train',
                 augment_fn=None,
                 homo_dir=None,
                 **kwargs):
        """Manage one scene of ScanNet Dataset.
        Args:
            root_dir (str): ScanNet root directory that contains scene folders.
            npz_path (str): {scene_id}.npz path. This contains image pair information of a scene.
            intrinsic_path (str): path to depth-camera intrinsic file.
            mode (str): options are ['train', 'val', 'test'].
            augment_fn (callable, optional): augments images with pre-defined visual effects.
            pose_dir (str): ScanNet root directory that contains all poses.
                (we use a separate (optional) pose_dir since we store images and poses separately.)
        """
        super().__init__()
        self.root_dir = root_dir
        self.scene = scene
        self.homo_dir = homo_dir if homo_dir is not None else root_dir
        self.mode = mode
        self.length = length

        # prepare data_names, intrinsics and extrinsics(T)


        # for training LoFTR
        self.augment_fn = augment_fn if mode == 'train' else None

    def __len__(self):
        return self.length

    # def __read_gt_homo__(self, scene_name, gt_homo):
    #     pth = osp.join(os.readlink(osp.join(self.root_dir, scene_name)), f'{gt_homo}.txt')
    #     return read_homo(pth)

    def __getitem__(self, idx):

        ratio = 640/475
        # read the grayscale image which will be resized to (1, 640, 640)
        img_name0 = osp.join(self.root_dir,'Frames_'+self.scene+"/FrameBuffer_"+f"{idx:04}"+".png")
        img_name1 = osp.join(self.root_dir,'Frames_'+self.scene+"/FrameBuffer_"+f"{(idx+1):04}"+".png")
        # TODO: Support augmentation & handle seeds for each worker correctly.
        print("!!!!!!!!!!!"+img_name0)
        image0 = Image.open(img_name0)
        image0 = ImageOps.grayscale(image0)
        image0 = cv2.resize(np.array(image0),(640, 640)).astype('f')
     
        #    augment_fn=np.random.choice([self.augment_fn, None], p=[0.5, 0.5]))
        image1 = Image.open(img_name1)
        image1 = ImageOps.grayscale(image1)
        image1 = cv2.resize(np.array(image1),(640, 640)).astype('f')
        #    augment_fn=np.random.choice([self.augment_fn, None], p=[0.5, 0.5]))

        _, poses = get_poses(self.scene,self.root_dir)

        T_0to1 =  torch.tensor(get_relative_pose(poses[idx+1],poses[idx]).astype('f'))
        T_1to0 = T_0to1.inverse()
        K_0 = K_1 = np.array([[227.60416*ratio, 0, 227.60416*ratio],[0,237.5*ratio,237.5*ratio], [0,0,1]],dtype='f')
        depth_name0 = osp.join(self.root_dir,'Frames_'+self.scene,"Depth_"+f"{idx:04}"+".png")
        depth_name1 = osp.join(self.root_dir,'Frames_'+self.scene,"Depth_"+f"{(idx+1):04}"+".png")

        depth0 = read_syntheticColon_depth(depth_name0,(640, 640))
  
        depth1 = read_syntheticColon_depth(depth_name1,(640, 640))

        #T_0to1 = torch.from_numpy(T_0to1).cuda()

        data = {
            'image0': image0[np.newaxis,:],  # (1, h, w)
            'depth0': depth0, 
            'image1': image1[np.newaxis,:],
            'depth1': depth1, 
            'dataset_name': 'Colon',
            'scene_id': self.scene,
            'pair_id': idx,
            'pair_names': (osp.join(self.scene, f'{idx}.jpg'),
                           osp.join(self.scene, f'{idx+1}.jpg')),
            'T_0to1': T_0to1,   # (4, 4)
            'T_1to0': T_1to0,
            'K0': K_0,  # (3, 3)
            'K1': K_1,
        }

        return data

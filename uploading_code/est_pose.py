from scipy.spatial.transform import Rotation as R 
import torch
import numpy as np
from scipy.spatial import KDTree
import torch.nn.functional as F
import cv2
from new_transformer import ViTModel


def estimate_pose(kpts0, kpts1, conf=0.99999, flag = True):
    if flag: 
        if len(kpts0) < 5:
            return None
        # normalize keypoints
        K0 = np.array([[1000, 0, 500],
                 [0, 1000, 400],
                 [0, 0, 1]])
        K1 = np.array([[1000, 0, 500],
                 [0, 1000, 400],
                 [0, 0, 1]])
        kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
        kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]
    
    
        # compute pose with cv2
        E, mask = cv2.findEssentialMat(
            kpts0, kpts1, np.eye(3), prob=conf, method=cv2.RANSAC)
        if E is None:
            print("\nE is None while trying to recover pose.\n")
            return None
    
        # recover pose from E
        best_num_inliers = 0
        ret = None
        for _E in np.split(E, len(E) / 3):
            n, R, t, _ = cv2.recoverPose(_E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
            if n > best_num_inliers:
                ret = (R, t[:, 0])
                best_num_inliers = n
                return ret
    else: 
        model = ViTModel().cuda().eval()
        state_dict = torch.load("save56.pt")
        for k in list(state_dict.keys()):
                    if k.startswith('module.'):
                        state_dict[k.replace('module.', '', 1)] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        calc_pose = model(mkpts0, mkpts1, feats0, feats1)
        return calc_pose.cpu().numpy()
        
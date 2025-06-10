from scipy.spatial.transform import Rotation as R 
import torch
import numpy as np
from scipy.spatial import KDTree
import torch.nn.functional as F
import cv2
# from new_transformer import ViTModel



def rot_to_quat(R):
    if len(R.shape) == 2:
        w = 0.5 * torch.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2])
        
        x = 0.5 * torch.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
        y = 0.5 * torch.sqrt(1 - R[0, 0] + R[1, 1] - R[2, 2])
        z = 0.5 * torch.sqrt(1 - R[0, 0] - R[1, 1] + R[2, 2])
        
        
        x *= torch.sign(x * (R[2, 1] - R[1, 2]))
        y *= torch.sign(y * (R[0, 2] - R[2, 0]))
        z *= torch.sign(z * (R[1, 0] - R[0, 1]))
        
        
        q = torch.stack([w, x, y, z], dim=0)
    else:
        n = R.shape[0] 
        q = torch.zeros((n, 4))
        for i in range(n):
            
            w = 0.5 * torch.sqrt(1 + R[i, 0, 0] + R[i, 1, 1] + R[i, 2, 2] + 1e-6)
            
            x = 0.5 * torch.sqrt(1 + R[i, 0, 0] - R[i, 1, 1] - R[i, 2, 2] + 1e-6)
            y = 0.5 * torch.sqrt(1 - R[i, 0, 0] + R[i, 1, 1] - R[i, 2, 2] + 1e-6)
            z = 0.5 * torch.sqrt(1 - R[i, 0, 0] - R[i, 1, 1] + R[i, 2, 2] + 1e-6)
            
            # print("w, x, y, z: ", w, x, y, z)
            # print("R: ", R)
            x_new = x * torch.sign(x * (R[i, 2, 1] - R[i, 1, 2])) / w
            
            y_new = y * torch.sign(y * (R[i, 0, 2] - R[i, 2, 0])) / w
            z_new = z * torch.sign(z * (R[i, 1, 0] - R[i, 0, 1])) / w
            
            
            q[i] = torch.stack([torch.tensor(1.0).cuda(), x_new, y_new, z_new], dim=0)
    return q


@torch.no_grad()
def quat_to_rot(quat):
    R = torch.zeros(quat.shape[0], 3, 3)
    for i in range(quat.shape[0]):
        q = quat[i]
        w, x, y, z = q[0], q[1], q[2], q[3]
        r = torch.stack([
            1 - 2 * y**2 - 2 * z**2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w,
            2 * x * y + 2 * z * w, 1 - 2 * x**2 - 2 * z**2, 2 * y * z - 2 * x * w,
            2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x**2 - 2 * y**2
        ], dim=-1).reshape(3, 3)
        R[i, :, :] = r
    return R
    
def undistort(xy, k):
    fx, fy = k[0, 0], k[1, 1]
    cx, cy = k[:2, 2]
    x, y = xy.astype(float)
    x = (x - cx) / fx
    y = (y - cy) / fy
    
def calc_depth(left_points, right_points, f, b):
    disparity = abs(left_points[:, 0] - right_points[:, 0]) + 1e-5
    depth = f * b/disparity
    return depth

def project_to_3D(left_points, depth, P2_inv): 
    # print(left_points.get_device())
    homo_points = torch.cat([left_points, torch.ones(left_points.size(0), 1).cuda()], dim=1)
    # print("homo points: ", homo_points.shape)
    homo_points = homo_points * depth.view(-1, 1)
    world_points = torch.matmul(P2_inv.float(), homo_points.float().t()).t()
    return world_points
    
def get_world_points(left_points, right_points, f, b, P2_inv):
    depth = calc_depth(left_points, right_points, f, b)
    world_points = project_to_3D(left_points, depth, P2_inv)
    return world_points
    
    
def calc_depth_from_cloud(left_points, pointcloud, K):
    Tr_ve_to_cam = np.array([4.276802385584e-04, -9.999672484946e-01, -8.084491683471e-03, -1.198459927713e-02, -7.210626507497e-03, 8.081198471645e-03, -9.999413164504e-01, -5.403984729748e-02, 9.999738645903e-01, 4.859485810390e-04, -7.206933692422e-03, -2.921968648686e-01]).reshape(3, 4)
    R = Tr_ve_to_cam[:, :3]
    t = Tr_ve_to_cam[:, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    Tr_cam_to_velo = np.eye(4)
    Tr_cam_to_velo[:3, :3] = R_inv
    Tr_cam_to_velo[:3, 3] = t_inv
    xyz_points = pointcloud[:, :3].cpu().numpy()
    kdtree = KDTree(xyz_points)
    depth_guesses = np.arange(-79.0, 78.0, 0.1)
    depth_values = []
    for (u, v) in left_points:
        depths = []
        for depth in depth_guesses:
            x = (u - K[0, 2]) * depth / K[0, 0]
            y = (v - K[1, 2]) * depth / K[1, 1]
            camera_point = np.array([float(x), float(y), depth, 1])
            point_cloud_coordinates = Tr_cam_to_velo @ camera_point
            X_velo, Y_velo, Z_velo = point_cloud_coordinates[:3]

            # check kd tree
            dist, idx = kdtree.query([X_velo, Y_velo, Z_velo], k=5)  
            # calc weighted depth
            if dist[0] < 10:  
                weights = 1 / (dist + 1e-6)  
                weighted_depth = np.average(xyz_points[idx, 2], weights=weights)
                weighted_depth = np.random.random(1)[0] * 100
                depths.append(weighted_depth)
        # depths.append(np.random.random(1)[0] * 100)
        
        
        # 
        if depths:
            depth_values.append(np.mean(depths))
        else:
            depth_values.append(0)
    return torch.from_numpy(np.array(depth_values)).cuda()

def new_get_world_points(left_points, velodyne_pointclouds, K, P2_inv):
    depth = calc_depth_from_cloud(left_points, velodyne_pointclouds, K)
    world_points = project_to_3D(left_points, depth, P2_inv)
    return world_points

def gram_schmidt(vectors):
    u = torch.zeros_like(vectors)
    for i in range(vectors.shape[0]):
        b1 = vectors[i][:3]  # first base
        b2 = vectors[i][3:]  # second base
        
        u1 = b1 / torch.norm(b1)
        
        proj_u1_b2 = torch.dot(u1, b2) * u1
        u2 = b2 - proj_u1_b2
        u2 = u2 / torch.norm(u2) if torch.norm(u2) != 0 else u2
        
        u[i][:3] = u1
        u[i][3:] = u2

    return u


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    d6 = gram_schmidt(d6)
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def diffuse_features(points, feats, total_feats, channel=64, img_shape=(480, 640)):
    b, n, _ = points.shape
    h, w = img_shape
    # expanded_f = torch.ones((b, channel, h, w), device=feats.device)

    x_coords = points[..., 0].long()
    y_coords = points[..., 1].long()
    
    expanded_x = x_coords + (x_coords < w - 1).long()
    expanded_y = y_coords + (y_coords < h - 1).long()
    
    # weight0 = torch.zeros((b, n, 1), device=feats.device)
    weight0 = (points[..., 0] - x_coords.float()) + (points[..., 1] - y_coords.float())
    
    # weight1 = torch.zeros((b, n, 1), device=feats.device)
    weight1 = (x_coords.float() + 1 - points[..., 0]) + (y_coords.float() + 1 - points[..., 1])


    expanded_f = total_feats[torch.arange(b).view(-1, 1), :, y_coords, x_coords] + (feats * weight0.unsqueeze(-1))
    expanded_f += total_feats[torch.arange(b).view(-1, 1), :, expanded_y, expanded_x] + (feats * weight1.unsqueeze(-1))    

    return expanded_f
    

def isRotationMatrix(M):
    I = np.identity(M.shape[0])
    return np.linalg.norm(I - np.matmul(M, M.T)) < 1e-6
    

    
def compute_relative_pose(poses):
    n = poses.shape[0]
    relative_poses = np.zeros((n - 1, 12)) 

    for i in range(1, n):
        my_prev_pose = poses[i-1].reshape((3, 4))
        R_prev = my_prev_pose[:, :3]
        t_prev = my_prev_pose[:, 3:]
        my_curr_pose = poses[i].reshape((3, 4))
        R_curr = my_curr_pose[:, :3]
        t_curr = my_curr_pose[:, 3:]

        R_rel = np.linalg.inv(R_prev) @ R_curr
        t_rel = np.linalg.inv(R_prev) @ (t_curr - t_prev)

        relative_poses[i - 1][:9] = R_rel.flatten()
        relative_poses[i - 1][9:12] = t_rel.flatten()
    return relative_poses

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
        
        return None
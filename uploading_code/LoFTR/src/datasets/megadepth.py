import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from loguru import logger

from src.utils.dataset import read_megadepth_gray, read_megadepth_depth

pair_infos = []

def check_overlap(i0, j0, i1, j1):
    if i0 == i1:
        return j0, i0, j1
    elif i0 == j1:
        return j0, i0, i1
    elif j0 == i1:
        return i0, j0, j1
    elif j0 == j1:
        return i0, j0, i1
    return None
    
def generate_indices(k):
    for i in range(k):
        for j in range(i):
            yield (i, j)

def calculate_final_result(number):
    # print(number)
    # i = comb0[number]
    # j = comb1[number]
    i, j = number
    pair_info_1 = pair_infos[i]
    pair_info_2 = pair_infos[j]
    
    [i0, j0] = pair_info_1[0]
    [i1, j1] = pair_info_2[0]
    overlap_1 = pair_info_1[1]
    overlap_2 = pair_info_2[1]
    
    k = check_overlap(i0, j0, i1, j1)
    if k:
        new_pair_info = np.array([k[0], k[1], k[2]])
        new_overlap = min(overlap_1, overlap_2)
        
        return np.array([new_pair_info, new_overlap, pair_info_1[2]], dtype=object)


class MegaDepthDataset(Dataset):
    def __init__(self,
                 root_dir,
                 npz_path,
                 mode='train',
                 min_overlap_score=0.4,
                 img_resize=None,
                 df=None,
                 img_padding=False,
                 depth_padding=False,
                 augment_fn=None,
                 fp16=False,
                 **kwargs):
        """
        Manage one scene(npz_path) of MegaDepth dataset.
        
        Args:
            root_dir (str): megadepth root directory that has `phoenix`.
            npz_path (str): {scene_id}.npz path. This contains image pair information of a scene.
            mode (str): options are ['train', 'val', 'test']
            min_overlap_score (float): how much a pair should have in common. In range of [0, 1]. Set to 0 when testing.
            img_resize (int, optional): the longer edge of resized images. None for no resize. 640 is recommended.
                                        This is useful during training with batches and testing with memory intensive algorithms.
            df (int, optional): image size division factor. NOTE: this will change the final image size after img_resize.
            img_padding (bool): If set to 'True', zero-pad the image to squared size. This is useful during training.
            depth_padding (bool): If set to 'True', zero-pad depthmap to (2000, 2000). This is useful during training.
            augment_fn (callable, optional): augments images with pre-defined visual effects.
        """
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode
        self.scene_id = npz_path.split('.')[0]

        # prepare scene_info and pair_info
        if mode == 'test' and min_overlap_score != 0:
            logger.warning("You are using `min_overlap_score`!=0 in test mode. Set to 0.")
            min_overlap_score = 0
        '''
        print(npz_path)
        new_file = npz_path.replace("3pairs", "3pairs_limited")
        if osp.isfile(new_file): 
            return
        '''
        self.scene_info = dict(np.load(npz_path, allow_pickle=True))
        self.pair_infos = self.scene_info['pair_infos'].copy()
        
        del self.scene_info['pair_infos']
        self.pair_infos = [pair_info for pair_info in self.pair_infos if pair_info[1] > min_overlap_score]
        # print("oh?")
        # too_large = ["/temp/megadepth_indices/scene_info_0.1_0.7_3pairs/0004_0.1_0.3.npz", "/temp/megadepth_indices/scene_info_0.1_0.7_3pairs/0080_0.1_0.3.npz", "/temp/megadepth_indices/scene_info_0.1_0.7_3pairs/0331_0.1_0.3.npz", "/temp/megadepth_indices/scene_info_0.1_0.7_3pairs/0036_0.1_0.3.npz"]
        # sample_size = 500000
        # new_file = npz_path.replace("3pairs", "3pairs_limited")
        # if len(self.pair_infos) > 500000:
            # import random
            # self.pair_infos = random.sample(self.pair_infos, sample_size)
        # self.scene_info['pair_infos'] = self.pair_infos
            
        # np.savez(new_file, **self.scene_info)
        '''
        if len(self.pair_infos) > 300000:
        print(len(self.pair_infos))
        print(new_file)
        '''
        # if new_file == "/temp/megadepth_indices/scene_info_0.1_0.7_3pairs/0411_0.1_0.3.npz":
        
        # if len(self.pair_infos) < 10000:
        # if True:
        '''
        if not osp.isfile(new_file):
            if new_file in too_large:
                return
            else: 
                print(new_file)
            self.scene_info = dict(np.load(new_file, allow_pickle=True))
            self.pair_infos = self.scene_info['pair_infos'].copy()
            self.pair_infos = [pair_info for pair_info in self.pair_infos if pair_info[1] > min_overlap_score]
            data = self.__getitem__(7)
            import pdb; pdb.set_trace()
        
            new_pair_infos = []
    
            global pair_infos
            pair_infos = self.pair_infos
            
            

            print("start counting: ")
            # global comb0, comb1 
            # comb0, comb1 = np.triu_indices(len(self.pair_infos), k=1)
            # print("how fast is this?")
            import time
            from functools import partial
            # my_func = partial(calculate_final_result, comb0, comb1, pair_infos, check_overlap)
            
            import multiprocessing
            
            
            # if len(comb0) > 10000000:
            if False:
                print("too much")
                total_pair_infos = []
                start_time = time.time()
                for i in range(0, len(comb0), 5000000):
                    start_point = i
                    end_point = min(i + 5000000, len(comb0))
                    # print(start_point, end_point)
                    with multiprocessing.Pool(processes=6) as pool:
                        new_pair_infos = pool.map(calculate_final_result, range(start_point, end_point))
                    new_pair_infos = [p for p in new_pair_infos if p is not None]
                    total_pair_infos.extend(new_pair_infos)
                    end_time = time.time()
                    print(f"cut multi:  {end_time - start_time:.6f}")
                new_pair_infos = np.array(total_pair_infos)
                
                    
            else: 
                print("not so much")
                total_pair_infos = []
                start_time = time.time()
                gen = generate_indices(len(self.pair_infos))
                batch = []
                with multiprocessing.Pool(processes=6) as pool:
                    for value in gen:
                        batch.append(value)
                        if len(batch) == 10000000:
                            new_pair_infos = pool.map(calculate_final_result, batch)
                            new_pair_infos = [p for p in new_pair_infos if p is not None]
                            batch = []
                            total_pair_infos.extend(new_pair_infos)
                
                            end_time = time.time()
                            print(f"also cut multi:  {end_time - start_time:.6f}")
                    if batch:
                        new_pair_infos = pool.map(calculate_final_result, batch)
                        new_pair_infos = [p for p in new_pair_infos if p is not None]
                        total_pair_infos.extend(new_pair_infos)
                
                        end_time = time.time()
                        print(f"also cut multi:  {end_time - start_time:.6f}")
                    new_pair_infos = np.array(total_pair_infos)
            
            total = []
            start_time = time.time()
            for i in range(len(comb0)):
                result = my_func(i)
                total.append(result)
            
            end_time = time.time()
            # print(f"for:  {end_time - start_time:.6f}")
            
            new_save_keys = {}
            import pdb;pdb.set_trace()
            for i in self.scene_info.keys():
                if i != 'pair_infos':
                    new_save_keys[i] = self.scene_info[i]
                else: 
                    new_save_keys[i] = new_pair_infos
            
            # import pdb; pdb.set_trace()
            self.scene_info['pair_infos'] = new_pair_infos
            
            np.savez(new_file, **self.scene_info)
            print("done: ", new_file)
            # del new_save_keys
            del self.scene_info['pair_infos']
        '''
        
        if len(self.pair_infos) == 0:
            import pdb; pdb.set_trace()

        # parameters for image resizing, padding and depthmap padding
        if mode == 'train':
            assert img_resize is not None and img_padding and depth_padding
        self.img_resize = img_resize
        self.df = df
        self.img_padding = img_padding
        self.depth_max_size = 2000 if depth_padding else None  # the upperbound of depthmaps size in megadepth.

        # for training LoFTR
        self.augment_fn = augment_fn if mode == 'train' else None
        self.coarse_scale = getattr(kwargs, 'coarse_scale', 0.125)
        
        self.fp16 = fp16
        
    def __len__(self):
        return len(self.pair_infos)

    def __getitem__(self, idx):
        if self.mode in ['train', 'val']:
            (idx0, idx1, idx2), overlap_score, central_matches = self.pair_infos[idx]
    
            # read grayscale image and mask. (1, h, w) and (h, w)
            img_name0 = osp.join(self.root_dir, self.scene_info['image_paths'][idx0])
            img_name1 = osp.join(self.root_dir, self.scene_info['image_paths'][idx1])
            img_name2 = osp.join(self.root_dir, self.scene_info['image_paths'][idx2])
        
            # TODO: Support augmentation & handle seeds for each worker correctly.
            image0, mask0, scale0 = read_megadepth_gray(
                img_name0, self.img_resize, self.df, self.img_padding, None)
                # np.random.choice([self.augment_fn, None], p=[0.5, 0.5]))
            image1, mask1, scale1 = read_megadepth_gray(
                img_name1, self.img_resize, self.df, self.img_padding, None)
                # np.random.choice([self.augment_fn, None], p=[0.5, 0.5]))
            image2, mask2, scale2 = read_megadepth_gray(
                img_name2, self.img_resize, self.df, self.img_padding, None)
    
            # read depth. shape: (h, w)
            if self.mode in ['train', 'val']:
                depth0 = read_megadepth_depth(
                    osp.join(self.root_dir, self.scene_info['depth_paths'][idx0]), pad_to=self.depth_max_size)
                depth1 = read_megadepth_depth(
                    osp.join(self.root_dir, self.scene_info['depth_paths'][idx1]), pad_to=self.depth_max_size)
                depth2 = read_megadepth_depth(
                    osp.join(self.root_dir, self.scene_info['depth_paths'][idx2]), pad_to=self.depth_max_size)
            else:
                depth0 = depth1 = depth2 = torch.tensor([])
    
            # read intrinsics of original size
            K_0 = torch.tensor(self.scene_info['intrinsics'][idx0].copy(), dtype=torch.float).reshape(3, 3)
            K_1 = torch.tensor(self.scene_info['intrinsics'][idx1].copy(), dtype=torch.float).reshape(3, 3)
            K_2 = torch.tensor(self.scene_info['intrinsics'][idx2].copy(), dtype=torch.float).reshape(3, 3)
    
            # read and compute relative poses
            T0 = self.scene_info['poses'][idx0]
            T1 = self.scene_info['poses'][idx1]
            T2 = self.scene_info['poses'][idx2]
            T_0to1 = torch.tensor(np.matmul(T1, np.linalg.inv(T0)), dtype=torch.float)[:4, :4]  # (4, 4)
            T_1to0 = T_0to1.inverse()
            T_1to2 = torch.tensor(np.matmul(T2, np.linalg.inv(T1)), dtype=torch.float)[:4, :4]  # (4, 4)
            T_2to1 = T_1to2.inverse()
            T_0to2 = torch.tensor(np.matmul(T2, np.linalg.inv(T0)), dtype=torch.float)[:4, :4]
            T_2to0 = T_0to2.inverse()
    
            if self.fp16:
                image0, image1, image2, depth0, depth1, depth2, scale0, scale1, scale2 = map(lambda x: x.half(),
                                                                     [image0, image1, image2, depth0, depth1, depth2, scale0, scale1, scale2])
            data = {
                'image0': image0,  # (1, h, w)
                'depth0': depth0,  # (h, w)
                'image1': image1,
                'depth1': depth1,
                'image2': image2,
                'depth2': depth2,
                'T_0to1': T_0to1,  # (4, 4)
                'T_1to0': T_1to0,
                'T_1to2': T_1to2,  # (4, 4)
                'T_2to1': T_2to1,
                'T_0to2': T_0to2,  # (4, 4)
                'T_2to0': T_2to0,            
                'K0': K_0,  # (3, 3)
                'K1': K_1,
                'K2': K_2,
                'scale0': scale0,  # [scale_w, scale_h]
                'scale1': scale1,
                'scale2': scale2,
                'dataset_name': 'MegaDepth',
                'scene_id': self.scene_id,
                'pair_id': idx,
                'pair_names': (self.scene_info['image_paths'][idx0], self.scene_info['image_paths'][idx1], self.scene_info['image_paths'][idx2]),
            }
            # for LoFTR training
            if mask0 is not None:  # img_padding is True
                if self.coarse_scale:
                    [ts_mask_0, ts_mask_1, ts_mask_2] = F.interpolate(torch.stack([mask0, mask1, mask2], dim=0)[None].float(),
                                                            scale_factor=self.coarse_scale,
                                                            mode='nearest',
                                                            recompute_scale_factor=False)[0].bool()
                data.update({'mask0': ts_mask_0, 'mask1': ts_mask_1, 'mask2': ts_mask_2})
    
            return data
        else: 
            (idx0, idx1), overlap_score, central_matches = self.pair_infos[idx]

            # read grayscale image and mask. (1, h, w) and (h, w)
            img_name0 = osp.join(self.root_dir, self.scene_info['image_paths'][idx0])
            img_name1 = osp.join(self.root_dir, self.scene_info['image_paths'][idx1])
        
            # TODO: Support augmentation & handle seeds for each worker correctly.
            image0, mask0, scale0 = read_megadepth_gray(
                img_name0, self.img_resize, self.df, self.img_padding, None)
                # np.random.choice([self.augment_fn, None], p=[0.5, 0.5]))
            image1, mask1, scale1 = read_megadepth_gray(
                img_name1, self.img_resize, self.df, self.img_padding, None)
                # np.random.choice([self.augment_fn, None], p=[0.5, 0.5]))
    
            # read depth. shape: (h, w)
            if self.mode in ['train', 'val']:
                depth0 = read_megadepth_depth(
                    osp.join(self.root_dir, self.scene_info['depth_paths'][idx0]), pad_to=self.depth_max_size)
                depth1 = read_megadepth_depth(
                    osp.join(self.root_dir, self.scene_info['depth_paths'][idx1]), pad_to=self.depth_max_size)
            else:
                depth0 = depth1 = torch.tensor([])
    
            # read intrinsics of original size
            K_0 = torch.tensor(self.scene_info['intrinsics'][idx0].copy(), dtype=torch.float).reshape(3, 3)
            K_1 = torch.tensor(self.scene_info['intrinsics'][idx1].copy(), dtype=torch.float).reshape(3, 3)
    
            # read and compute relative poses
            T0 = self.scene_info['poses'][idx0]
            T1 = self.scene_info['poses'][idx1]
            T_0to1 = torch.tensor(np.matmul(T1, np.linalg.inv(T0)), dtype=torch.float)[:4, :4]  # (4, 4)
            T_1to0 = T_0to1.inverse()
    
            if self.fp16:
                image0, image1, depth0, depth1, scale0, scale1 = map(lambda x: x.half(),
                                                                     [image0, image1, depth0, depth1, scale0, scale1])
            data = {
                'image0': image0,  # (1, h, w)
                'depth0': depth0,  # (h, w)
                'image1': image1,
                'depth1': depth1,
                'T_0to1': T_0to1,  # (4, 4)
                'T_1to0': T_1to0,
                'K0': K_0,  # (3, 3)
                'K1': K_1,
                'scale0': scale0,  # [scale_w, scale_h]
                'scale1': scale1,
                'dataset_name': 'MegaDepth',
                'scene_id': self.scene_id,
                'pair_id': idx,
                'pair_names': (self.scene_info['image_paths'][idx0], self.scene_info['image_paths'][idx1]),
            }
            # for LoFTR training
            if mask0 is not None:  # img_padding is True
                if self.coarse_scale:
                    [ts_mask_0, ts_mask_1] = F.interpolate(torch.stack([mask0, mask1], dim=0)[None].float(),
                                                            scale_factor=self.coarse_scale,
                                                            mode='nearest',
                                                            recompute_scale_factor=False)[0].bool()
                data.update({'mask0': ts_mask_0, 'mask1': ts_mask_1})
    
            return data

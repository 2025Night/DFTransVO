import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia.geometry.subpix import dsnt
from kornia.utils.grid import create_meshgrid

from loguru import logger

class FineMatching(nn.Module):
    """FineMatching with s2d paradigm"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.local_regress_temperature = config['match_fine']['local_regress_temperature']
        self.local_regress_slicedim = config['match_fine']['local_regress_slicedim']
        self.fp16 = config['half']
        self.validate = False

    def forward(self, feat_0, feat_1, data):
        """
        Args:
            feat0 (torch.Tensor): [M, WW, C]
            feat1 (torch.Tensor): [M, WW, C]
            data (dict)
        Update:
            data (dict):{
                'expec_f' (torch.Tensor): [M, 3],
                'mkpts0_f' (torch.Tensor): [M, 2],
                'mkpts1_f' (torch.Tensor): [M, 2]}
        """
        M, WW, C = feat_0.shape
        W = int(math.sqrt(WW))
        scale = data['hw0_i'][0] / data['hw0_f'][0]
        self.M, self.W, self.WW, self.C, self.scale = M, W, WW, C, scale

        # corner case: if no coarse matches found
        if M == 0:
            assert self.training == False, "M is always > 0 while training, see coarse_matching.py"
            data.update({
                'conf_matrix_f': torch.empty(0, WW, WW, device=feat_0.device),
                'mkpts0_f': data['mkpts0_c'],
                'mkpts1_f': data['mkpts1_c'],
            })
            return

        # compute pixel-level confidence matrix
        with torch.autocast(enabled=True if not (self.training or self.validate) else False, device_type='cuda'):
            feat_f0, feat_f1 = feat_0[...,:-self.local_regress_slicedim], feat_1[...,:-self.local_regress_slicedim]
            feat_ff0, feat_ff1 = feat_0[...,-self.local_regress_slicedim:], feat_1[...,-self.local_regress_slicedim:]
            feat_f0, feat_f1 = feat_f0 / C**.5, feat_f1 / C**.5
            conf_matrix_f = torch.einsum('mlc,mrc->mlr', feat_f0, feat_f1)
            conf_matrix_ff = torch.einsum('mlc,mrc->mlr', feat_ff0, feat_ff1 / (self.local_regress_slicedim)**.5)
            conf_matrix_ff_reverse = torch.einsum('mlc,mrc->mlr', feat_ff1, feat_ff0 / (self.local_regress_slicedim)**.5)

        softmax_matrix_f = F.softmax(conf_matrix_f, 1) * F.softmax(conf_matrix_f, 2)
        softmax_matrix_f = softmax_matrix_f.reshape(M, self.WW, self.W+2, self.W+2)
        softmax_matrix_f = softmax_matrix_f[...,1:-1,1:-1].reshape(M, self.WW, self.WW)

        # for fine-level supervision
        if self.training or self.validate:
            data.update({'sim_matrix_ff': conf_matrix_ff})
            data.update({'conf_matrix_f': softmax_matrix_f})

        # compute pixel-level absolute kpt coords
        self.get_fine_ds_match(softmax_matrix_f, data)
        
        '''
        data.update({
            'mkpts0_f': data['mkpts0_c'],
            'mkpts1_f': data['mkpts1_c'],
        })
        return
        '''

        # generate seconde-stage 3x3 grid
        idx_l, idx_r = data['idx_l'], data['idx_r']
        idx_exep_l, idx_exep_r = data['idx_exep_l'], data['idx_exep_r']
        del data['idx_l'], data['idx_r'], data['idx_exep_l'], data['idx_exep_r']
        m_ids = torch.arange(M, device=idx_l.device, dtype=torch.long).unsqueeze(-1)
        m_exep_ids = m_ids
        m_ids = m_ids[:len(data['mconf'])]
        idx_r_iids, idx_r_jids = idx_r // W, idx_r % W
        idx_exep_r_iids, idx_exep_r_jids = idx_exep_r // W, idx_exep_r % W
        idx_l_iids, idx_l_jids = idx_l // W, idx_l % W
        idx_exep_l_iids, idx_exep_l_jids = idx_exep_l // W, idx_exep_l % W

        m_ids, idx_l, idx_r_iids, idx_r_jids = m_ids.reshape(-1), idx_l.reshape(-1), idx_r_iids.reshape(-1), idx_r_jids.reshape(-1)
        idx_r, idx_l_iids, idx_l_jids = idx_r.reshape(-1), idx_l_iids.reshape(-1), idx_l_jids.reshape(-1)
        delta = create_meshgrid(3, 3, True, conf_matrix_ff.device).to(torch.long) # [1, 3, 3, 2]

        m_ids = m_ids[...,None,None].expand(-1, 3, 3)
        idx_l = idx_l[...,None,None].expand(-1, 3, 3) # [m, k, 3, 3]
        idx_r = idx_r[...,None,None].expand(-1, 3, 3) # [m, k, 3, 3]

        idx_r_iids = idx_r_iids[...,None,None].expand(-1, 3, 3) + delta[None, ..., 1]
        idx_r_jids = idx_r_jids[...,None,None].expand(-1, 3, 3) + delta[None, ..., 0]
        idx_l_iids = idx_l_iids[...,None,None].expand(-1, 3, 3) + delta[None, ..., 1]
        idx_l_jids = idx_l_jids[...,None,None].expand(-1, 3, 3) + delta[None, ..., 0]
        
        
        m_exep_ids, idx_exep_l, idx_exep_r_iids, idx_exep_r_jids = m_exep_ids.reshape(-1), idx_exep_l.reshape(-1), idx_exep_r_iids.reshape(-1), idx_exep_r_jids.reshape(-1)
        idx_exep_r, idx_exep_l_iids, idx_exep_l_jids = idx_exep_r.reshape(-1), idx_exep_l_iids.reshape(-1), idx_exep_l_jids.reshape(-1)

        m_exep_ids = m_exep_ids[...,None,None].expand(-1, 3, 3)
        idx_exep_l = idx_exep_l[...,None,None].expand(-1, 3, 3) # [m, k, 3, 3]
        idx_exep_r = idx_exep_r[...,None,None].expand(-1, 3, 3) # [m, k, 3, 3]

        idx_exep_r_iids = idx_exep_r_iids[...,None,None].expand(-1, 3, 3) + delta[None, ..., 1]
        idx_exep_r_jids = idx_exep_r_jids[...,None,None].expand(-1, 3, 3) + delta[None, ..., 0]
        idx_exep_l_iids = idx_exep_l_iids[...,None,None].expand(-1, 3, 3) + delta[None, ..., 1]
        idx_exep_l_jids = idx_exep_l_jids[...,None,None].expand(-1, 3, 3) + delta[None, ..., 0]

        if idx_l.numel() == 0:
            data.update({
                'mkpts0_f': data['mkpts0_c'],
                'mkpts1_f': data['mkpts1_c'],
            })
            return

        # compute second-stage heatmap
        conf_matrix_ff = conf_matrix_ff.reshape(M, self.WW, self.W+2, self.W+2)
        conf_matrix_exep_ff = conf_matrix_ff[m_exep_ids, idx_exep_l, idx_exep_r_iids, idx_exep_r_jids]
        conf_matrix_ff = conf_matrix_ff[m_ids, idx_l, idx_r_iids, idx_r_jids]
        conf_matrix_ff = conf_matrix_ff.reshape(-1, 9)
        conf_matrix_exep_ff = conf_matrix_exep_ff.reshape(-1, 9)
        conf_matrix_exep_ff = F.softmax(conf_matrix_exep_ff / self.local_regress_temperature, -1)
        conf_matrix_ff = F.softmax(conf_matrix_ff / self.local_regress_temperature, -1)
        heatmap = conf_matrix_ff.reshape(-1, 3, 3)
        heatmap_exep = conf_matrix_exep_ff.reshape(-1, 3, 3)
        
        # reverse_part
        conf_matrix_ff_reverse = conf_matrix_ff_reverse.reshape(M, self.WW, self.W+2, self.W+2)
        conf_matrix_exep_ff_reverse = conf_matrix_ff_reverse[m_exep_ids, idx_exep_r, idx_exep_l_iids, idx_exep_l_jids]
        conf_matrix_ff_reverse = conf_matrix_ff_reverse[m_ids, idx_r, idx_l_iids, idx_l_jids]
        conf_matrix_ff_reverse = conf_matrix_ff_reverse.reshape(-1, 9)
        conf_matrix_exep_ff_reverse = conf_matrix_exep_ff_reverse.reshape(-1, 9)
        conf_matrix_exep_ff_reverse = F.softmax(conf_matrix_exep_ff_reverse / self.local_regress_temperature, -1)
        conf_matrix_ff_reverse = F.softmax(conf_matrix_ff_reverse / self.local_regress_temperature, -1)
        heatmap_reverse = conf_matrix_ff_reverse.reshape(-1, 3, 3)
        heatmap_exep_reverse = conf_matrix_exep_ff_reverse.reshape(-1, 3, 3)

        # compute coordinates from heatmap
        coords_normalized = dsnt.spatial_expectation2d(heatmap[None], True)[0]
        coords_normalized_exep = dsnt.spatial_expectation2d(heatmap_exep[None], True)[0]
        
        # reverse part        
        coords_normalized_reverse = dsnt.spatial_expectation2d(heatmap_reverse[None], True)[0]
        coords_normalized_exep_reverse = dsnt.spatial_expectation2d(heatmap_exep_reverse[None], True)[0]

        if data['bs'] == 1:
            scale1 = scale * data['scale1'] if 'scale0' in data else scale
            scale1_exep = scale1
        else:
            scale1 = scale * data['scale1'][data['b_ids']][:len(data['mconf']), ...][:,None,:].expand(-1, -1, 2).reshape(-1, 2) if 'scale0' in data else scale
            scale1_exep = scale * data['scale1'][data['b_ids']][:,None,:].expand(-1, -1, 2).reshape(-1, 2) if 'scale0' in data else scale

        # compute subpixel-level absolute kpt coords
        self.get_fine_match_local(coords_normalized, coords_normalized_reverse, data, scale1)
        self.get_fine_match_exep(coords_normalized_exep, coords_normalized_exep_reverse, data, scale1_exep)

    def get_fine_match_local(self, coords_normed, coords_normed_reverse, data, scale1):
        # import pdb; pdb.set_trace()
        W, WW, C, scale = self.W, self.WW, self.C, self.scale

        mkpts0_c, mkpts1_c = data['mkpts0_c'], data['mkpts1_c']
        # expec0_f, expec1_f = data['expec0_f'], data['expec1_f']

        # mkpts0_f and mkpts1_f
        mkpts0_f = mkpts0_c + (coords_normed_reverse[:len(data['mconf'])] * (3 // 2) * scale1)
        mkpts1_f = mkpts1_c + (coords_normed[:len(data['mconf'])] * (3 // 2) * scale1)
        # expec1_f = expec1_f + (coords_normed * (3 // 2) * scale1)

        # import pdb; pdb.set_trace() 
        data.update({
            "mkpts0_f": mkpts0_f,
            "mkpts1_f": mkpts1_f,
            # "expec0_f": expec0_f,
            # "expec1_f": expec1_f
        })
        
    def get_fine_match_exep(self, coords_normed, coords_normed_reverse, data, scale1):
        # import pdb; pdb.set_trace()

        mkpts0_c, mkpts1_c = data['expec0_f'], data['expec1_f']
        # expec0_f, expec1_f = data['expec0_f'], data['expec1_f']

        # mkpts0_f and mkpts1_f
        mkpts0_f = mkpts0_c + (coords_normed_reverse[:self.M] * (3 // 2) * scale1)
        mkpts1_f = mkpts1_c + (coords_normed[:self.M] * (3 // 2) * scale1)
        # expec1_f = expec1_f + (coords_normed * (3 // 2) * scale1)

        # import pdb; pdb.set_trace() 
        data.update({
            "expec0_f": mkpts0_f,
            "expec1_f": mkpts1_f,
            # "expec0_f": expec0_f,
            # "expec1_f": expec1_f
        })
        

    @torch.no_grad()
    def get_fine_ds_match(self, conf_matrix, data):
        # import pdb; pdb.set_trace()
        W, WW, C, scale = self.W, self.WW, self.C, self.scale
        m, _, _ = conf_matrix.shape

        conf_matrix = conf_matrix.reshape(m, -1)
        _, idx_exep = torch.max(conf_matrix, dim = -1)
        idx_exep = idx_exep[:,None]
        idx_exep_l, idx_exep_r = idx_exep // WW, idx_exep % WW
        
        conf_matrix = conf_matrix[:len(data['mconf']),...]
        val, idx = torch.max(conf_matrix, dim = -1)
        idx = idx[:,None]
        idx_l, idx_r = idx // WW, idx % WW

        data.update({'idx_l': idx_l, 'idx_r': idx_r})
        data.update({'idx_exep_l': idx_exep_l, 'idx_exep_r': idx_exep_r})

        if self.fp16:
            grid = create_meshgrid(W, W, False, conf_matrix.device, dtype=torch.float16) - W // 2 + 0.5 # kornia >= 0.5.1
        else:
            grid = create_meshgrid(W, W, False, conf_matrix.device) - W // 2 + 0.5
        grid = grid.reshape(1, -1, 2).expand(m, -1, -1)
        
        delta_l = torch.gather(grid, 1, idx_l.unsqueeze(-1).expand(-1, -1, 2))
        delta_r = torch.gather(grid, 1, idx_r.unsqueeze(-1).expand(-1, -1, 2))
        
        delta_exep_l = torch.gather(grid, 1, idx_exep_l.unsqueeze(-1).expand(-1, -1, 2))
        delta_exep_r = torch.gather(grid, 1, idx_exep_r.unsqueeze(-1).expand(-1, -1, 2))

        scale0 = scale * data['scale0'][data['b_ids']] if 'scale0' in data else scale
        scale1 = scale * data['scale1'][data['b_ids']] if 'scale0' in data else scale
        # scale0 = scale0 / 2
        # scale1 = scale1 / 2

        if torch.is_tensor(scale0) and scale0.numel() > 1: # scale0 is a tensor
            mkpts0_f = (data['mkpts0_c'][:,None,:] + (delta_l * scale0[:len(data['mconf']),...][:,None,:])).reshape(-1, 2)
            mkpts1_f = (data['mkpts1_c'][:,None,:] + (delta_r * scale1[:len(data['mconf']),...][:,None,:])).reshape(-1, 2)
            expec0_f = (data['expec0_f'][:,None,:] + (delta_exep_l * scale0[:,None,:])).reshape(-1, 2)
            expec1_f = (data['expec1_f'][:,None,:] + (delta_exep_r * scale1[:,None,:])).reshape(-1, 2)
        else: # scale0 is a float
            mkpts0_f = (data['mkpts0_c'][:,None,:] + (delta_l * scale0)).reshape(-1, 2)
            mkpts1_f = (data['mkpts1_c'][:,None,:] + (delta_r * scale1)).reshape(-1, 2)
            expec0_f = (data['expec0_f'][:,None,:] + (delta_exep_l * scale0)).reshape(-1, 2)
            expec1_f = (data['expec1_f'][:,None,:] + (delta_exep_r * scale1)).reshape(-1, 2)
         
        data.update({
            "mkpts0_c": mkpts0_f,
            "mkpts1_c": mkpts1_f,
            "expec0_f": expec0_f,
            "expec1_f": expec1_f
        })
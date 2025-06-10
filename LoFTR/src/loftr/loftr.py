import torch
import torch.nn as nn
from einops.einops import rearrange

from .backbone import build_backbone
from .loftr_module import LocalFeatureTransformer, FinePreprocess
from .utils.coarse_matching import CoarseMatching
from .utils.fine_matching import FineMatching
from ..utils.misc import detect_NaN

from loguru import logger

def reparameter(matcher):
    module = matcher.backbone.layer0
    if hasattr(module, 'switch_to_deploy'):
        module.switch_to_deploy()
    for modules in [matcher.backbone.layer1, matcher.backbone.layer2, matcher.backbone.layer3]:
        for module in modules:
            if hasattr(module, 'switch_to_deploy'):
                module.switch_to_deploy()
    for modules in [matcher.fine_preprocess.layer2_outconv2, matcher.fine_preprocess.layer1_outconv2]:
        for module in modules:
            if hasattr(module, 'switch_to_deploy'):
                module.switch_to_deploy()
    return matcher

class LoFTR(nn.Module):
    def __init__(self, config, profiler=None):
        super().__init__()
        # Misc
        self.config = config
        self.profiler = profiler

        # Modules
        self.backbone = build_backbone(config)            
        self.loftr_coarse = LocalFeatureTransformer(config)
        self.coarse_matching = CoarseMatching(config['match_coarse'])
        self.fine_preprocess = FinePreprocess(config)
        self.fine_matching = FineMatching(config)

    def forward(self, data):
        """ 
        Update:
            data (dict): {
                'image0': (torch.Tensor): (N, 1, H, W)
                'image1': (torch.Tensor): (N, 1, H, W)
                'mask0'(optional) : (torch.Tensor): (N, H, W) '0' indicates a padded position
                'mask1'(optional) : (torch.Tensor): (N, H, W)
            }
        """
        # 1. Local Feature CNN
        data.update({
            'bs': data['image0'].size(0),
            'hw0_i': data['image0'].shape[2:], 'hw1_i': data['image1'].shape[2:]
        })
        # import pdb; pdb.set_trace()
        if data['hw0_i'] == data['hw1_i']:  # faster & better BN convergence
            if self.training: 
                            
                ret_dict = self.backbone(torch.cat([data['image0'], data['image1'], data['image2']], dim=0))
                feats_c = ret_dict['feats_c']
                '''
                feats_x1_0, feats_x1_1, feats_x1_2 = torch.chunk(ret_dict['feats_x1'], 3, dim=0)
                feats_x2_0, feats_x2_1, feats_x2_2 = torch.chunk(ret_dict['feats_x2'], 3, dim=0)
                feats_x1_first = torch.cat([feats_x1_0, feats_x1_1], dim=0)
                feats_x1_second = torch.cat([feats_x1_1, feats_x1_2], dim=0)
                feats_x2_first = torch.cat([feats_x2_0, feats_x2_1], dim=0)
                feats_x2_second = torch.cat([feats_x2_1, feats_x2_2], dim=0)
                data.update({
                    'feats_x2': feats_x2_first,
                    'feats_x1': feats_x1_first,
                })
                '''
                data.update({
                    'feats_x2': ret_dict['feats_x2'],
                    'feats_x1': ret_dict['feats_x1'],
                })
                (feat_c0, feat_c1, feat_c2) = feats_c.split(data['bs'])
            else: 
                ret_dict = self.backbone(torch.cat([data['image0'], data['image1']], dim=0))
                feats_c = ret_dict['feats_c']
                data.update({
                    'feats_x2': ret_dict['feats_x2'],
                    'feats_x1': ret_dict['feats_x1'],
                })
                (feat_c0, feat_c1) = feats_c.split(data['bs'])
        else:  # handle different input shapes
            ret_dict0, ret_dict1 = self.backbone(data['image0']), self.backbone(data['image1'])
            feat_c0 = ret_dict0['feats_c']
            feat_c1 = ret_dict1['feats_c']
            data.update({
                'feats_x2_0': ret_dict0['feats_x2'],
                'feats_x1_0': ret_dict0['feats_x1'],
                'feats_x2_1': ret_dict1['feats_x2'],
                'feats_x1_1': ret_dict1['feats_x1'],
            })


        mul = self.config['resolution'][0] // self.config['resolution'][1]
        if self.training: 
                data.update({
            'hw0_c': feat_c0.shape[2:], 'hw1_c': feat_c1.shape[2:], 'hw2_c': feat_c2.shape[2:],
            'hw0_f': [feat_c0.shape[2] * mul, feat_c0.shape[3] * mul],
            'hw1_f': [feat_c1.shape[2] * mul, feat_c1.shape[3] * mul],
            'hw2_f': [feat_c2.shape[2] * mul, feat_c2.shape[3] * mul]
        })
        else: 
            data.update({
                'hw0_c': feat_c0.shape[2:], 'hw1_c': feat_c1.shape[2:],
                'hw0_f': [feat_c0.shape[2] * mul, feat_c0.shape[3] * mul],
                'hw1_f': [feat_c1.shape[2] * mul, feat_c1.shape[3] * mul]
            })

        # 2. coarse-level loftr module
        mask_c0 = mask_c1 = None  # mask is useful in training
        if 'mask0' in data:
            mask_c0, mask_c1 = data['mask0'], data['mask1']
        if self.training: 
            mask_c2 = data['mask2']
            feat_c0, feat_c01 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)
            feat_c11, feat_c2 = self.loftr_coarse(feat_c1, feat_c2, mask_c1, mask_c2)            
    
            feat_c0 = rearrange(feat_c0, 'n c h w -> n (h w) c')
            feat_c01 = rearrange(feat_c01, 'n c h w -> n (h w) c')
            
            feat_c11 = rearrange(feat_c11, 'n c h w -> n (h w) c')
            feat_c2 = rearrange(feat_c2, 'n c h w -> n (h w) c')                        
            
            # detect NaN during mixed precision training
            if self.config['replace_nan'] and (torch.any(torch.isnan(feat_c0)) or torch.any(torch.isnan(feat_c01)) or torch.any(torch.isnan(feat_c11))
                or torch.any(torch.isnan(feat_c2))):
                detect_NaN(feat_c0, feat_c01)
                detect_NaN(feat_c2, feat_c11)                
                
            # 3. match coarse-level
            data_new = data.copy()
            self.coarse_matching(feat_c0, feat_c01, data, 
                                    mask_c0=mask_c0.view(mask_c0.size(0), -1) if mask_c0 is not None else mask_c0, 
                                    mask_c1=mask_c1.view(mask_c1.size(0), -1) if mask_c1 is not None else mask_c1
                                    )
            self.coarse_matching(feat_c11, feat_c2, data_new, 
                                    mask_c0=mask_c1.view(mask_c1.size(0), -1) if mask_c1 is not None else mask_c1, 
                                    mask_c1=mask_c2.view(mask_c2.size(0), -1) if mask_c2 is not None else mask_c2
                                    )
    
            # prevent fp16 overflow during mixed precision training
            feat_c0, feat_c01, feat_c11, feat_c2  = map(lambda feat: feat / feat.shape[-1]**.5,
                            [feat_c0, feat_c01, feat_c11, feat_c2])
    
            # 4. fine-level refinement 
            # import pdb; pdb.set_trace()
            (feats_x10, feats_x11, feats_x12) = data['feats_x1'].split(data['bs'])
            (feats_x20, feats_x21, feats_x22) = data['feats_x2'].split(data['bs'])
            feats_x1 = torch.cat([feats_x10, feats_x11], dim=0)
            feats_x2 = torch.cat([feats_x20, feats_x21], dim=0)
            data.update({
                'feats_x2': feats_x2,
                'feats_x1': feats_x1,
            })
            feats_x1 = torch.cat([feats_x11, feats_x12], dim=0)
            feats_x2 = torch.cat([feats_x21, feats_x22], dim=0)    
            data_new.update({
                'feats_x2': feats_x2,
                'feats_x1': feats_x1,
            })
                             
            feat_f0_unfold, feat_f1_unfold = self.fine_preprocess(feat_c0, feat_c01, data)
            '''
            new_i = data['i_ids']
            new_j = data['j_ids']
            data.update({
                'feats_x2': feats_x2_second,
                'feats_x1': feats_x1_second,
                'i_ids': new_j,
                'j_ids': new_i
            })
            '''
            feat_f11_unfold, feat_f2_unfold = self.fine_preprocess(feat_c11, feat_c2, data_new)
            '''
            data.update({
                'i_ids': new_i,
                'j_ids': new_j
            })
            '''
            # detect NaN during mixed precision training
            if self.config['replace_nan'] and (torch.any(torch.isnan(feat_f0_unfold)) or torch.any(torch.isnan(feat_f1_unfold)) or torch.any(torch.isnan(feat_f2_unfold))):
                detect_NaN(feat_f0_unfold, feat_f1_unfold)
                detect_NaN(feat_f1_unfold, feat_f2_unfold)                
                                
            
            del feat_c0, feat_c1,feat_c01, feat_c11, feat_c2, mask_c0, mask_c1, mask_c2, feats_x10, feats_x11, feats_x12, feats_x20, feats_x21, feats_x22
    
            # 5. match fine-level            
            self.fine_matching(feat_f0_unfold, feat_f1_unfold, data)
            self.fine_matching(feat_f11_unfold, feat_f2_unfold, data_new)
            # data.update({'sim_matrix_ff_new': data_new['sim_matrix_ff']})
            
            # data.update({'conf_matrix_f_new': data_new['conf_matrix_f']})
            '''
            data.update({
            "mkpts0_f": data_new['mkpts1_f'],
            "expec0_f": data_new['expec1_f'],
            "mkpts0_f_new": data_new['mkpts0_f'],
            "mkpts1_f_new": data_new['mkpts1_f']})
            '''
            data.update({'expec1_f_new': data_new['expec0_f']})
            del data_new
                                                                        
                                
        else:                         
    
            feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)        
    
            feat_c0 = rearrange(feat_c0, 'n c h w -> n (h w) c')
            feat_c1 = rearrange(feat_c1, 'n c h w -> n (h w) c')
            
            # detect NaN during mixed precision training
            if self.config['replace_nan'] and (torch.any(torch.isnan(feat_c0)) or torch.any(torch.isnan(feat_c1))):
                detect_NaN(feat_c0, feat_c1)
                
            # 3. match coarse-level
            self.coarse_matching(feat_c0, feat_c1, data, 
                                    mask_c0=mask_c0.view(mask_c0.size(0), -1) if mask_c0 is not None else mask_c0, 
                                    mask_c1=mask_c1.view(mask_c1.size(0), -1) if mask_c1 is not None else mask_c1
                                    )
    
            # prevent fp16 overflow during mixed precision training
            feat_c0, feat_c1 = map(lambda feat: feat / feat.shape[-1]**.5,
                            [feat_c0, feat_c1])
    
            # 4. fine-level refinement
            feat_f0_unfold, feat_f1_unfold = self.fine_preprocess(feat_c0, feat_c1, data)
            
            # detect NaN during mixed precision training
            if self.config['replace_nan'] and (torch.any(torch.isnan(feat_f0_unfold)) or torch.any(torch.isnan(feat_f1_unfold))):
                detect_NaN(feat_f0_unfold, feat_f1_unfold)
            
            del feat_c0, feat_c1, mask_c0, mask_c1
    
            # 5. match fine-level            
            self.fine_matching(feat_f0_unfold, feat_f1_unfold, data)

    def load_state_dict(self, state_dict, *args, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith('matcher.'):
                state_dict[k.replace('matcher.', '', 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)
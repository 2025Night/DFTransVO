import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange, repeat

from loguru import logger
# from .elan_block import ELAB

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution without padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class FinePreprocess(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        block_dims = config['backbone']['block_dims']
        self.W = self.config['fine_window_size']
        self.fine_d_model = block_dims[0]

        self.layer3_outconv = conv1x1(block_dims[2], block_dims[2])
        self.layer2_outconv = conv1x1(block_dims[1], block_dims[2])
        self.layer2_outconv2 = nn.Sequential(
            conv3x3(block_dims[2], block_dims[2]),
            nn.BatchNorm2d(block_dims[2]),
            nn.LeakyReLU(),
            conv3x3(block_dims[2], block_dims[1]),
        )
        self.layer1_outconv = conv1x1(block_dims[0], block_dims[1])
        self.layer1_outconv2 = nn.Sequential(
            conv3x3(block_dims[1], block_dims[1]),
            nn.BatchNorm2d(block_dims[1]),
            nn.LeakyReLU(),
            conv3x3(block_dims[1], block_dims[0]),
        )
        '''        
        self.m_elan = 6
        self.n_share = 0
        self.c_elan = 60
        self.r_expand = 2
        self.scale = 2
        self.window_sizes = [2, 4, 8]
        m_head0 = [nn.Conv2d(64, self.c_elan, kernel_size=3, stride=1, padding=1)]
        m_head1 = [nn.Conv2d(100, self.c_elan, kernel_size=3, stride=1, padding=1)]
        m_body = []
        for i in range(self.m_elan // (1+self.n_share)):
            if (i+1) % 2 == 1: 
                m_body.append(
                    ELAB(
                        self.c_elan, self.c_elan, self.r_expand, 0, 
                        self.window_sizes, shared_depth=self.n_share
                    )
                )
            else:              
                m_body.append(
                    ELAB(
                        self.c_elan, self.c_elan, self.r_expand, 1, 
                        self.window_sizes, shared_depth=self.n_share
                    )
                )
        # define tail module
        m_tail0 = [
            nn.Conv2d(self.c_elan, 64*self.scale*self.scale, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(self.scale)
        ]
        m_tail1 = [
            nn.Conv2d(self.c_elan, 100*self.scale*self.scale, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(self.scale)
        ]

        self.head0 = nn.Sequential(*m_head0)
        self.head1 = nn.Sequential(*m_head1)
        self.body0 = nn.Sequential(*m_body)
        self.body1 = nn.Sequential(*m_body)
        self.tail0 = nn.Sequential(*m_tail0)
        self.tail1 = nn.Sequential(*m_tail1)
        '''

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p, mode="fan_out", nonlinearity="relu")

    def inter_fpn(self, feat_c, x2, x1, stride):
        feat_c = self.layer3_outconv(feat_c)
        feat_c = F.interpolate(feat_c, scale_factor=2., mode='bilinear', align_corners=False)

        x2 = self.layer2_outconv(x2)
        x2 = self.layer2_outconv2(x2+feat_c)
        x2 = F.interpolate(x2, scale_factor=2., mode='bilinear', align_corners=False)

        x1 = self.layer1_outconv(x1)
        x1 = self.layer1_outconv2(x1+x2)
        x1 = F.interpolate(x1, scale_factor=2., mode='bilinear', align_corners=False)
        return x1
    '''
    def inter_shuffle(self, feat, label=0):
        # feat = feat.reshape(-1, self.W, self.W, 64)
        feat = rearrange(feat, 'n c (w h)  -> n c w h', w=self.W, h = self.W)
        if label == 0:
            x = self.head0(feat)
            res = self.body0(x)
            res = res + x
            x = self.tail0(res)
            feat = rearrange(x, 'n c w h -> n c (w h) ', w=self.W * self.scale, h = self.W * self.scale)
        else: 
            x = self.head1(feat)
            res = self.body1(x)
            res = res + x
            x = self.tail1(res)
            feat = rearrange(x, 'n c w h -> n c (w h) ', w=self.W * self.scale, h = self.W * self.scale)
        return feat
    '''
    
    def forward(self, feat_c0, feat_c1, data):
        W = self.W
        stride = data['hw0_f'][0] // data['hw0_c'][0]

        data.update({'W': W})
        if data['b_ids'].shape[0] == 0:
            feat0 = torch.empty(0, self.W**2, self.fine_d_model, device=feat_c0.device)
            feat1 = torch.empty(0, self.W**2, self.fine_d_model, device=feat_c0.device)
            return feat0, feat1

        if data['hw0_i'] == data['hw1_i']:
            feat_c = rearrange(torch.cat([feat_c0, feat_c1], 0), 'b (h w) c -> b c h w', h=data['hw0_c'][0]) # 1/8 feat
            x2 = data['feats_x2'] # 1/4 feat
            x1 = data['feats_x1'] # 1/2 feat
            del data['feats_x2'], data['feats_x1']

            # 1. fine feature extraction
            x1 = self.inter_fpn(feat_c, x2, x1, stride)                    
            feat_f0, feat_f1 = torch.chunk(x1, 2, dim=0)

            # 2. unfold(crop) all local windows
            feat_f0 = F.unfold(feat_f0, kernel_size=(W, W), stride=stride, padding=0)
            feat_f0 = rearrange(feat_f0, 'n (c ww) l -> n l ww c', ww=W**2)
            feat_f1 = F.unfold(feat_f1, kernel_size=(W+2, W+2), stride=stride, padding=1)
            feat_f1 = rearrange(feat_f1, 'n (c ww) l -> n l ww c', ww=(W+2)**2)

            # 3. select only the predicted matches
            feat_f0 = feat_f0[data['b_ids'], data['i_ids']]  # [n, ww, cf]
            feat_f1 = feat_f1[data['b_ids'], data['j_ids']]
            
            # feat_f0 = self.inter_shuffle(feat_f0, 0)
            # feat_f1 = self.inter_shuffle(feat_f1, 1)

            return feat_f0, feat_f1
        else:  # handle different input shapes
            feat_c0, feat_c1 = rearrange(feat_c0, 'b (h w) c -> b c h w', h=data['hw0_c'][0]), rearrange(feat_c1, 'b (h w) c -> b c h w', h=data['hw1_c'][0]) # 1/8 feat
            x2_0, x2_1 = data['feats_x2_0'], data['feats_x2_1'] # 1/4 feat
            x1_0, x1_1 = data['feats_x1_0'], data['feats_x1_1'] # 1/2 feat
            del data['feats_x2_0'], data['feats_x1_0'], data['feats_x2_1'], data['feats_x1_1']

            # 1. fine feature extraction
            feat_f0, feat_f1 = self.inter_fpn(feat_c0, x2_0, x1_0, stride), self.inter_fpn(feat_c1, x2_1, x1_1, stride)

            # 2. unfold(crop) all local windows
            feat_f0 = F.unfold(feat_f0, kernel_size=(W, W), stride=stride, padding=0)
            feat_f0 = rearrange(feat_f0, 'n (c ww) l -> n l ww c', ww=W**2)
            feat_f1 = F.unfold(feat_f1, kernel_size=(W+2, W+2), stride=stride, padding=1)
            feat_f1 = rearrange(feat_f1, 'n (c ww) l -> n l ww c', ww=(W+2)**2)

            # 3. select only the predicted matches
            feat_f0 = feat_f0[data['b_ids'], data['i_ids']]  # [n, ww, cf]
            feat_f1 = feat_f1[data['b_ids'], data['j_ids']]

            return feat_f0, feat_f1
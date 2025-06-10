import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import rot_to_quat, quat_to_rot, diffuse_features
import time



class ViTEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads):
        super(ViTEncoder, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, num_heads, batch_first=True), num_layers=4)

    def forward(self, x):
        x = self.fc(x)  # Shape: [B, n, hidden_dim]
        x = self.transformer(x)  # Shape: [B, n, hidden_dim]
        return x  # Shape: [B, n, hidden_dim] 


class AttentionDecoder(nn.Module):
    def __init__(self, hidden_dim, num_layers = 9):
        super(AttentionDecoder, self).__init__()
        self.num_layers = num_layers
        self.self_attention = nn.ModuleList([nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True) for _ in range(self.num_layers)])
        self.cross_attention =nn.ModuleList([nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True) for _ in range(self.num_layers)])
        

    def forward(self, enc0, enc1):
        # Self-attention on the first encoding
        attn_out0, _ = self.self_attention[0](enc0, enc0, enc0)  # Shape: [B, n, hidden_dim]
        attn_out1, _ = self.self_attention[0](enc1, enc1, enc1)  # Shape: [B, n, hidden_dim]
        # Cross-attention between the two encodings
        attn_out2, _ = self.cross_attention[0](attn_out0, attn_out1, attn_out1)  # Shape: [B, n, hidden_dim]
        attn_out3, _ = self.cross_attention[0](attn_out1, attn_out0, attn_out0)  # Shape: [B, n, hidden_dim]
        for i in range(1, self.num_layers):
            attn_out0, _ = self.self_attention[i](attn_out2, attn_out2, attn_out2)  # Shape: [B, n, hidden_dim]
            attn_out1, _ = self.self_attention[i](attn_out3, attn_out3, attn_out3)  # Shape: [B, n, hidden_dim]
            # Cross-attention between the two encodings
            attn_out2, _ = self.cross_attention[i](attn_out0, attn_out1, attn_out1)  # Shape: [B, n, hidden_dim]
            attn_out3, _ = self.cross_attention[i](attn_out1, attn_out0, attn_out0)  # Shape: [B, n, hidden_dim]
        return attn_out2, attn_out3


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x):
        # Self-Attention
        attn_output, _ = self.attention(x, x, x)
        return attn_output


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, query, key_value):
        # Cross-Attention:
        attn_output, _ = self.attention(query, key_value, key_value)
        return attn_output


class PoseTransformer(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4, ff_dim=32, num_layers=9):
        super(PoseTransformer, self).__init__()
        # self.embed_dim = embed_dim
        
        
        # self.feature_proj = nn.Linear(input_dim, embed_dim)
        
        # Self-Attention and Cross-Attention 
        self.rotation_self_attention1 = nn.ModuleList([SelfAttention(embed_dim, num_heads) for _ in range(num_layers)])
        self.rotation_self_attention2 = nn.ModuleList([SelfAttention(embed_dim, num_heads) for _ in range(num_layers)])
        self.rotation_cross_attention = nn.ModuleList([CrossAttention(embed_dim, num_heads) for _ in range(num_layers)])
        
        self.translation_self_attention1 = nn.ModuleList([SelfAttention(embed_dim, num_heads) for _ in range(num_layers)])
        self.translation_self_attention2 = nn.ModuleList([SelfAttention(embed_dim, num_heads) for _ in range(num_layers)])
        self.translation_cross_attention = nn.ModuleList([CrossAttention(embed_dim, num_heads) for _ in range(num_layers)])
        
        #
        self.rotation_feedforward = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, ff_dim),
                nn.ReLU(),
                nn.Linear(ff_dim, embed_dim)
            ) for _ in range(num_layers)
        ])
        
        self.translation_feedforward = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, ff_dim),
                nn.ReLU(),
                nn.Linear(ff_dim, embed_dim)
            ) for _ in range(num_layers)
        ])
        '''
        self.rotation_feedforward =nn.Sequential(
                nn.Linear(embed_dim, ff_dim),
                nn.ReLU(),
                nn.Linear(ff_dim, embed_dim))
        
        self.translation_feedforward = nn.Sequential(
                nn.Linear(embed_dim, ff_dim),
                nn.ReLU(),
                nn.Linear(ff_dim, embed_dim)
            )
        '''
        
        # Learnable Query Embedding
        self.rotation_query_embed = nn.Parameter(torch.randn(1, 7, embed_dim))  # [1, 7, embed_dim]
        self.translation_query_embed = nn.Parameter(torch.randn(1, 7, embed_dim))  # [1, 7, embed_dim]
        
        #
        self.fc_out_rotation = nn.Linear(embed_dim, 4)
        self.fc_out_translation = nn.Linear(embed_dim, 3)

    def forward(self, feature1, feature2):
        b, n, _ = feature1.size()
        
        
        # feature1 = self.feature_proj(feature1)  # [b, n, embed_dim]
        # feature2 = self.feature_proj(feature2)  # [b, n, embed_dim]
        
        # Query Embedding
        rotation_query = self.rotation_query_embed.expand(b, -1, -1)  # [b, 7, embed_dim]
        translation_query = self.translation_query_embed.expand(b, -1, -1)  # [b, 7, embed_dim]
        feature_rotation1 = feature1
        feature_rotation2 = feature2
        feature_translation1 = feature1
        feature_translation2 = feature2
        #  Self-Attention and Cross-Attention
        for i in range(len(self.rotation_self_attention1)):
            # Self-Attention on feature1
            feature_rotation1 = feature_rotation1 + self.rotation_self_attention1[i](feature_rotation1)  # Residual connection
            
            # Self-Attention on feature2
            feature_rotation2 = feature_rotation2 + self.rotation_self_attention2[i](feature_rotation2)  # Residual connection
            
            # Cross-Attention: query attends to feature1 and feature2 (fused information)
            rotation_query = rotation_query + self.rotation_cross_attention[i](rotation_query, torch.cat([feature_rotation1, feature_rotation2], dim=1))  # Residual connection
            
            # Feedforward Network
            rotation_query = rotation_query + self.rotation_feedforward[i](rotation_query)  # Residual connection
            
            # Self-Attention on feature1
            feature_translation1 = feature_translation1 + self.translation_self_attention1[i](feature_translation1)  # Residual connection
            
            # Self-Attention on feature2
            feature_translation2 = feature_translation2 + self.translation_self_attention2[i](feature_translation2)  # Residual connection
            
            # Cross-Attention: query attends to feature1 and feature2 (fused information)
            translation_query = translation_query + self.translation_cross_attention[i](translation_query, torch.cat([feature_translation1, feature_translation2], dim=1))  # Residual connection
            
            # Feedforward Network
            translation_query = translation_query + self.translation_feedforward[i](translation_query)  # Residual connection
        
        # final
        pose_rotation = self.fc_out_rotation(rotation_query)  # [b, 7, 7]
        pose_translation = self.fc_out_translation(translation_query)
        
        # [b, 7]
        pose_rotation = pose_rotation.mean(dim=1)  # [b, 7]
        pose_translation = pose_translation.mean(dim=1)  # [b, 7]
        pose = torch.cat([pose_rotation, pose_translation], dim=1)
        
        return pose


class MLPModel(nn.Module):
    def __init__(self, channel, h, w):
        super(MLPModel, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)  
        self.fc1 = nn.Linear(channel, 128)           
        self.fc2 = nn.Linear(128, 64)                
        self.fc3 = nn.Linear(64, 7)                  

    def forward(self, x):
        # x: [b, channel, h, w]
        # x = self.global_pool(x)                     # [b, channel, 1, 1]
        x = x.view(x.size(0), -1)                   # [b, channel]
        x = F.relu(self.fc1(x))                     
        x = F.relu(self.fc2(x))                     
        x = self.fc3(x)                             
        return x


class ViTModel(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, num_heads=4):
        super(ViTModel, self).__init__()
        self.encoder = ViTEncoder(input_dim, hidden_dim, num_heads)
        self.decoder = AttentionDecoder(hidden_dim)
        self.pose_decoder = PoseTransformer()
        # self.fc_out = MLPModel(hidden_dim * 2, 480, 640)  # 4 rotation + 3 translation
        
        # self.merging

    def forward(self, points0, points1, feats0, feats1):
        # start = time.time()

        enc0 = self.encoder(points0)  # Shape: [B, n, hidden_dim]
        enc1 = self.encoder(points1) 
        attn0, attn1 = self.decoder(enc0, enc1)  # Shape: [B, n, hidden_dim]
        # end = time.time()
        # print("vit time: ", end - start)
        # start = time.time()                
        attn0 = diffuse_features(points0, attn0, feats0)# Shape: [B, hidden_dim, h, w]
        attn1 = diffuse_features(points1, attn1, feats1)
        # end = time.time()
        # print("diffuse time: ", end - start)
        # start = time.time()
        if attn0.shape[1] < 512: 
            pad_length = 512 - attn0.shape[1]
            attn0 = F.pad(attn0, (0, 0, 0, pad_length), mode='constant', value=0)
            attn1 = F.pad(attn1, (0, 0, 0, pad_length), mode='constant', value=0)
        # end = time.time()
        # print("mlp time: ", end - start)
        pose = self.pose_decoder(attn0, attn1)                
        return pose
        
        
        
        
        
        
class AbsoluteError(nn.Module):
    def __init__(self):
        super(AbsoluteError, self).__init__()
        
        
    @torch.no_grad()
    def set_param(self, b, f, P2, P2_inv): 
        self.f = f
        self.b = b
        self.P2_inv = torch.from_numpy(P2_inv).float().cuda()
        self.P2 = torch.from_numpy(P2).float().cuda()

    def compute_absoluteloss(self, real_pose, calc_pose):
        # import pdb; pdb.set_trace()
        real_R = rot_to_quat(real_pose[:, :3, :3]).cuda()
        # print("real_R: ", real_R)
        real_t = real_pose[:, 3:, :].cuda()
        # calc_R = rot_to_quat(calc_pose[:, :3, :3])
        calc_R = calc_pose[:, :4]
        # print("calc_R: ", calc_R)
        calc_t = calc_pose[:, 4:]
        # import pdb; pdb.set_trace()
        # print("real R rotation: ", is_rotation_matrix(real_pose[0][:3, :3]))
        # print("calc R rotation: ", is_rotation_matrix(calc_pose[0][:3, :3]))
        error_R = torch.mean(torch.sqrt(torch.sum((calc_R - real_R)**2, dim=1)))
        error_t = torch.mean(torch.sqrt(torch.sum((calc_t - real_t)**2, dim=1)))
        # error = error_R + error_t
        return error_R, error_t
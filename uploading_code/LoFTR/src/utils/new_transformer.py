import torch
import torch.nn as nn
import torch.nn.functional as F
# from utils import rot_to_quat, quat_to_rot, diffuse_features
import time


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
    def __init__(self, hidden_dim, num_layers = 4):
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
    def __init__(self, embed_dim=64, num_heads=4, ff_dim=32, num_layers=4):
        super(PoseTransformer, self).__init__()
        # self.embed_dim = embed_dim
        
        
        # self.feature_proj = nn.Linear(input_dim, embed_dim)
        
        # Self-Attention and Cross-Attention 
        self.self_attention1 = nn.ModuleList([SelfAttention(embed_dim, num_heads) for _ in range(num_layers)])
        self.self_attention2 = nn.ModuleList([SelfAttention(embed_dim, num_heads) for _ in range(num_layers)])
        self.cross_attention = nn.ModuleList([CrossAttention(embed_dim, num_heads) for _ in range(num_layers)])
        
        #
        self.feedforward = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, ff_dim),
                nn.ReLU(),
                nn.Linear(ff_dim, embed_dim)
            ) for _ in range(num_layers)
        ])
        
        # Learnable Query Embedding
        self.query_embed = nn.Parameter(torch.randn(1, 7, embed_dim))  # [1, 7, embed_dim]
        
        #
        self.fc_out = nn.Linear(embed_dim, 7)

    def forward(self, feature1, feature2):
        b, n, _ = feature1.size()
        
        
        # feature1 = self.feature_proj(feature1)  # [b, n, embed_dim]
        # feature2 = self.feature_proj(feature2)  # [b, n, embed_dim]
        
        # Query Embedding
        query = self.query_embed.expand(b, -1, -1)  # [b, 7, embed_dim]
        
        #  Self-Attention and Cross-Attention
        for i in range(len(self.self_attention1)):
            # Self-Attention on feature1
            feature1 = feature1 + self.self_attention1[i](feature1)  # Residual connection
            
            # Self-Attention on feature2
            feature2 = feature2 + self.self_attention2[i](feature2)  # Residual connection
            
            # Cross-Attention: query attends to feature1 and feature2 (fused information)
            query = query + self.cross_attention[i](query, torch.cat([feature1, feature2], dim=1))  # Residual connection
            
            # Feedforward Network
            query = query + self.feedforward[i](query)  # Residual connection
        
        # final
        pose = self.fc_out(query)  # [b, 7, 7]
        
        # [b, 7]
        pose = pose.mean(dim=1)  # [b, 7]
        
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
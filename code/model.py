import torch
import torch.nn as nn
import torch.nn.functional as F
import open3d as o3d
import numpy as np

class EnhancedMultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1, use_gating=True):
        super(EnhancedMultiHeadSelfAttention, self).__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        assert self.head_dim * heads == dim,
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.use_gating = use_gating
        if use_gating:
            self.gating = nn.Linear(dim, heads)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        q = self.query(x).view(batch_size, seq_len, self.heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.key(x).view(batch_size, seq_len, self.heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.value(x).view(batch_size, seq_len, self.heads, self.head_dim).permute(0, 2, 1, 3)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        if self.use_gating:
            gate_values = torch.sigmoid(self.gating(x)).permute(0, 2, 1).unsqueeze(
                -1)
            attn = attn * gate_values
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.dim)
        out = self.out_proj(out)
        return out, attn

class EnhancedFeedForward(nn.Module):
    def __init__(self, dim, ffn_dim=2048, dropout=0.1, activation='gelu'):
        super(EnhancedFeedForward, self).__init__()
        self.linear1 = nn.Linear(dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, dim)
        self.dropout = nn.Dropout(dropout)
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'swish':
            self.activation = lambda x: x * torch.sigmoid(x)
        else:
            self.activation = nn.GELU()
        self.shortcut = nn.Conv1d(dim, dim, kernel_size=1)

    def forward(self, x):
        identity = x
        identity_conv = self.shortcut(identity.transpose(1, 2)).transpose(1, 2)
        out = self.linear2(self.dropout(self.activation(self.linear1(x))))
        out = out + identity_conv
        return out

class EnhancedLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, conditional=False, cond_dim=None):
        super(EnhancedLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps
        self.conditional = conditional
        if conditional:
            self.cond_proj = nn.Linear(cond_dim, dim * 2)

    def forward(self, x, cond=None):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        if self.conditional and cond is not None:
            cond_gamma, cond_beta = self.cond_proj(cond).chunk(2, dim=-1)
            gamma = self.gamma * (1 + cond_gamma)
            beta = self.beta + cond_beta
        else:
            gamma, beta = self.gamma, self.beta
        return gamma * (x - mean) / (std + self.eps) + beta

class DeepTransformerEncoderLayer(nn.Module):
    def __init__(self, dim, heads=8, ffn_dim=2048, dropout=0.1, activation='gelu', use_gating=True):
        super(DeepTransformerEncoderLayer, self).__init__()
        self.self_attn = EnhancedMultiHeadSelfAttention(dim, heads, dropout, use_gating)
        self.feed_forward = EnhancedFeedForward(dim, ffn_dim, dropout, activation)
        self.norm1 = EnhancedLayerNorm(dim)
        self.norm2 = EnhancedLayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.intermediate_proj = nn.Linear(dim, dim)
        self.intermediate_norm = EnhancedLayerNorm(dim)

    def forward(self, x, mask=None):
        attn_output, attn_weights = self.self_attn(x, mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        x_intermediate = self.intermediate_proj(x)
        x = x + self.dropout(x_intermediate)
        x = self.intermediate_norm(x)
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output) 
        x = self.norm2(x) 
        return x, attn_weights

class DeepTransformerEncoder(nn.Module):
    def __init__(self, dim, num_layers=8, heads=8, ffn_dim=2048, dropout=0.1, activation='gelu'):
        super(DeepTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            DeepTransformerEncoderLayer(
                dim,
                heads,
                ffn_dim,
                dropout,
                activation,
                use_gating=(i % 2 == 0)
            ) for i in range(num_layers)
        ])

    def forward(self, x, mask=None):
        attentions = []
        for layer in self.layers:
            x, attn = layer(x, mask)
            attentions.append(attn)
        return x, attentions

class MultiScaleCrossAttention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1, scales=8):
        super(MultiScaleCrossAttention, self).__init__()
        self.dim = dim
        self.heads = heads
        self.scales = scales
        self.cross_attns = nn.ModuleList([
            CrossAttentionModule(dim // scales, heads // scales, dropout)
            for _ in range(scales)
        ])
        self.in_projs = nn.ModuleList([
            nn.Linear(dim, dim // scales) for _ in range(scales)
        ])
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, q_input, kv_input):
        outputs = []
        attns = []
        for i in range(self.scales):
            q_proj = self.in_projs[i](q_input)
            kv_proj = self.in_projs[i](kv_input)

            out, attn = self.cross_attns[i](q_proj, kv_proj)
            outputs.append(out)
            attns.append(attn)
        combined = torch.cat(outputs, dim=-1)
        out = self.out_proj(combined)
        avg_attn = torch.stack(attns, dim=0).mean(dim=0)
        return out, avg_attn

class CrossAttentionModule(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super(CrossAttentionModule, self).__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.rel_pos_encoding = nn.Parameter(torch.randn(1, heads, 1, 1))
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, q_input, kv_input):
        batch_size, q_len, _ = q_input.shape
        _, kv_len, _ = kv_input.shape
        q = self.query(q_input).view(batch_size, q_len, self.heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.key(kv_input).view(batch_size, kv_len, self.heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.value(kv_input).view(batch_size, kv_len, self.heads, self.head_dim).permute(0, 2, 1, 3)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn + self.rel_pos_encoding
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, q_len, self.dim)
        out = self.out_proj(out)
        return out, attn

class DeepCrossTransformerEncoderLayer(nn.Module):
    def __init__(self, dim, heads=8, ffn_dim=2048, dropout=0.1, activation='gelu'):
        super(DeepCrossTransformerEncoderLayer, self).__init__()
        self.cross_attn = CrossAttentionModule(dim, heads, dropout)
        self.feed_forward = EnhancedFeedForward(dim, ffn_dim, dropout, activation)
        self.norm1 = EnhancedLayerNorm(dim)
        self.norm2 = EnhancedLayerNorm(dim)
        self.norm_kv = EnhancedLayerNorm(dim)

    def forward(self, q_input, kv_input):
        q_norm = self.norm1(q_input)
        kv_norm = self.norm_kv(kv_input)
        attn_output, attn_weights = self.cross_attn(q_norm, kv_norm)
        x = q_input + self.dropout(attn_output)
        x_norm = self.norm2(x)
        ff_output = self.feed_forward(x_norm)
        x = x + self.dropout(ff_output)
        return x, attn_weights

class DeepCrossTransformerEncoder(nn.Module):
    def __init__(self, dim, num_layers=8, heads=8, ffn_dim=2048, dropout=0.1, activation='gelu'):
        super(DeepCrossTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            DeepCrossTransformerEncoderLayer(dim, heads, ffn_dim, dropout, activation)
            for _ in range(num_layers)
        ])
        self.multi_scale_layer = MultiScaleCrossAttention(dim, heads, dropout, scales=8)

    def forward(self, q_input, kv_input):
        attentions = []
        x = q_input
        ms_out, ms_attn = self.multi_scale_layer(q_input, kv_input)
        x = x + ms_out
        attentions.append(ms_attn)
        for layer in self.layers:
            x, attn = layer(x, kv_input)
            attentions.append(attn)
        return x, attentions

class MultiScalePointCloudFeatureExtractor(nn.Module):
    def __init__(self, feature_dim=256):
        super(MultiScalePointCloudFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, feature_dim, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(feature_dim)
        self.scale_convs = nn.ModuleList([
            nn.Conv1d(3, 64, kernel_size=1, dilation=d) for d in [1, 2, 4]
        ])
        self.scale_bns = nn.ModuleList([
            nn.BatchNorm1d(64) for _ in range(3)
        ])
        self.scale_fusion = nn.Conv1d(64 * 3, 128, kernel_size=1)
        self.scale_fusion_bn = nn.BatchNorm1d(128)
        self.linear = nn.Linear(feature_dim, feature_dim)
        self.transformer = DeepTransformerEncoder(
            dim=feature_dim,
            num_layers=8,
            heads=8,
            ffn_dim=feature_dim * 4,
            dropout=0.1,
            activation='gelu'
        )
        self.attention_pooling = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x, return_all_features=False):
        batch_size, num_points, _ = x.shape
        scale_features = []
        for i, (conv, bn) in enumerate(zip(self.scale_convs, self.scale_bns)):
            sf = F.relu(bn(conv(x_trans)))
            scale_features.append(sf)
        multi_scale = torch.cat(scale_features, dim=1)
        fused_scales = F.relu(self.scale_fusion_bn(self.scale_fusion(multi_scale)))
        x1 = F.relu(self.bn1(self.conv1(x_trans)))
        x2 = F.relu(self.bn2(self.conv2(x1)))
        x2 = x2 + fused_scales
        x3 = F.relu(self.bn3(self.conv3(x2)))
        x4 = F.relu(self.bn4(self.conv4(x3)))
        x = x4.transpose(2, 1)
        x, self_attentions = self.transformer(x)
        point_features = x
        attn_weights = F.softmax(self.attention_pooling(x).squeeze(-1), dim=1)
        global_feature = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)
        if return_all_features:
            return global_feature, point_features, self_attentions[-1], {
                'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4
            }
        else:
            return global_feature, point_features, self_attentions[-1]
class AdaptiveAttentionPointSelector(nn.Module):
    def __init__(self, feature_dim=256, k=256):
        super(AdaptiveAttentionPointSelector, self).__init__()
        self.k = k
        self.attention_adjustment = nn.Sequential(
            nn.Linear(feature_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, source_points, target_points, source_features, target_features, attention_weights):
        batch_size = source_points.shape[0]
        avg_attention = attention_weights.mean(dim=1)
        selected_weights = []
        for b in range(batch_size):
            s_feats = source_features[b]
            t_feats = target_features[b]
            s_expanded = s_feats.unsqueeze(1).expand(-1, t_feats.size(0), -1)
            t_expanded = t_feats.unsqueeze(0).expand(s_feats.size(0), -1, -1)
            combined_feats = torch.cat([s_expanded, t_expanded], dim=-1)
            adjustment = self.attention_adjustment(combined_feats).squeeze(-1)
            adjusted_attn = avg_attention[b] * adjustment
            selected_weights.append(adjusted_attn)
        adjusted_attention = torch.stack(selected_weights)
        _, indices = torch.topk(adjusted_attention, k=1, dim=2)
        indices = indices.squeeze(-1)
        if self.k < source_points.shape[1]:
            max_attn_per_source = torch.max(adjusted_attention, dim=2)[0]
            _, top_source_indices = torch.topk(max_attn_per_source, k=self.k, dim=1)
            batch_indices = torch.arange(batch_size, device=source_points.device).view(-1, 1).repeat(1, self.k)
            selected_source_points = source_points[batch_indices.flatten(), top_source_indices.flatten()].view(
                batch_size, self.k, 3)
            selected_source_features = source_features[batch_indices.flatten(), top_source_indices.flatten()].view(
                batch_size, self.k, -1)
            selected_target_indices = indices[batch_indices.flatten(), top_source_indices.flatten()].view(batch_size,
                                                                                                          self.k)
            batch_indices = torch.arange(batch_size, device=source_points.device).view(-1, 1).repeat(1, self.k)
            selected_target_points = target_points[batch_indices.flatten(), selected_target_indices.flatten()].view(
                batch_size, self.k, 3)
            selected_target_features = target_features[batch_indices.flatten(), selected_target_indices.flatten()].view(
                batch_size, self.k, -1)
        else:
            selected_source_points = source_points
            selected_source_features = source_features
            batch_indices = torch.arange(batch_size, device=source_points.device).view(-1, 1).repeat(1,
                                                                                                     source_points.shape[
                                                                                                         1])
            selected_target_points = target_points[batch_indices.flatten(), indices.flatten()].view(
                batch_size, source_points.shape[1], 3)
            selected_target_features = target_features[batch_indices.flatten(), indices.flatten()].view(
                batch_size, source_points.shape[1], -1)

        return (selected_source_points, selected_target_points,
                selected_source_features, selected_target_features)

class EnhancedTransformEstimator(nn.Module):
    def __init__(self, feature_dim=256):
        super(EnhancedTransformEstimator, self).__init__()
        self.feature_fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.corr_fusion = nn.Sequential(
            nn.Linear(feature_dim * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.correspondence_processor = nn.Sequential(
            nn.Linear(feature_dim * 2, 256), 
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.global_correspondence_fusion = nn.Sequential(
            nn.Linear(512 + 256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 12)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.2)
        self.quat_regressor = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 4)
        )
        self.trans_regressor = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 3)
        )

    def forward(self, source_global, enhanced_source_global, correspondence_features=None,
                selected_source_points=None, selected_target_points=None,
                selected_source_features=None, selected_target_features=None):
        batch_size = source_global.size(0)
        if correspondence_features is not None:
            if len(correspondence_features.shape) > 2:
                correspondence_features = torch.mean(correspondence_features, dim=1)
            x = torch.cat([source_global, enhanced_source_global, correspondence_features], dim=1)
            x = self.corr_fusion(x)
        else:
            x = torch.cat([source_global, enhanced_source_global], dim=1)
            x = self.feature_fusion(x)
        if (selected_source_features is not None and
                selected_target_features is not None and
                selected_source_features.size(1) > 0):
            B, K, D = selected_source_features.shape
            flat_source_features = selected_source_features.reshape(B * K, D)
            flat_target_features = selected_target_features.reshape(B * K, D)
            flat_correspondence_features = torch.cat([flat_source_features, flat_target_features], dim=1)
            processed_correspondence = self.correspondence_processor(flat_correspondence_features)
            processed_correspondence = processed_correspondence.reshape(B, K, -1)
            correspondence_global = torch.mean(processed_correspondence, dim=1)
            x = torch.cat([x, correspondence_global], dim=1)
            x = self.global_correspondence_fusion(x)
        quat = self.quat_regressor(x)
        quat = F.normalize(quat, p=2, dim=1)
        translation = self.trans_regressor(x)
        rotation = self._quaternion_to_rotation_matrix(quat)
        transform = torch.eye(4, device=x.device).unsqueeze(0).repeat(batch_size, 1, 1)
        transform[:, :3, :3] = rotation
        transform[:, :3, 3] = translation
        return transform, quat, translation

    def _quaternion_to_rotation_matrix(self, quaternion):
        batch_size = quaternion.size(0)
        qr, qi, qj, qk = quaternion[:, 0], quaternion[:, 1], quaternion[:, 2], quaternion[:, 3]
        n = torch.sum(quaternion ** 2, dim=1)
        s = 2.0 / n 
        qs = qr * s
        qis = qi * s
        qjs = qj * s
        qks = qk * s
        qri = qr * qis
        qrj = qr * qjs
        qrk = qr * qks
        qij = qi * qjs
        qik = qi * qks
        qjk = qj * qks
        rotation = torch.zeros(batch_size, 3, 3, device=quaternion.device)
        rotation[:, 0, 0] = 1.0 - (qj * qjs + qk * qks)
        rotation[:, 0, 1] = qij - qrk
        rotation[:, 0, 2] = qik + qrj
        rotation[:, 1, 0] = qij + qrk
        rotation[:, 1, 1] = 1.0 - (qi * qis + qk * qks)
        rotation[:, 1, 2] = qjk - qri
        rotation[:, 2, 0] = qik - qrj
        rotation[:, 2, 1] = qjk + qri
        rotation[:, 2, 2] = 1.0 - (qi * qis + qj * qjs)
        return rotation

class EnhancedPointCloudRegistration(nn.Module):
    def __init__(self, feature_dim=256, transformer_layers=4, k_correspondences=512):
        super(EnhancedPointCloudRegistration, self).__init__()
        self.feature_extractor = MultiScalePointCloudFeatureExtractor(feature_dim=feature_dim)
        self.cross_transformer = DeepCrossTransformerEncoder(
            dim=feature_dim,
            num_layers=transformer_layers,
            heads=8,
            ffn_dim=feature_dim * 4,
            dropout=0.1
        )

        self.point_selector = AdaptiveAttentionPointSelector(feature_dim=feature_dim, k=k_correspondences)
        self.transform_estimator = EnhancedTransformEstimator(feature_dim)
        self.correspondence_feature_fusion = nn.Sequential(
            nn.Conv1d(feature_dim * 2, feature_dim, 1),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU()
        )

    def forward(self, source_points, target_points):
        source_global, source_local, _ = self.feature_extractor(source_points)
        target_global, target_local, _ = self.feature_extractor(target_points)
        enhanced_source, cross_attns = self.cross_transformer(source_local, target_local)
        selected_source, selected_target, selected_source_features, selected_target_features = self.point_selector(
            source_points, target_points, source_local, target_local, cross_attns[-1])
        enhanced_source_global = torch.mean(enhanced_source, dim=1)
        transform, quaternion, translation = self.transform_estimator(
            source_global,
            enhanced_source_global,
            correspondence_features=None,
            selected_source_points=selected_source,
            selected_target_points=selected_target,
            selected_source_features=selected_source_features,
            selected_target_features=selected_target_features
        )

        return transform, selected_source, selected_target, cross_attns[-1], quaternion, translation

    def apply_transform(self, points, transform):
        homogeneous = torch.ones(points.size(0), points.size(1), 1, device=points.device)
        points_homogeneous = torch.cat([points, homogeneous], dim=2)
        points_transformed = torch.bmm(points_homogeneous, transform.transpose(1, 2))
        return points_transformed[:, :, :3]
    def compute_correspondence_loss(self, selected_source, selected_target, transform):
        transformed_source = self.apply_transform(selected_source, transform)
        dist = torch.norm(transformed_source - selected_target, dim=2)
        return dist.mean()

    def visualize_attention(self, source_points, target_points, attention_weights, k=10):
        attn = attention_weights.mean(dim=1).cpu().numpy()[0]
        flat_indices = np.argsort(attn.flatten())[-k:]
        source_indices, target_indices = np.unravel_index(flat_indices, attn.shape)
        source_pcd = o3d.geometry.PointCloud()
        source_pcd.points = o3d.utility.Vector3dVector(source_points[0].cpu().numpy())
        source_pcd.paint_uniform_color([1, 0, 0])

        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(target_points[0].cpu().numpy())
        target_pcd.paint_uniform_color([0, 1, 0])
        s_points = np.asarray(source_pcd.points)
        s_colors = np.asarray(source_pcd.colors)
        s_colors[source_indices] = [1, 1, 0]
        t_points = np.asarray(target_pcd.points)
        t_colors = np.asarray(target_pcd.colors)
        t_colors[target_indices] = [1, 1, 0]
        lines = []
        line_colors = []
        for i in range(k):
            lines.append([source_indices[i], target_indices[i] + len(s_points)])
            score = attn[source_indices[i], target_indices[i]]
            line_colors.append([score, score, score])
        line_colors = np.array(line_colors)
        if line_colors.max() > 0:
            line_colors = line_colors / line_colors.max()
        points = np.vstack((s_points, t_points))
        colors = np.vstack((s_colors, t_colors))
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(line_colors)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([pcd, line_set])
        return source_indices, target_indices
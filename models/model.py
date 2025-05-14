import torch
from torch import nn
from .layer import Transformer, HhyperLearningEncoder, CrossTransformer
from .bert import BertTextEncoder
from .gate import CrossAttentionTA, GateController
from .vision import VisionFeatureAggregator, ResNetWithDropout
from einops import repeat



        
        
class EMRCALBlock(nn.Module):
    def __init__(self, dim=128, heads=8, mlp_dim=128, dropout=0.):
        super(EMRCALBlock, self).__init__()

        self.fusion = HhyperLearningEncoder(dim=dim, depth=1, heads=heads, dim_head=16, dropout=dropout)

        # E-step
        self.visual_predictor = Transformer(
            num_frames=16,
            save_hidden=False,
            token_len=None,  # 直接输出 token_len 长度的序列
            dim=dim,
            depth=1,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout=dropout
        )

        # M-step
        self.feedback_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        
        self.cross_ta = CrossAttentionTA(dim=dim, heads=heads, mlp_dim=mlp_dim)
        self.gate_controller = GateController(dim=2*dim)  # one per layer


    def forward(self, h_v_list, h_l, h_a, h_hyper_v, layer_idx):
        h_ta_full = self.cross_ta(h_l, h_a)  # [B, 9, D] with [CLS] + 8 tokens
        gate_input = torch.cat([h_hyper_v[:,0],h_ta_full[:, 0]] ,dim=-1) 
        h_ta = h_ta_full[:, 1:]  # [B, 8, D] - remaining tokens for gated fusion
        gate, log_prob, prob = self.gate_controller(gate_input)  # [B, 1]
        gate = gate.unsqueeze(2)  # [B, 1, 1]

        # Apply gate to h_ta
        h_ta_gated = gate * h_ta

        h_hyper_v = self.fusion([h_v_list[layer_idx]], h_l, h_a, h_hyper_v)
        
        h_hyper_v = h_hyper_v + h_ta_gated

        # === E步：基于融合结果预测视觉表示
        h_v_feature = self.visual_predictor(h_hyper_v)  # (B, token_len, dim)
        
        if layer_idx+1<len(h_v_list):
            # === M步：将 h_v_feature 映射并反馈更新 h_v_list[layer_idx]
            h_v_feedback = self.feedback_proj(h_v_feature)  # (B, token_len, dim)

            h_v_list[layer_idx+1] = h_v_list[layer_idx+1] + h_v_feedback
            
        return h_hyper_v, h_v_list, h_v_feature, log_prob, gate, prob




class Model(nn.Module):
    def __init__(self, AHL_depth=3,dataset='mosi'):
        super(Model, self).__init__()

        self.h_hyper = nn.Parameter(torch.ones(1, 8, 128))

        # 基础编码器
        if dataset=='chsims':
            self.bertmodel = BertTextEncoder(use_finetune=True, transformers='bert', pretrained='bert-base-chinese')
        else:
            self.bertmodel = BertTextEncoder(use_finetune=True, transformers='bert', pretrained='bert-base-uncased')
            
        self.img_extractor = ResNetWithDropout(dropout_rate=0.3)
        self.vf_aggregator = VisionFeatureAggregator(dim=512)

        # 投影层
        self.proj_l0 = nn.Linear(768, 128)
        if dataset=='mosi':
            self.proj_a0 = nn.Linear(5, 128)
            fusion_layer_depth=2
        elif dataset=='mosei':
            self.proj_a0 = nn.Linear(74,128)
            fusion_layer_depth=4
        elif dataset=='chsims':
            self.proj_a0 = nn.Linear(33,128)
            fusion_layer_depth=4
            
        self.proj_v0 = nn.Sequential(nn.Linear(512, 128), nn.Dropout(0.3))

        # 序列处理
        self.proj_l = Transformer(num_frames=50, save_hidden=False, token_len=8, dim=128, depth=1, heads=8, mlp_dim=128)
        self.proj_a = Transformer(num_frames=50, save_hidden=False, token_len=8, dim=128, depth=1, heads=8, mlp_dim=128)
        self.proj_v = Transformer(num_frames=5, save_hidden=False, token_len=8, dim=128, depth=1, heads=8, mlp_dim=128)

        

        # 视觉主干编码器（输出多个中间层）
        self.vision_encoder = Transformer(num_frames=8, save_hidden=True, token_len=None, dim=128, depth=AHL_depth-1, heads=8, mlp_dim=128)

        # 新增：多层 EMRCALBlock 替代 h_hyper_layer_v
        self.em_rcal_blocks = nn.ModuleList([
            EMRCALBlock(dim=128, heads=8, mlp_dim=128, dropout=0.)
            for _ in range(AHL_depth)
        ])

        # 跨模态融合 & 情感预测
        self.fusion_layer = CrossTransformer(source_num_frames=8, tgt_num_frames=8, dim=128, depth=fusion_layer_depth, heads=8, mlp_dim=128)
        self.cls_head = nn.Linear(128, 1)
        
        self.vision_feature_extractor = nn.Sequential(
            Transformer(num_frames=8, save_hidden=False, token_len=1, dim=128, depth=1, heads=8, mlp_dim=64, dropout=0.),   
            nn.Dropout(0.3)
        )

        
    def forward(self, x_visual, x_audio, x_text):
        
        log_probs, gates, probs = [], [], []

        b = x_visual.size(0)
        h_hyper_v = repeat(self.h_hyper, '1 n d -> b n d', b=b)

        # 视觉特征提取
        x_visual = self.img_extractor(x_visual.view(-1, 3, 224, 224))
        x_visual = x_visual.view(b, 5, 512)
        x_visual = self.proj_v0(self.vf_aggregator(x_visual))



        x_audio = self.proj_a0(x_audio)
        x_text = self.bertmodel(x_text)
        x_text = self.proj_l0(x_text)

        h_v = self.proj_v(x_visual)[:, :8]
        h_a = self.proj_a(x_audio)[:, :8]
        h_l = self.proj_l(x_text)[:, :8]
        
        
        
        h_v_list = list(self.vision_encoder(h_v))  # 多层视觉表示
        

        for i, block in enumerate(self.em_rcal_blocks):
            h_hyper_v, h_v_list, h_v_feature, log_prob, gate, prob = block(h_v_list, h_l, h_a, h_hyper_v, layer_idx=i)
            log_probs.append(log_prob)
            gates.append(gate)
            probs.append(prob)

        # 情感预测
        feat = self.fusion_layer(h_hyper_v, h_v_list[-1])[:, 0]
        output = self.cls_head(feat)

        return output, self.vision_feature_extractor(h_v_feature)[:,0], log_probs, gates, probs


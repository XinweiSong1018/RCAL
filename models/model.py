import torch
from torch import nn
from .layer import Transformer, HhyperLearningEncoder, CrossTransformer
from .bert import BertTextEncoder
from .gate import CrossAttentionTA, GateController
from .vision import VisionFeatureAggregator, ResNetWithDropout
from einops import repeat


class EMRCALBlock(nn.Module):
    """
    A single EMRCAL refinement block combining cross-modal attention, gate-based fusion,
    and EM-style iterative updating for visual features.

    E-step: Predict refined visual features from current hyper features.
    M-step: Use visual feedback to update latent visual states.

    Components:
        - CrossAttentionTA: Cross-attention from text to audio.
        - GateController: Gating mechanism to control influence of audio-text fusion.
        - HhyperLearningEncoder: Multi-modal fusion into visual hyper representation.
        - Transformer: Visual feature predictor (E-step).
        - Feedback projection: Applies M-step update to next visual layer.
    """
    def __init__(self, dim=128, heads=8, mlp_dim=128, dropout=0.):
        super(EMRCALBlock, self).__init__()

        self.fusion = HhyperLearningEncoder(dim=dim, depth=1, heads=heads, dim_head=16, dropout=dropout)

        # E-step: transformer on visual hyperfeatures
        self.visual_predictor = Transformer(
            num_frames=16,
            save_hidden=False,
            token_len=None,
            dim=dim,
            depth=1,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout=dropout
        )

        # M-step: feedback network for next visual update
        self.feedback_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        
        self.cross_ta = CrossAttentionTA(dim=dim, heads=heads, mlp_dim=mlp_dim)
        self.gate_controller = GateController(dim=2*dim)  # Uses [CLS]_ta + [CLS]_hyper


    def forward(self, h_v_list, h_l, h_a, h_hyper_v, layer_idx):
        """
        Args:
            h_v_list: List of visual token sequences from previous layers
            h_l: Text tokens
            h_a: Audio tokens
            h_hyper_v: Visual hyper token sequence
            layer_idx: Current EM block index

        Returns:
            Updated hyper visual features, updated visual states, visual prediction, and gate stats
        """
        h_ta_full = self.cross_ta(h_l, h_a)  # [B, 9, D]
        gate_input = torch.cat([h_hyper_v[:, 0], h_ta_full[:, 0]], dim=-1)
        h_ta = h_ta_full[:, 1:]  # Remove CLS token for fusion
        gate, log_prob, prob = self.gate_controller(gate_input)
        gate = gate.unsqueeze(2)  # [B, 1, 1]
        h_ta_gated = gate * h_ta

        # Multimodal fusion into hyper visual
        h_hyper_v = self.fusion([h_v_list[layer_idx]], h_l, h_a, h_hyper_v)
        h_hyper_v = h_hyper_v + h_ta_gated

        # E-step: predict visual feature
        h_v_feature = self.visual_predictor(h_hyper_v)

        # M-step: update next visual layer if not last
        if layer_idx + 1 < len(h_v_list):
            h_v_feedback = self.feedback_proj(h_v_feature)
            h_v_list[layer_idx + 1] = h_v_list[layer_idx + 1] + h_v_feedback

        return h_hyper_v, h_v_list, h_v_feature, log_prob, gate, prob



class Model(nn.Module):
    """
    The full EMRCAL model for multimodal affect recognition (visual, audio, text).
    - Visual: ResNet features aggregated with Transformer
    - Audio/Text: Projected and encoded with Transformers
    - Fusion: Iterative EM-style gated fusion with visual feedback
    - Output: Regressed emotion intensity via [CLS] token

    Args:
        args (Namespace): All model hyperparameters
    """
    def __init__(self, args):
        super(Model, self).__init__()
        args = args.model

        # Learnable shared visual hyper representation
        self.h_hyper = nn.Parameter(torch.ones(1, args.token_length, args.token_dim))

        # Text Encoder
        self.bertmodel = BertTextEncoder(use_finetune=True, transformers='bert', pretrained=args.bert_pretrained)

        # Visual Encoder
        self.img_extractor = ResNetWithDropout(dropout_rate=args.vf_drop)
        self.vf_aggregator = VisionFeatureAggregator(dim=args.v_input_dim)

        # Linear projections for T, A, V modalities
        self.proj_l0 = nn.Sequential(nn.Linear(args.l_input_dim, args.proj_dst_dim), nn.Dropout(args.l_proj_drop))
        self.proj_a0 = nn.Sequential(nn.Linear(args.a_input_dim, args.proj_dst_dim), nn.Dropout(args.a_proj_drop))
        self.proj_v0 = nn.Sequential(nn.Linear(args.v_input_dim, args.proj_dst_dim), nn.Dropout(args.v_proj_drop))

        # Sequential Transformers (align temporal structure of each modality)
        self.proj_l = Transformer(args.l_input_length, args.token_length, False, args.proj_input_dim, args.proj_depth, args.proj_heads, args.proj_mlp_dim)
        self.proj_a = Transformer(args.a_input_length, args.token_length, False, args.proj_input_dim, args.proj_depth, args.proj_heads, args.proj_mlp_dim)
        self.proj_v = Transformer(args.v_input_length, args.token_length, False, args.proj_input_dim, args.proj_depth, args.proj_heads, args.proj_mlp_dim)

        # Initial visual encoder layers
        self.vision_encoder = Transformer(num_frames=args.token_length, save_hidden=True, token_len=None, dim=args.proj_input_dim, depth=args.emrcal_depth-1, heads=args.v_enc_heads, mlp_dim=args.v_enc_mlp_dim)

        # EMRCAL refinement blocks
        self.em_rcal_blocks = nn.ModuleList([
            EMRCALBlock(dim=args.v_enc_mlp_dim, heads=args.v_enc_heads, mlp_dim=args.v_enc_mlp_dim, dropout=0.)
            for _ in range(args.emrcal_depth)
        ])

        # Final fusion and prediction
        self.fusion_layer = CrossTransformer(source_num_frames=args.token_length, tgt_num_frames=args.token_length, dim=args.proj_input_dim, depth=args.fusion_layer_depth, heads=args.fusion_heads, mlp_dim=args.fusion_mlp_dim)
        self.cls_head = nn.Linear(args.token_dim, 1)

        # Visual feature extractor for auxiliary loss
        self.vision_feature_extractor = nn.Sequential(
            Transformer(num_frames=args.token_length, save_hidden=False, token_len=1, dim=args.proj_input_dim, depth=1, heads=args.v_enc_heads, mlp_dim=64, dropout=0.),
            nn.Dropout(args.vfe_drop)
        )


    def forward(self, x_visual, x_audio, x_text):
        """
        Forward pass of the EMRCAL model.

        Args:
            x_visual (Tensor): Raw image frames [B, 8, 3, 224, 224]
            x_audio (Tensor): Audio features [B, T, D]
            x_text (Tensor): Text BERT input [B, 3, L]

        Returns:
            Tuple: final output, auxiliary visual embedding, gate log_probs, gates, probabilities
        """
        log_probs, gates, probs = [], [], []

        b = x_visual.size(0)
        h_hyper_v = repeat(self.h_hyper, '1 n d -> b n d', b=b)

        # --- Vision Pipeline ---
        x_visual = self.img_extractor(x_visual.view(-1, 3, 224, 224))  # CNN features
        x_visual = x_visual.view(b, 5, 512)
        x_visual = self.proj_v0(self.vf_aggregator(x_visual))

        # --- Audio & Text Pipeline ---
        x_audio = self.proj_a0(x_audio)
        x_text = self.bertmodel(x_text)
        x_text = self.proj_l0(x_text)

        # --- Token Sequence Preparation ---
        h_v = self.proj_v(x_visual)[:, :8]
        h_a = self.proj_a(x_audio)[:, :8]
        h_l = self.proj_l(x_text)[:, :8]

        # --- Initial Visual Representation ---
        h_v_list = list(self.vision_encoder(h_v))

        # --- EMRCAL Refinement ---
        for i, block in enumerate(self.em_rcal_blocks):
            h_hyper_v, h_v_list, h_v_feature, log_prob, gate, prob = block(
                h_v_list, h_l, h_a, h_hyper_v, layer_idx=i)
            log_probs.append(log_prob)
            gates.append(gate)
            probs.append(prob)

        # --- Final Fusion and Prediction ---
        feat = self.fusion_layer(h_hyper_v, h_v_list[-1])[:, 0]  # [CLS]
        output = self.cls_head(feat)

        return output, self.vision_feature_extractor(h_v_feature)[:, 0], log_probs, gates, probs
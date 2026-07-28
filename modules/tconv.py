import pdb
import copy
import torch
import collections
import torch.nn as nn
import torch.nn.functional as F

from modules.attention_pool import AttentionPool1d


class TemporalConv(nn.Module):
    def __init__(self, input_size, hidden_size, conv_type=2, use_bn=False, num_classes=-1,
                 use_attn_pool=False, attn_pool_window=8, attn_pool_dim=256,
                 attn_pool_heads=4, attn_pool_dropout=0.1):
        super(TemporalConv, self).__init__()
        self.use_bn = use_bn
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.conv_type = conv_type
        self.use_attn_pool = use_attn_pool

        if self.conv_type == 0:
            self.kernel_size = ['K3']
        elif self.conv_type == 1:
            self.kernel_size = ['K5', "P2"]
        elif self.conv_type == 2:
            self.kernel_size = ['K5', "P2", 'K5', "P2"]

        # `layer_spec` runs parallel to temporal_conv and records what each entry
        # does to the sequence length, so forward() can track the running length
        # and hand it to the pooling layers for masking.
        modules = []
        self.layer_spec = []
        for layer_idx, ks in enumerate(self.kernel_size):
            input_sz = self.input_size if layer_idx == 0 else self.hidden_size
            if ks[0] == 'P':
                if self.use_attn_pool:
                    modules.append(AttentionPool1d(
                        self.hidden_size, window=attn_pool_window,
                        stride=int(ks[1]), dim=attn_pool_dim,
                        num_heads=attn_pool_heads, dropout=attn_pool_dropout))
                    self.layer_spec.append(('attn_pool', int(ks[1])))
                else:
                    modules.append(nn.MaxPool1d(kernel_size=int(ks[1]), ceil_mode=False))
                    self.layer_spec.append(('pool', int(ks[1])))
            elif ks[0] == 'K':
                modules.append(
                    nn.Conv1d(input_sz, self.hidden_size, kernel_size=int(ks[1]), stride=1, padding=0)
                )
                self.layer_spec.append(('conv', int(ks[1])))
                modules.append(nn.BatchNorm1d(self.hidden_size))
                self.layer_spec.append(('keep', None))
                modules.append(nn.ReLU(inplace=True))
                self.layer_spec.append(('keep', None))
        # ModuleList, not Sequential, so pooling layers can be passed the running
        # length. Both index their children by position, so checkpoints trained
        # with the Sequential version still load.
        self.temporal_conv = nn.ModuleList(modules)

        if self.num_classes != -1:
            self.fc = nn.Linear(self.hidden_size, self.num_classes)

    def update_lgt(self, lgt):
        feat_len = copy.deepcopy(lgt)
        for ks in self.kernel_size:
            if ks[0] == 'P':
                feat_len = torch.div(feat_len, 2)
            else:
                feat_len -= int(ks[1]) - 1
        return feat_len

    def forward(self, frame_feat, lgt):
        visual_feat = frame_feat
        # Mirrors update_lgt() step by step. Kept in sync by construction: both
        # walk the same kernel_size list with the same per-layer arithmetic.
        cur_lgt = lgt
        for module, (kind, size) in zip(self.temporal_conv, self.layer_spec):
            if kind == 'attn_pool':
                visual_feat = module(visual_feat, cur_lgt)
                cur_lgt = torch.div(cur_lgt, size)
            elif kind == 'pool':
                visual_feat = module(visual_feat)
                cur_lgt = torch.div(cur_lgt, size)
            else:
                visual_feat = module(visual_feat)
                if kind == 'conv':
                    cur_lgt = cur_lgt - (size - 1)
        lgt = self.update_lgt(lgt)
        logits = None if self.num_classes == -1 \
            else self.fc(visual_feat.transpose(1, 2)).transpose(1, 2)
        return {
            "visual_feat": visual_feat.permute(2, 0, 1),
            "conv_logits": logits.permute(2, 0, 1),
            "feat_len": lgt.cpu(),
        }

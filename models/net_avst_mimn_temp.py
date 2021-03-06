# -*- coding: utf-8 -*-
# file: ram.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

from .layers.dynamic_rnn import DynamicLSTM
from .layers.attention import Attention
import torch
import torch.nn as nn
import torch.nn.functional as F

# import torchvision.models as Models
import numpy as np
from einops import rearrange, repeat
from .visual_net import resnet18
import argparse


class FeedForward(nn.Module):
    '''A two-feed-forward-layer module'''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super(FeedForward, self).__init__()
        self.downsample = nn.Linear(d_in * 2, d_in)
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.tanh = nn.Tanh()

    def forward(self, x1, x2=None):
        x = torch.cat((x1, x2), dim=-1)
        x = self.tanh(x)
        x = F.relu(self.downsample(x))
        residual = x
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x = x + residual
        x = self.layer_norm(x)
        return x


# Co-attention between audio and video, question is treated as query
class CoAttention(nn.Module):
    def __init__(self, hops, hidden_dim, dropout=0.1):
        super(CoAttention, self).__init__()
        self.hops = hops
        self.hidden_dim = hidden_dim
        self.embed_dim = 512
        self.attention_audio = Attention(self.hidden_dim * 2, score_function='mlp')
        self.attention_video = Attention(self.hidden_dim * 2, score_function='mlp')
        self.attention_audio2video = Attention(
            self.hidden_dim * 2, score_function='mlp'
        )
        self.attention_video2audio = Attention(
            self.hidden_dim * 2, score_function='mlp'
        )
        self.FFN_audio = FeedForward(self.hidden_dim * 2, self.hidden_dim * 4)
        self.FFN_video = FeedForward(self.hidden_dim * 2, self.hidden_dim * 4)

        self.gru_cell_video = nn.GRUCell(self.hidden_dim * 2, self.hidden_dim * 2)
        self.gru_cell_audio = nn.GRUCell(self.hidden_dim * 2, self.hidden_dim * 2)

    def forward(self, question_memory, audio_memory, video_memory):
        et_audio = question_memory
        et_video = question_memory
        for _ in range(self.hops):
            it_al_audio2audio = self.attention_audio(audio_memory, et_audio).squeeze(
                dim=1
            )
            # audio branch
            it_al_video2audio = self.attention_video2audio(
                audio_memory, et_video
            ).squeeze(dim=1)
            it_al_audio = (it_al_audio2audio + it_al_video2audio) / 2
            # video branch
            it_al_video2video = self.attention_video(video_memory, et_video).squeeze(
                dim=1
            )
            it_al_audio2video = self.attention_audio2video(
                video_memory, et_audio
            ).squeeze(dim=1)
            it_al_video = (it_al_video2video + it_al_audio2video) / 2
            # et_audio = self.gru_cell_audio(it_al_audio, et_audio)
            # et_video = self.gru_cell_video(it_al_video, et_video)

            # combined_feature = rearrange(combined_feature, 'b t c -> b (t c)')
            et_audio = self.FFN_audio(it_al_audio, et_audio)
            et_video = self.FFN_video(it_al_video, et_video)

        return et_audio, et_video


# Temporal attention, question is used as query, audio and video are k, v
class temp_attention(nn.Module):
    def __init__(self, hidden_dim):
        super(temp_attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attention = nn.MultiheadAttention(
            self.hidden_dim * 2, 4, dropout=0.1, batch_first=True
        )
        # time branch and co-attention branch of video
        self.linear1 = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2)
        self.dropout1 = nn.Dropout(0.1)
        self.linear2 = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2)
        self.dropout2 = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(self.hidden_dim * 2)

    def forward(self, question_memory, et_feature):
        # feat_be = rearrange(et_feature, 'b (t c) -> b t c', t=1)
        feat_be = et_feature
        feat_att = self.attention(
            question_memory,
            feat_be,
            feat_be,
            attn_mask=None,
            key_padding_mask=None,
        )[0].squeeze(0)
        src = self.linear2(self.dropout1(F.relu(self.linear1(feat_att))))
        feat_att = feat_att + self.dropout2(src)
        feat_att = self.norm(feat_att)
        return feat_att


class AVQA_Fusion_Net(nn.Module):
    def __init__(self, args):
        super(AVQA_Fusion_Net, self).__init__()
        self.device = args.device
        self.qst_vocab_size = 93
        self.word_embed_size = 512
        self.embed_dim_audio = 128
        self.embed_dim_video = 512  # or 2048
        self.hidden_dim = 256
        self.num_classes = 42  # size of answer vocab
        self.hops = 3
        self.que_max_len = 14
        # self.img_extractor = nn.Sequential(
        #     *list(Models.resnet18(pretrained=True).children())[:-1]
        # )
        # get the feature from [-2] layer of resnet18
        self.img_extractor = nn.Sequential(
            *list(resnet18(pretrained=True, modal="vision").children())[:-1]
        )
        self.word2vec = nn.Embedding(self.qst_vocab_size, self.word_embed_size)
        self.bi_lstm_question = DynamicLSTM(
            self.word_embed_size,
            self.hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.bi_lstm_audio = DynamicLSTM(
            self.embed_dim_audio,
            self.hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.bi_lstm_video = DynamicLSTM(
            self.embed_dim_video,
            self.hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.avg_pooling = nn.AdaptiveAvgPool2d((1, 512))
        self.co_attention = CoAttention(self.hops, self.hidden_dim)

        self.question_audio_att = temp_attention(self.hidden_dim)
        self.question_video_att = temp_attention(self.hidden_dim)

        self.tanh = nn.Tanh()
        self.fc_fusion = nn.Linear(self.hidden_dim * 4, self.hidden_dim * 2)
        self.fc_ans = nn.Linear(
            self.hidden_dim * 2 * self.que_max_len, self.num_classes
        )

    # (self, audio, visual_posi, visual_nega, question)
    def forward(self, audio_posi, video_posi, video_nega, question):
        '''
        question      [B, C]
        audio         [B, T, C]
        video_posi    [B, T, C, H, W]
        video_nega    [B, T, C, H, W]
        '''
        B, T, C, H, W = video_posi.size()
        # question_memory_len = torch.sum(question != 0, dim=-1).to(self.device)
        question_memory_len = torch.tensor([self.que_max_len for i in range(B)]).to(
            self.device
        )
        # print(question_memory_len)
        audio_memory_len = torch.tensor([T for i in range(B)]).to(self.device)
        video_memory_len = torch.tensor([T for i in range(B)]).to(self.device)
        # nonzeros_question = torch.tensor(question_memory_len).to(self.device)

        question = self.word2vec(question)  # [B, maxseqlen, C] [B, 14, 512]
        video_posi = rearrange(video_posi, 'b t c h w -> (b t) c h w')
        video_posi = self.img_extractor(video_posi)  # [B*T, C, h w] [B*T, 512, 1, 1]
        video_posi = rearrange(video_posi, '(b t) c h w -> b t (c h w)', t=T)

        video_nega = rearrange(video_nega, 'b t c h w -> (b t) c h w')
        video_nega = self.img_extractor(video_nega)  # [B*T, C, h w] [B*T, 512, 1, 1]
        video_nega = rearrange(video_nega, '(b t) c h w -> b t (c h w)', t=T)

        # question_memory [B, 14, 512], audio_memory [B, T, 512], video_*_memory [B, T, 512]
        question_memory, (_, _) = self.bi_lstm_question(question, question_memory_len)
        audio_posi_memory, (_, _) = self.bi_lstm_audio(audio_posi, audio_memory_len)
        # audio_nega_memory, (_, _) = self.bi_lstm_audio(audio_nega, audio_memory_len)
        video_posi_memory, (_, _) = self.bi_lstm_video(video_posi, video_memory_len)
        video_nega_memory, (_, _) = self.bi_lstm_video(video_nega, video_memory_len)
        # print('question_memory: ', question_memory.shape)

        # # AvgPool [B, 1, hidden_dim*2] [B, 1, 512]
        # question_memory_pool = self.avg_pooling(question_memory)
        # question_memory_pool = rearrange(question_memory_pool, 'b t c -> b (t c)')
        question_memory_pool = question_memory  # [B, C] [2, 512]

        # co-attention branch of positive audio and positive video
        et_posi_audio, et_posi_video = self.co_attention(
            question_memory_pool, audio_posi_memory, video_posi_memory
        )

        # co-attention branch of positive audio and negative video
        _, et_nega_video = self.co_attention(
            question_memory_pool, audio_posi_memory, video_nega_memory
        )

        # co-attention branch of nega audio and positive video
        # et_nega_audio, _ = self.co_attention(
        #     question_memory_pool, audio_nega_memory, video_posi_memory
        # )

        ### attention, question as query on et_posi_video
        visual_posi_feat_att = self.question_video_att(question_memory, et_posi_video)

        ### attention, question as query on et_posi_audio
        audio_feat_att = self.question_video_att(question_memory, et_posi_audio)

        # fusion between fused video and fused audio
        feat = torch.cat(
            (audio_feat_att, visual_posi_feat_att),
            dim=-1,
        )
        feat = self.tanh(feat)
        feat = self.fc_fusion(feat)

        ## fusion with question
        combined_feature = torch.mul(feat, question_memory)
        combined_feature = self.tanh(combined_feature)
        combined_feature = rearrange(combined_feature, 'b t c -> b (t c)')
        out = self.fc_ans(combined_feature)  # [batch_size, ans_vocab_size]

        return (
            out,
            rearrange(et_posi_audio, 'b t c -> b (t c)'),
            rearrange(et_posi_video, 'b t c -> b (t c)'),
            rearrange(et_nega_video, 'b t c -> b (t c)'),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.device = 'cpu'
    model = AVQA_Fusion_Net(args)
    model.eval()
    audio = torch.randn(2, 10, 128)
    video_posi = torch.randn(2, 10, 3, 224, 224)
    video_nega = torch.randn(2, 10, 3, 224, 224)
    question = np.array([np.random.randint(0, 93, 14), np.random.randint(0, 93, 14)])
    question = torch.from_numpy(question).long()
    mask = torch.ones(2, 20).long()
    B, T, C, H, W = video_posi.shape
    question_memory_len = torch.sum(question != 0, dim=-1)
    audio_memory_len = torch.from_numpy(np.array([T for i in range(B)]))
    video_memory_len = torch.from_numpy(np.array([T for i in range(B)]))
    nonzeros_question = torch.from_numpy(np.array(question_memory_len))
    out_qa, et_posi_audio, et_posi_video, et_nega_video = model(
        audio, video_posi, video_nega, question
    )
    print('\nout_qa feature dimension ----- ', out_qa.size())
    print('audio_posi_feat feature dimension ----- ', et_posi_audio.size())
    print('audio_nega_feat feature dimension ----- ', et_posi_audio.size())
    print('visual_posi_feat feature dimension ----- ', et_posi_video.size())
    print('visual_nega_feat feature dimension ----- ', et_nega_video.size())
    print('-- model constructing successfully --')

# -*- coding: utf-8 -*-
# file: ram.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.
import sys
sys.path.append('..')
from layers.dynamic_rnn import DynamicLSTM
from layers.attention import Attention
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models
import numpy as np
from einops import rearrange
from models.visual_net import resnet18
import argparse

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
        self.stage1_hops = 2
        self.stage2_hops = 2
        self.num_heads = 4
        self.lstm_num_layers = 1
        self.que_max_len = 10
        # get the feature from [-2] layer of resnet18
        # self.img_extractor = nn.Sequential(*list(resnet18(pretrained=True, modal="vision").children())[:-1])
        # img_extractor = models.video.r2plus1d_18(pretrained=True)
        # self.img_extractor = nn.Sequential(*list(img_extractor.children())[:-1])
        # for p in self.img_extractor.parameters():
        #     p.requires_grad = False
        self.word2vec = nn.Embedding(self.qst_vocab_size, self.word_embed_size)
        self.bi_lstm_question = DynamicLSTM(self.word_embed_size,self.hidden_dim,num_layers=self.lstm_num_layers,batch_first=True,bidirectional=True,)
        self.bi_lstm_audio = DynamicLSTM(self.embed_dim_audio,self.hidden_dim,num_layers=self.lstm_num_layers,batch_first=True,bidirectional=True,)
        self.bi_lstm_video = DynamicLSTM(self.embed_dim_video,self.hidden_dim,num_layers=self.lstm_num_layers,batch_first=True,bidirectional=True,)
        
        self.tanh = nn.Tanh()
        self.EAVF_fusion = nn.Linear(self.hidden_dim * 6, self.hidden_dim * 2)
        self.MF_fusion = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2)
        self.LAF_fusion = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2)
        self.fc_ans = nn.Linear(self.hidden_dim * 2 * self.que_max_len, self.num_classes)

    # (self, audio, visual_posi, visual_nega, question)
    def forward(self, audio_posi, video_posi, video_nega, question):
        '''
        question      [B, C]
        audio         [B, T, C]
        video_posi    [B, T, C, H, W]
        video_nega    [B, T, C, H, W]
        '''
        B, T, C = video_posi.size()
        # question_memory_len = torch.sum(question != 0, dim=-1).to(self.device)
        question_memory_len = torch.tensor([self.que_max_len for i in range(B)]).to(self.device)
        # print(question_memory_len)
        audio_memory_len = torch.tensor([T for i in range(B)]).to(self.device)
        video_memory_len = torch.tensor([T for i in range(B)]).to(self.device)
        # nonzeros_question = torch.tensor(question_memory_len).to(self.device)

        question = self.word2vec(question)  # [B, maxseqlen, C] [B, 14, 512]

        # question_memory [B, 14, 512], audio_memory [B, T, 512], video_*_memory [B, T, 512]
        question_memory, (_, _) = self.bi_lstm_question(question, question_memory_len)
        audio_memory, (_, _) = self.bi_lstm_audio(audio_posi, audio_memory_len)
        video_posi_memory, (_, _) = self.bi_lstm_video(video_posi, video_memory_len)
        video_nega_memory, (_, _) = self.bi_lstm_video(video_nega, video_memory_len)
        # print('question_memory: ', question_memory.shape)
        
        # EAVF
        av_feat = torch.cat((audio_memory, video_posi_memory),dim=-1,)
        qav_feat = torch.cat((question_memory, av_feat),dim=-1,)
        EAVF_feat = self.tanh(qav_feat)
        EAVF_feat = self.EAVF_fusion(EAVF_feat)
        EAVF_feat = self.tanh(EAVF_feat)
        
        # MF
        qav_feat = audio_memory * video_posi_memory * question_memory
        MF_feat = self.tanh(qav_feat)
        MF_feat = self.MF_fusion(MF_feat)
        MF_feat = self.tanh(MF_feat)
        
        # LAF
        qav_feat = audio_memory * video_posi_memory * question_memory
        LAF_feat = self.tanh(qav_feat)
        LAF_feat = self.LAF_fusion(LAF_feat)
        LAF_feat = self.tanh(LAF_feat)
        
        averaged_feature = (EAVF_feat + MF_feat + LAF_feat)/3
        
        combined_feature = rearrange(averaged_feature, 'b t c -> b (t c)')
        out = self.fc_ans(combined_feature)  # [batch_size, ans_vocab_size]

        return out


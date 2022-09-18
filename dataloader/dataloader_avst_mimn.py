import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd
import glob
import json
import ast
import cv2
from PIL import Image
from munch import munchify
import logging
# from pytorch_pretrained import BertModel, BertTokenizer
from transformers import BertModel, BertTokenizer
from torchvision import transforms, utils

PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号

def ids_to_multinomial(id, categories):
    """label encoding
    Returns:
      1d array, multimonial representation, e.g. [1,0,1,0,0,...]
    """
    id_to_idx = {id: index for index, id in enumerate(categories)}

    return id_to_idx[id]



class AVQA_dataset(Dataset):
    def __init__(
        self,
        label,
        audio_dir,
        posi_video_dir,
        nega_video_dir,
        transform=None,
        mode_flag='train',
        num_frames = 60
    ):

        # 优化一些在get_item()中的操作，例如ToTensor, normalization，将这些操作在对数据进行初始化时就完成，而不是在每次取batch时再做
        samples = json.load(
            open('/home/jovyan/Bert/data/json/avqa-train.json', 'r')
        )
        # nax =  nne
        ques_vocab = ['<pad>']
        ans_vocab = []
        i = 0
        for sample in samples:
            i += 1
            question = sample['question_content'].rstrip().split(' ')
            question[-1] = question[-1][:-1]

            p = 0
            for pos in range(len(question)):
                if '<' in question[pos]:
                    question[pos] = ast.literal_eval(sample['templ_values'])[p]
                    p += 1

            for wd in question:
                if wd not in ques_vocab:
                    ques_vocab.append(wd)
            if sample['anser'] not in ans_vocab:
                ans_vocab.append(sample['anser'])

        self.ques_vocab = ques_vocab
        self.ans_vocab = ans_vocab
        self.word_to_ix = {word: i for i, word in enumerate(self.ques_vocab)}

        self.audio_dir = audio_dir
        self.posi_video_dir = posi_video_dir
        self.nega_video_dir = nega_video_dir
        self.transform = transform
        self.num_frames = num_frames

        self.samples = json.load(open(label, 'r'))
        self.max_len = 14  # question length

        posi_video_list = []
        nega_video_list = []
        audio_list = []
        question_list = []
        answer_list = []
        for idx in range(len(self.samples)):
            sample = self.samples[idx]
            video_name = sample['video_id']
            # positive video name
            posi_video_list.append(video_name)
            # # negative video name
            while 1:
                neg_video_id = np.random.randint(0, len(self.samples))
                if neg_video_id != idx:
                    break
            nega_video_list.append(self.samples[neg_video_id]['video_id'])
            # audio
            audio = np.load(os.path.join(self.audio_dir, video_name + '.npy'))[::6, :]
            audio_list.append(audio)

            # question
            question = sample['question_content'].rstrip().split(' ')
            question[-1] = question[-1][:-1]
            p = 0
            for pos in range(len(question)):
                if '<' in question[pos]:
                    question[pos] = ast.literal_eval(sample['templ_values'])[p]
                    p += 1
            if len(question) < self.max_len:
                n = self.max_len - len(question)
                for i in range(n):
                    question.append('<pad>')
            idxs = [self.word_to_ix[w] for w in question]
            # ques = torch.tensor(idxs, dtype=torch.long)
            question_list.append(idxs)
            # answer
            answer = sample['anser']
            label = ids_to_multinomial(answer, self.ans_vocab)
            answer = np.array(label)
            answer_list.append(answer)

        self.posi_video_list = posi_video_list
        self.nega_video_list = nega_video_list
        self.audio_list = torch.from_numpy(np.array(audio_list).astype(np.float32))
        self.question_list = torch.from_numpy(np.array(question_list)).long()
        self.answer_list = torch.from_numpy(np.array(answer_list)).long()

    def __len__(self):
        return len(self.posi_video_list)

    def __getitem__(self, idx):
        # positive video
        posi_name = self.posi_video_list[idx]
        
        visual_posi = np.load(os.path.join(self.posi_video_dir, posi_name + '.npy'))
        visual_posi = torch.from_numpy(np.array(visual_posi))

        # # negative video
        nega_name = self.nega_video_list[idx]
        visual_nega = np.load(os.path.join(self.nega_video_dir, nega_name + '.npy'))
        visual_nega = torch.from_numpy(visual_nega)

        #  audio
        audio = self.audio_list[idx]

        # question
        ques = self.question_list[idx]

        # answer, the label is the position of the anser in the ans_vocab
        label = self.answer_list[idx]
        sample = {
            'audio_posi': audio,
            'visual_posi': visual_posi,
            'visual_nega': visual_nega,
            'question': ques,
            'label': label,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    def __call__(self, sample):

        audio = sample['audio']
        visual_posi = sample['visual_posi']
        visual_nega = sample['visual_nega']
        label = sample['label']

        return {
            'audio': torch.from_numpy(audio),
            'visual_posi': sample['visual_posi'],
            'visual_nega': sample['visual_nega'],
            'question': sample['question'],
            'label': label,
        }

def TransformImage(img):

    transform_list = []
    mean = [0.43216, 0.394666, 0.37645]
    std = [0.22803, 0.22145, 0.216989]
    
    transform_list.append(transforms.ToPILImage())
    transform_list.append(transforms.Resize([224,224]))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean, std))
    trans = transforms.Compose(transform_list)
    frame_tensor = trans(img)
    
    return frame_tensor
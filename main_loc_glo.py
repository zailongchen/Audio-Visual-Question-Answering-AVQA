from __future__ import print_function
import sys
sys.path.append("/home/jovyan/Bert")
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from dataloader.dataloader_loc_glo import *
from models.net_loc_glo import AVQA_Fusion_Net
import ast
import json
import numpy as np
import time
from tqdm import tqdm
from info_nce import InfoNCE
import config
from torch import distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
import logging

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def reduce_loss(tensor, rank, world_size):
    with torch.no_grad():
        dist.reduce(tensor, dst=0)
        if rank == 0:
            tensor /= world_size

def batch_organize(out_match_posi, out_match_nega):
    out_match = torch.zeros(out_match_posi.shape[0] * 2, out_match_posi.shape[1])
    batch_labels = torch.zeros(out_match_posi.shape[0] * 2)
    for i in range(out_match_posi.shape[0]):
        out_match[i * 2, :] = out_match_posi[i, :]
        out_match[i * 2 + 1, :] = out_match_nega[i, :]
        batch_labels[i * 2] = 1
        batch_labels[i * 2 + 1] = 0

    return out_match, batch_labels


def train(args, model, train_loader, optimizer, criterion_qa, criterion_contr, criterion_tri, epoch):
    start_time = time.time()
    model.train()

    # for batch_idx, sample in enumerate(tqdm(train_loader)):
    for batch_idx, sample in enumerate(train_loader):
        # logging.info("data loading time -------------------------- %f ",time.time() - start_time,)
        batch_time = time.time()
        audio_posi, visual_posi, visual_nega, target, question = (
            sample['audio_posi'].to(args.device),
            sample['visual_posi'].to(args.device),
            sample['visual_nega'].to(args.device),
            sample['label'].to(args.device),
            sample['question'].to(args.device),)

        optimizer.zero_grad()
        (out_qa, loc_audio_posi_feat, loc_visual_posi_feat, loc_visual_nega_feat,
         glo_audio_posi_feat, glo_visual_posi_feat, glo_visual_nega_feat,) = model(audio_posi, visual_posi,visual_nega,question)

        loss_loc = criterion_contr(loc_audio_posi_feat, loc_visual_posi_feat, loc_visual_nega_feat)
        loss_glo = criterion_contr(glo_audio_posi_feat, glo_visual_posi_feat, glo_visual_nega_feat)
        
        loss_qa = criterion_qa(out_qa, target)
        loss = loss_qa + 0.1 * loss_loc + 0.1 * loss_glo
        # loss = loss_qa

        loss.backward()
        optimizer.step()
        # logging.info("data processing time ---------------------------------- %f ",time.time() - batch_time,)
        reduce_loss(loss, args.local_rank, args.world_size)
        if batch_idx % args.log_interval == 0:
            logging.info('Train Epoch: {0} [{1}/{2} ]\tLoss: {3}'.format(epoch,batch_idx * len(audio_posi),len(train_loader.dataset),loss.item(),)) 

        start_time = time.time()

def eval(model, val_loader, epoch):
    model.eval()
    total_qa = 0
    total_match = 0
    correct_qa = 0
    correct_match = 0
    with torch.no_grad():
        # for batch_idx, sample in enumerate(tqdm(val_loader)):
        for batch_idx, sample in enumerate(val_loader):
            audio_posi, visual_posi, visual_nega, target, question = (
                sample['audio_posi'].to(args.device),
                sample['visual_posi'].to(args.device),
                sample['visual_nega'].to(args.device),
                sample['label'].to(args.device),
                sample['question'].to(args.device),)

            (preds_qa, _, _, _, _, _, _,) = model(audio_posi, visual_posi,visual_nega,question)
            _, predicted = torch.max(preds_qa.data, 1)
            total_qa += preds_qa.size(0)
            correct_qa += (predicted == target).sum().item()
    logging.info('Accuracy qa: {}'.format(100 * correct_qa / total_qa))

    return 100 * correct_qa / total_qa


def test(model, test_loader):
    model.eval()
    total = 0
    correct = 0
    samples = json.load(open(args.label_test, 'r'))
    A_count = []
    A_cmp = []
    V_count = []
    V_loc = []
    AV_ext = []
    AV_count = []
    AV_loc = []
    AV_cmp = []
    AV_temp = []
    with torch.no_grad():
        # for batch_idx, sample in enumerate(tqdm(test_loader)):
        for batch_idx, sample in enumerate(test_loader):
            audio_posi, visual_posi, visual_nega, target, question = (
                sample['audio_posi'].to(args.device),
                sample['visual_posi'].to(args.device),
                sample['visual_nega'].to(args.device),
                sample['label'].to(args.device),
                sample['question'].to(args.device),)

            (preds_qa, _, _, _, _, _, _,) = model(audio_posi, visual_posi, visual_nega, question)
            preds = preds_qa
            # Get top3 predicted results
            rank = preds.clone()
            rank = rank.cpu().numpy().tolist()
            top3 = find_max(3, rank[0])
            print('Top 3: ',top3)
            
            _, predicted = torch.max(preds.data, 1)
            print('Top 1: ',predicted)
            print('Target: ',target, '\n')
            
            total += preds.size(0)
            correct += (predicted == target).sum().item()

            x = samples[batch_idx]
            type = ast.literal_eval(x['type'])
            if type[0] == 'Audio':
                if type[1] == 'Counting':
                    A_count.append((predicted == target).sum().item())
                elif type[1] == 'Comparative':
                    A_cmp.append((predicted == target).sum().item())
            elif type[0] == 'Visual':
                if type[1] == 'Counting':
                    V_count.append((predicted == target).sum().item())
                elif type[1] == 'Location':
                    V_loc.append((predicted == target).sum().item())
            elif type[0] == 'Audio-Visual':
                if type[1] == 'Existential':
                    AV_ext.append((predicted == target).sum().item())
                elif type[1] == 'Counting':
                    AV_count.append((predicted == target).sum().item())
                elif type[1] == 'Location':
                    AV_loc.append((predicted == target).sum().item())
                elif type[1] == 'Comparative':
                    AV_cmp.append((predicted == target).sum().item())
                elif type[1] == 'Temporal':
                    AV_temp.append((predicted == target).sum().item())
    
    logging.info('Audio Counting Accuracy: %.2f %%' % (100 * sum(A_count) / len(A_count)))
    logging.info('Audio Cmp Accuracy: %.2f %%' % (100 * sum(A_cmp) / len(A_cmp)))
    logging.info('Audio Accuracy: %.2f %%'% (100 * (sum(A_count) + sum(A_cmp)) / (len(A_count) + len(A_cmp))))
    logging.info('Visual Counting Accuracy: %.2f %%' % (100 * sum(V_count) / len(V_count)))
    logging.info('Visual Loc Accuracy: %.2f %%' % (100 * sum(V_loc) / len(V_loc)))
    logging.info('Visual Accuracy: %.2f %%'% (100 * (sum(V_count) + sum(V_loc)) / (len(V_count) + len(V_loc))))
    logging.info('AV Ext Accuracy: %.2f %%' % (100 * sum(AV_ext) / len(AV_ext)))
    logging.info('AV counting Accuracy: %.2f %%' % (100 * sum(AV_count) / len(AV_count)))
    logging.info('AV Loc Accuracy: %.2f %%' % (100 * sum(AV_loc) / len(AV_loc)))
    logging.info('AV Cmp Accuracy: %.2f %%' % (100 * sum(AV_cmp) / len(AV_cmp)))
    logging.info('AV Temporal Accuracy: %.2f %%' % (100 * sum(AV_temp) / len(AV_temp)))
    logging.info('AV Accuracy: %.2f %%'% (100* (sum(AV_count) + sum(AV_loc) + sum(AV_ext) + sum(AV_temp) + sum(AV_cmp))/ (len(AV_count) + len(AV_loc) + len(AV_ext) + len(AV_temp) + len(AV_cmp))))
    logging.info('Overall Accuracy: %.2f %%' % (100 * correct / total))

    return 100 * correct / total

def find_max(num, rank):
    max_index = []
    for _ in range(num):
        number = max(rank)
        index = rank.index(number)
        rank[index] = -10000
        max_index.append(index)
        
    return max_index

def main(args):
    # torch.manual_seed(args.seed + args.local_rank)
    torch.manual_seed(args.seed)
    os.environ["MASTER_ADDR"] = "localhost"
    # inplace operations (e.g., ReLU(inplace=True) or x+=y) are not allowed in the model, may results in ERRORs
    model = AVQA_Fusion_Net(args)
    if args.mode == 'train':
        logging.info("-------------- training dataset preparation --------------\n")
        train_dataset = AVQA_dataset(
            label=args.label_train,
            audio_dir=args.audio_dir,
            posi_video_dir=args.posi_video_dir,
            nega_video_dir=args.nega_video_dir,
            # transform=transforms.Compose([ToTensor()]),
            mode_flag='train',)
        sampler = DistributedSampler(train_dataset)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            sampler=sampler,)
        
        logging.info("--------- training dataset preparation completed ----------\n")
        logging.info("length of training dataset ----------- %d\n", len(train_loader))
        logging.info("-------------- validation dataset preparation --------------\n")
        val_dataset = AVQA_dataset(
            label=args.label_val,
            audio_dir=args.audio_dir,
            posi_video_dir=args.posi_video_dir,
            nega_video_dir=args.nega_video_dir,
            # transform=transforms.Compose([ToTensor()]),
            mode_flag='val',)
        val_loader = DataLoaderX(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,)
        logging.info("-------- validation dataset preparation completed ----------")
        logging.info("length of validation dataset ----------- %d", len(val_loader))

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
        # # 多间隔调整学习率
        milestones = [8, 14, 18]
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)
        criterion_qa = nn.CrossEntropyLoss()
        criterion_contr = InfoNCE()
        criterion_tri = nn.TripletMarginLoss(margin=1.0, p=2.)
        best_F = 0
        start_epoch = 1
        if args.resume:
            logging.info("-------------- loading checkpoint ----------------")
            checkpoint = torch.load(os.path.join(args.model_save_dir, args.checkpoint, 'checkpoint' + '.tar'), map_location='cpu')
            if 'module.' in next(iter(checkpoint['model_state_dict'])):
                # 当模型保存时加了module.前缀，需将其去掉
                from collections import OrderedDict

                new_state_dict = OrderedDict()
                for k, v in checkpoint['model_state_dict'].items():
                    name = k[7:]  # remove `module.`，表面从第7个key值字符取到最后一个字符，正好去掉了module.
                    new_state_dict[name] = v  # 新字典的key值对应的value为一一对应的值。
                # load params
                model.load_state_dict(new_state_dict)  # 从新加载这个模型。
            else:
                # 当模型保存时未添加module.前缀，需将其补齐
                from collections import OrderedDict

                new_state_dict = OrderedDict()
                for k, v in checkpoint['model_state_dict'].items():
                    name = 'module.' + k  # 加上'module.'前缀
                    new_state_dict[name] = v  # 新字典的key值对应的value为一一对应的值。
                model.load_state_dict(new_state_dict)  # 从新加载这个模型。
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            best_F = checkpoint['Acc']
            logging.info("-------- checkpoint loading successfully ----------")

        model = model.to(args.device)
        model = DDP(model,device_ids=[args.local_rank],output_device=args.local_rank,find_unused_parameters=True,)
        logging.info("-------------- start training --------------\n")
        for epoch in range(start_epoch, args.epoches + 1):
            sampler.set_epoch(epoch)
            train(args,model,train_loader,optimizer,criterion_qa,criterion_contr,criterion_tri,epoch=epoch,)
            scheduler.step()
            F = eval(model, val_loader, epoch)
            if args.local_rank == 0:
                model_save_path = os.path.join(args.model_save_dir, args.checkpoint)
                if not os.path.isdir(model_save_path):
                    os.mkdir(model_save_path)
                # Save weights of the network, only save the model it-self
                model_to_save = model.module if hasattr(model, 'module') else model
                optimizer_to_save = optimizer
                (ckpt_name, best_F) = (('checkpoint', best_F) if best_F >= F else ('best', F))
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': model_to_save.state_dict(),
                        'optimizer_state_dict': optimizer_to_save.state_dict(),
                        'Acc': best_F,
                    },
                    os.path.join(model_save_path, ckpt_name + '.tar'),
                )
        logging.info("-------------- end of training --------------\n")
        
    elif args.mode == 'test':
        # model = AVQA_Fusion_Net(args)
        test_dataset = AVQA_dataset(
            label=args.label_test,
            audio_dir=args.audio_dir,
            posi_video_dir=args.posi_video_dir,
            nega_video_dir=args.nega_video_dir,
            # transform=transforms.Compose([ToTensor()]),
            mode_flag='test',)
        test_loader = DataLoaderX(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        logging.info("length of test dataset ----------- %d", len(test_loader))
        logging.info("-------------- loading checkpoint ----------------")
        checkpoint = torch.load(os.path.join(args.model_save_dir, args.checkpoint, 'best' '.tar'))
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(args.device)
        model = DDP(model,device_ids=[args.local_rank],output_device=args.local_rank,find_unused_parameters=True,)
        end_epoch = checkpoint['epoch']
        best_F = checkpoint['Acc']
        logging.info("-------- checkpoint loading successfully ----------")
        logging.info('********************************************************')
        logging.info('the best epoch ---------- {0}'.format(end_epoch))
        logging.info('the best train acc ------ {0}'.format(best_F))
        logging.info('********************************************************')
        test(model, test_loader)


if __name__ == '__main__':
    args = config.args
    # 给主要进程（args.local_rank=0）设置低输出等级，给其他进程设置高输出等级。
    logging.basicConfig(level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logging.info('********************************************************')
    logging.info('****************** Experiment setting ******************')
    logging.info('label_train dir --------- {0}'.format(args.label_train))
    logging.info('label_val dir ----------- {0}'.format(args.label_val))
    logging.info('label_test dir ---------- {0}'.format(args.label_test))
    logging.info('audio dir --------------- {0}'.format(args.audio_dir))
    logging.info('posi_video dir ---------- {0}'.format(args.posi_video_dir))
    logging.info('nega_video dir ---------- {0}'.format(args.nega_video_dir))
    logging.info('batch size -------------- {0}'.format(args.batch_size))
    logging.info('num of epoches ---------- {0}'.format(args.epoches))
    logging.info('learning rate ----------- {0}'.format(args.lr))
    logging.info('name of model ----------- {0}'.format(args.model))
    logging.info('train or test ----------- {0}'.format(args.mode))
    logging.info('model save dir ---------- {0}'.format(args.model_save_dir + args.checkpoint))
    logging.info('num of dataloader workers {0}'.format(args.num_workers))
    logging.info('gpu id ------------------ {0}'.format(args.gpu))
    logging.info('********************************************************')
    # define the order of visible GPU to our appointing cards
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        # local_rank is the index of gpu, e.g., 0 or 1 or 2 or 3
        args.device = torch.device('cuda', args.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')

    logging.info("------------ Audio-Visual Spatial-Temporal Model ------------ \n")

    main(args)

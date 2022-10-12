import argparse
import os

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Implementation of Audio-Visual Question Answering')
parser.add_argument("--audio_dir",type=str,default='./data/feats/vggish',help="audio dir",)
parser.add_argument("--posi_video_dir",type=str,default='/data/avst/10f_video_2plus1/',help="posi_video dir",)
parser.add_argument("--nega_video_dir",type=str,default='/data/avst/10f_video_2plus1/',help="nega_video dir",)
parser.add_argument("--label_train",type=str,default="./data/json_update/avqa-train.json", help="train csv file",)
parser.add_argument("--label_val",type=str,default="./data/json_update/avqa-val.json",help="val csv file",)
parser.add_argument("--label_test",type=str,default="./data/json_update/avqa-test.json",help="test csv file",)
parser.add_argument('--epoches',type=int,default=15,metavar='N',help='number of epoches to train (default: 60)',)
parser.add_argument('--lr',type=float,default=1e-4,metavar='LR',help='learning rate (default: 3e-4)',)
parser.add_argument("--model", type=str, default='AVQA_Fusion_Net', help="with model to use")   
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--log-interval',type=int,default=100,metavar='N',help='how many batches to wait before logging training status',)
parser.add_argument("--model_save_dir",type=str,default='./data/checkpoints/',help="model save dir",)
parser.add_argument('--local_rank', type=int, help="local gpu id")
parser.add_argument('--world_size', type=int, help="num of processes")
########################   regularly modified configuration   ########################
parser.add_argument('--batch-size',type=int,default=24,metavar='N')
parser.add_argument("--resume",type=str,default=False,help="load the checkpoint to resume the training",)
parser.add_argument('--num_workers', type=int, default='2', help='num_workers of dataloader')
parser.add_argument("--mode", type=str, default='test', help="with mode to use")
parser.add_argument("--checkpoint",type=str,default='checkpoint_loc_glo_3D_end',help="folder name of saved model",)
parser.add_argument('--gpu', type=str, default='0,1,2,3', help='gpu device number')
####################################################################################

args = parser.parse_args()
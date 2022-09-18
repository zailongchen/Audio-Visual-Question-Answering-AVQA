import argparse
import os

root_dir = os.getcwd()
# Training settings
parser = argparse.ArgumentParser(description='PyTorch Implementation of Audio-Visual Question Answering')
parser.add_argument("--audio_dir",type=str,default='./data/feats/vggish',help="audio dir",)
parser.add_argument("--posi_video_dir",type=str,default='/data/frames_1fps/10f_video/',help="posi_video dir",)
parser.add_argument("--nega_video_dir",type=str,default='/data/frames_1fps/10f_video/',help="nega_video dir",)
parser.add_argument("--label_train",type=str,default="./data/json/avqa-train.json", help="train csv file",)
parser.add_argument("--label_val",type=str,default="./data/json/avqa-val.json",help="val csv file",)
parser.add_argument("--label_test",type=str,default="./data/json/avqa-test.json",help="test csv file",)
parser.add_argument('--epoches',type=int,default=20,metavar='N',help='number of epoches to train (default: 60)',)
parser.add_argument('--lr',type=float,default=1e-4,metavar='LR',help='learning rate (default: 3e-4)',)
parser.add_argument("--model", type=str, default='AVQA_Fusion_Net', help="with model to use")   
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--log-interval',type=int,default=10,metavar='N',help='how many batches to wait before logging training status',)
parser.add_argument("--model_save_dir",type=str,default='./data/checkpoints/',help="model save dir",)
parser.add_argument('--local_rank', type=int, help="local gpu id")
parser.add_argument('--world_size', type=int, help="num of processes")
########################   regularly modified configuration   ########################
parser.add_argument('--num_frames',type=int,default=10,metavar='N')
parser.add_argument('--batch-size',type=int,default=32,metavar='N')
parser.add_argument("--resume",type=str,default=False,help="load the checkpoint to resume the training",)
parser.add_argument('--num_workers', type=int, default='0', help='num_workers of dataloader')
parser.add_argument("--mode", type=str, default='train', help="with mode to use")
parser.add_argument("--checkpoint",type=str,default='avst_mimn_coatt',help="folder name of saved model",)
parser.add_argument('--gpu', type=str, default='0,1', help='gpu device number')
####################################################################################

args = parser.parse_args()
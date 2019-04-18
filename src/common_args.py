import argparse
import datetime
import os
import torch

def get_args():
    parser = argparse.ArgumentParser(description='face recognition')
    # Set log names
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    dataset = 'LFW'
    logname = '{}_{}_{}'.format(dataset, current_time, 'runlog.txt')
    logpath = '../log'
    log_path = os.path.join(logpath, logname)
    parser.add_argument('--log_path', default=log_path, type=str, help='log path')
    parser.add_argument('--WebFace_path', default='/home/aaron/projects/FYP_Face_Verification/data/WebFace.csv', type=str, help='WebFace dataset path')

    # PCA
    parser.add_argument('--usePCA', default=False, type=bool, help='use PCA or not')
    parser.add_argument('--rgb2gray', default=True, type=bool, help='use gray image or not')
    parser.add_argument('--num_components', default=50, type=int, help='number of principal components for pca ')

    # LBP
    parser.add_argument('--useLBP', default=False, type=bool, help='use LBP or not')
    parser.add_argument('--sub_region_height', default=8, type=int, help='height of sub regions for LBP')
    parser.add_argument('--sub_region_width', default=8, type=int, help='width of sub regions for LBP')
    parser.add_argument('--uniform', default=True, type=bool, help='use uniform pattern LBP or not')

    # Spherenet20
    loss_type = 'amsoftmax'
    parser.add_argument('--useDeepFace', default=True, type=bool, help='use Deep Neural Network or not')
    parser.add_argument('--use_pool', default=False, type=bool, help='use global pooling?')
    parser.add_argument('--use_dropout', default=False, type=bool, help='use dropout?')
    parser.add_argument('--feature_dim', default=512, type=int, help='feature dimension')
    parser.add_argument('--gpu_ids', default='0,1', type=str, help='gpu ids: e.g. 0 0,1,2, 0,2.')
    parser.add_argument('--loss_type', default=loss_type, type=str,help='softmax/asoftmax/amsoftmax')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--bs', default=256, type=int, help='default: 256')
    parser.add_argument('--iterations', default=2000, type=int, help='number of training epochs')
    parser.add_argument('--decay_steps', default='16000, 24000, 28000', type=str, help='The step where learning rate decay by 0.1')
    parser.add_argument('--weight_decay', default=0.0005, type=float, help='weight decay')
    parser.add_argument('--use_f_norm', default=True, type=bool, help='feature normalization?')
    parser.add_argument('--use_w_norm', default=True, type=bool, help='weight normalization?')
    parser.add_argument('--s', default=20, type=float, help='to re-scale feature norm')
    parser.add_argument('--m_1', default=4, type=float, help='margin of SphereFace')
    parser.add_argument('--m_3', default=0.35, type=float, help='margin of CosineFace')
    parser.add_argument('--eval_freq', default=1000, type=int, help='frequency of evaluation')

    # Spherenet20 LR
    model_root = '/home/tangjiawei/project/fyp/saved/30000_net_backbone.pth'
    parser.add_argument('--downsampling_factor', default=9, type=int, help='downsampling factor')
    parser.add_argument('--model_root', default=model_root, type=str)

    args = parser.parse_args()
    args.device = torch.device(1)
    return args


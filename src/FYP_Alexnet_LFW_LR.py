import pandas as pd
import cv2
import numpy as np
from matlab_cp2tform import get_similarity_transform_for_cv2
import torch
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from sklearn import preprocessing
# from mtcnn.mtcnn import MTCNN
from torchvision import transforms
import torch
import torchvision.models as models
#from torchsummary import summary

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
import os
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import accuracy_score, roc_curve, auc
import argparse
import random
import math
import datetime
from tqdm import tqdm
from util import save_log, KFold, find_best_threshold, eval_acc, tensor_pair_cosine_distance
import csv
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve

import logging
logging.basicConfig(filename='FYP_Alexnet.log', level=logging.INFO)
import matplotlib.pyplot as plt

# Set log names
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
dataset = 'LFW'
logname = '{}_{}_{}'.format(dataset, current_time, 'runlog.txt')
logpath = '../log'
log_path = os.path.join(logpath, logname)

parser = argparse.ArgumentParser(description='AlexNet_LFW')
parser.add_argument('--lr', default=0.0003, type=float, help='learning rate')
parser.add_argument('--batch_size', default=10, type=int, help='')
parser.add_argument('--num_epochs', default=200, type=int, help='')
parser.add_argument('--model_root', default='//home/tangjiawei/project/fyp/saved/FYP_test.pkl')
parser.add_argument('--model_root2', default='//home/tangjiawei/project/fyp/saved/FYP_test2.pkl')
parser.add_argument('--model_root_lr', default='//home/tangjiawei/project/fyp/saved/FYP_test_lr.pkl')# Set log names model_root_lr_2
parser.add_argument('--model_root_lr_2', default='//home/tangjiawei/project/fyp/saved/FYP_test_lr_12.pkl')
parser.add_argument('--iterations', default=5000, type=int, help='number of training epochs')

parser.add_argument('--log_path', default=log_path, type=str, help='log path')

args = parser.parse_args()

downsampling_factor = 7

def alignment(src_img, src_pts, size=None):
    #left eye, right eye, nose, mouth left, mouth right
    ref_pts = [[30.2946, 51.6963], [65.5318, 51.5014],
               [48.0252, 71.7366], [33.5493, 92.3655],
               [62.7299, 92.2041]]
    ref_pts = [[30.2946, 31.6963], [65.5318, 31.5014],
               [48.0252, 51.7366], [33.5493, 82.3655],
               [62.7299, 82.2041]]
    if size is not None:
        ref_pts = np.array(ref_pts)
        ref_pts[:,0] = ref_pts[:,0] * size/96
        ref_pts[:,1] = ref_pts[:,1] * size/96
        crop_size = (int(size), int(112/(96/size)))
    else:
        crop_size = (96, 112)
    src_pts = np.array(src_pts).reshape(5, 2)
    s = np.array(src_pts).astype(np.float32)
    r = np.array(ref_pts).astype(np.float32)
    tfm = get_similarity_transform_for_cv2(s, r)
    face_img = cv2.warpAffine(src_img, tfm, crop_size, flags=cv2.INTER_CUBIC)
    if size is not None:
        face_img = cv2.resize(face_img, dsize=(96, 112), interpolation=cv2.INTER_CUBIC)
    return face_img


def face_ToTensor(img):
    return (ToTensor()(img) - 0.5) * 2

# _lfw_root = '/home/jiawei/Documents/polyU/FYP/data_set/lfw/'
# _lfw_landmarks = '/home/jiawei/Documents/polyU/FYP/data_set/LFW.csv'
# _lfw_pairs = '../data/lfw_pairs.txt'
# _casia_root = '/home/johnny/Datasets/CASIA-WebFace/CASIA-WebFace/'
# _casia_landmarks = '../data/CASIA-maxpy-clean_remove_lfw_megaface.csv'
# _scface_root = '/home/jiawei/Documents/polyU/FYP/MDS_MATLAB/data/sc/HR/'

_lfw_root = '/home/tangjiawei/project/dataset/lfw/'
_lfw_landmarks = '/home/tangjiawei/project/dataset/LFW.csv'
_lfw_pairs = '/home/tangjiawei/project/dataset/lfw_pairs.txt'
_casia_root = '/home/tangjiawei/project/dataset/CASIA-WebFace/'
_casia_landmarks = '/home/tangjiawei/project/dataset/CASIA-maxpy-clean_remove_lfw.csv'
_hr_alexnet_result = '//home/tangjiawei/project/fyp/saved/hr_alexnet_result.csv'

#use for PCA and MDS
def load_LFWFace(is_aug=True):
    data = []
    df = pd.read_csv(_lfw_landmarks, delimiter=",", header=None)
    numpyMatrix = df.values
    faces_path = numpyMatrix[:, 0]
    landmarks = numpyMatrix[:, 1:]
    class_id = [path.split('/')[0] for path in faces_path]
    unique_class_id = np.unique(class_id)
    le = preprocessing.LabelEncoder()
    le.fit(unique_class_id)
    targets = le.transform(class_id).reshape(-1, 1)
    for index in range(len(faces_path)):
        face = cv2.imread(_lfw_root + faces_path[index])
        face = alignment(face, landmarks[index].reshape(-1, 2))
        if is_aug:
            if np.random.random() > 0.5:
                face = cv2.flip(face, 1)
        data.append(face)

    data = np.asarray(data, 'uint8')

    return data, targets


#use for alexnet
class LFWDataset(torch.utils.data.Dataset):
    def __init__(self, size=False, is_aug=True):
        super(LFWDataset, self).__init__()
        # Original num. subjects(num. images): 10,575(494,414)
        df = pd.read_csv(_lfw_landmarks, delimiter=",", header=None)
        numpyMatrix = df.values
        self.transform = transforms.Compose([
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                             ])
        self.faces_path = numpyMatrix[:, 0]
        self.landmarks = numpyMatrix[:, 1:]
        self.targets, self.num_class = self.get_targets()
        self.is_aug = is_aug
        self.size = size

    def __getitem__(self, index):
        face = cv2.imread(_lfw_root + self.faces_path[index])
        face = alignment(face, self.landmarks[index].reshape(-1,2))
        if self.is_aug:
            if np.random.random() > 0.5:
                face = cv2.flip(face, 1)
        face = cv2.resize(face, (224, 224))
        return self.transform(face), torch.LongTensor(self.targets[index])

    def __len__(self):
        return self.faces_path.shape[0]

    def get_targets(self):
        class_id = [path.split('/')[0] for path in self.faces_path]
        unique_class_id = np.unique(class_id)
        le = preprocessing.LabelEncoder()
        le.fit(unique_class_id)
        targets = le.transform(class_id).reshape(-1,1)
        num_class = unique_class_id.shape[0]
        return targets.astype(np.float32), num_class

class LFWDataset_gz(torch.utils.data.Dataset):
    def __init__(self):
        super(LFWDataset_gz, self).__init__()
        df = pd.read_csv(_lfw_landmarks, delimiter=",", header=None)
        numpyMatrix = df.values
        self.landmarks = numpyMatrix[:, 1:]
        self.df = df
        with open(_lfw_pairs) as f:
            pairs_lines = f.readlines()[1:]
        self.pairs_lines = pairs_lines
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        p = self.pairs_lines[index].replace('\n', '').split('\t')
        if 3 == len(p):
            sameflag = np.int32(1).reshape(1)
            name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
            name2 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[2]))
        if 4 == len(p):
            sameflag = np.int32(0).reshape(1)
            name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
            name2 = p[2] + '/' + p[2] + '_' + '{:04}.jpg'.format(int(p[3]))
        img1 = alignment(cv2.imread(_lfw_root + name1),
                         self.landmarks[self.df.loc[self.df[0] == name1].index.values[0]])
        img2 = alignment(cv2.imread(_lfw_root + name2),
                         self.landmarks[self.df.loc[self.df[0] == name2].index.values[0]])
        img1 = cv2.resize(img1, (224, 224))
        img2 = cv2.resize(img2, (224, 224))
        img1_flip = cv2.flip(img1, 1)
        img2_flip = cv2.flip(img2, 1)
        return self.transform(img1), self.transform(img2), \
               self.transform(img1_flip), self.transform(img2_flip), \
               torch.LongTensor(sameflag)

    def __len__(self):
        return len(self.pairs_lines)

class LFWDataset_gz_lr(torch.utils.data.Dataset):
    def __init__(self, N = 1):
        super(LFWDataset_gz_lr, self).__init__()
        df = pd.read_csv(_lfw_landmarks, delimiter=",", header=None)
        numpyMatrix = df.values
        self.landmarks = numpyMatrix[:, 1:]
        self.df = df
        self.N = N
        with open(_lfw_pairs) as f:
            pairs_lines = f.readlines()[1:]
        self.pairs_lines = pairs_lines
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        p = self.pairs_lines[index].replace('\n', '').split('\t')
        if 3 == len(p):
            sameflag = np.int32(1).reshape(1)
            name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
            name2 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[2]))
        if 4 == len(p):
            sameflag = np.int32(0).reshape(1)
            name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
            name2 = p[2] + '/' + p[2] + '_' + '{:04}.jpg'.format(int(p[3]))
        img1 = alignment(cv2.imread(_lfw_root + name1),
                         self.landmarks[self.df.loc[self.df[0] == name1].index.values[0]])
        img2 = alignment(cv2.imread(_lfw_root + name2),
                         self.landmarks[self.df.loc[self.df[0] == name2].index.values[0]])
        img1 = cv2.resize(img1, (int(112/self.N), int(96/self.N)))
        img2 = cv2.resize(img2, (int(112/self.N), int(96/self.N)))
        img1 = cv2.resize(img1, (224, 224))
        img2 = cv2.resize(img2, (224, 224))
        img1_flip = cv2.flip(img1, 1)
        img2_flip = cv2.flip(img2, 1)
        return self.transform(img1), self.transform(img2), \
               self.transform(img1_flip), self.transform(img2_flip), \
               torch.LongTensor(sameflag)

    def __len__(self):
        return len(self.pairs_lines)


def Load_SCFace_path():
    source_face = []
    source_label = []
    target_face = []
    target_label = []
    flag = []
    shuffle_index = np.arange(200)
    np.random.shuffle(shuffle_index)
    for i in range(200):
        img_path = _scface_root + str(i + 1) + '.jpg'
        label = (i // 5)
        source_face.append(img_path)
        source_label.append(label)
    for i in range(200):
        target_face.append(source_face[((i+2) % 200)])
        if source_label[i] == source_label[((i+2) % 200)]:
            sameflag = np.int32(1).reshape(1)
        else:
            sameflag = np.int32(0).reshape(1)
        flag.append(sameflag)
    flag = np.asarray(flag, 'uint8')
    return source_face, target_face, flag

class SCFDataset(torch.utils.data.Dataset):
    def __init__(self):
        super(SCFDataset, self).__init__()
        source_face, target_face, flag = Load_SCFace_path()
        self.source_face = source_face
        self.target_face = target_face
        self.flag = flag
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # self.landmarks = numpyMatrix[:, 1:]
        # self.df = df
        # with open(_lfw_pairs) as f:
        #     pairs_lines = f.readlines()[1:]
        # self.pairs_lines = pairs_lines

    def __getitem__(self, index):
        img1 = cv2.imread(self.source_face[index])
        img2 = cv2.imread(self.target_face[index])
        img1 = cv2.resize(img1, (224, 224))
        img2 = cv2.resize(img2, (224, 224))


        img1_flip = cv2.flip(img1, 1)
        img2_flip = cv2.flip(img2, 1)
        return self.transform(img1), self.transform(img2),\
               self.transform(img1_flip), self.transform(img2_flip), torch.LongTensor(self.flag[index])

    # def __getitem__(self, index):
    #     detector = MTCNN()
    #     img1 = cv2.imread(self.source_face[index])
    #     img1_MTCNN_result = detector.detect_faces(img1)
    #     img1_landmarks = img1_MTCNN_result[0]["keypoints"]
    #     img1_landmarks_numpyMatrix = [img1_landmarks["left_eye"][0], img1_landmarks["left_eye"][1],
    #                                   img1_landmarks["right_eye"][0], img1_landmarks["right_eye"][1],
    #                                   img1_landmarks["nose"][0], img1_landmarks["nose"][1],
    #                                   img1_landmarks["mouth_left"][0], img1_landmarks["mouth_left"][1],
    #                                   img1_landmarks["mouth_right"][0], img1_landmarks["mouth_right"][1]]
    #     img1_aligned = alignment(img1, img1_landmarks_numpyMatrix)
    #
    #     img2 = cv2.imread(self.target_face[index])
    #     img2_MTCNN_result = detector.detect_faces(img2)
    #     img2_landmarks = img2_MTCNN_result[0]["keypoints"]
    #     img2_landmarks_numpyMatrix = [img2_landmarks["left_eye"][0], img2_landmarks["left_eye"][1],
    #                                   img2_landmarks["right_eye"][0], img2_landmarks["right_eye"][1],
    #                                   img2_landmarks["nose"][0], img2_landmarks["nose"][1],
    #                                   img2_landmarks["mouth_left"][0], img2_landmarks["mouth_left"][1],
    #                                   img2_landmarks["mouth_right"][0], img2_landmarks["mouth_right"][1]]
    #     img2_aligned = alignment(img2, img2_landmarks_numpyMatrix)
    #
    #     img1_aligned = cv2.resize(img1_aligned, (224, 224))
    #     img2_aligned = cv2.resize(img2_aligned, (224, 224))
    #     img1_flip = cv2.flip(img1_aligned, 1)
    #     img2_flip = cv2.flip(img2_aligned, 1)
    #     return self.transform(img1_aligned), self.transform(img2_aligned),\
    #            self.transform(img1_flip), self.transform(img2_flip), torch.LongTensor(self.flag[index])

    def __len__(self):
        return len(self.source_face)

class WebFaceDataset(torch.utils.data.Dataset):
    def __init__(self, size=False, is_aug=True):
        super(WebFaceDataset, self).__init__()
        # Original num. subjects(num. images): 10,575(494,414)
        df = pd.read_csv(_casia_landmarks, delimiter=",", header=None)
        numpyMatrix = df.values
        self.faces_path = numpyMatrix[:, 0]
        self.landmarks = numpyMatrix[:, 1:]
        self.targets, self.num_class = self.get_targets()
        self.is_aug = is_aug
        self.size = size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    def __getitem__(self, index):
        face = cv2.imread(_casia_root + self.faces_path[index])
        face = alignment(face, self.landmarks[index].reshape(-1, 2))
        lr_face = cv2.resize(face, (int(96/downsampling_factor), int(112/downsampling_factor)))       #第一次 16 16
        face = cv2.resize(face, (224, 224))
        lr_face = cv2.resize(lr_face, (224, 224))
        if self.is_aug:
            if np.random.random() > 0.5:
                face = cv2.flip(face, 1)
                lr_face = cv2.flip(lr_face, 1)
        return self.transform(face), self.transform(lr_face), torch.LongTensor(self.targets[index])

    def __len__(self):
        return self.faces_path.shape[0]

    def get_targets(self):
        class_id = [path.split('/')[0] for path in self.faces_path]
        unique_class_id = np.unique(class_id)
        le = preprocessing.LabelEncoder()
        le.fit(unique_class_id)
        targets = le.transform(class_id).reshape(-1,1)
        num_class = unique_class_id.shape[0]
        return targets.astype(np.float32), num_class

class get_loader():
    def __init__(self,  name, batch_size, N = 1, is_aug=True, shuffle=True, drop_last=True):
        if name == 'lfw':
            dataset = LFWDataset()
            num_class = dataset.num_class
        if name == 'webface':
            dataset = WebFaceDataset(is_aug=is_aug)
            num_class = dataset.num_class
        if name == 'scf':
            dataset = SCFDataset()
            num_class = None
            shuffle = False
            drop_last = False
        if name == 'LFWDataset_gz':
            dataset = LFWDataset_gz()
            num_class = None
            shuffle = False
            drop_last = False
        if name == 'LFWDataset_gz_lr':
            dataset = LFWDataset_gz_lr(N)
            num_class = None
            shuffle = False
            drop_last = False
        self.dataloader = DataLoader(dataset=dataset, num_workers=4, batch_size=batch_size,
                                     pin_memory=True, shuffle=shuffle, drop_last=drop_last)
        self.num_class = num_class
        self.train_iter = iter(self.dataloader)


    def next(self):
        try:
            data = next(self.train_iter)
        except:
            self.train_iter = iter(self.dataloader)
            data = next(self.train_iter)
        return data

def SCF_verification(net, feature_dim, device, step = None):
    net.eval()
    dataloader = get_loader('scf', batch_size=50).dataloader
    features11_total = torch.Tensor(np.zeros((200, feature_dim), dtype=np.float32)).to()
    features12_total, features21_total, features22_total = torch.zeros_like(features11_total), \
                                                           torch.zeros_like(features11_total), \
                                                           torch.zeros_like(features11_total)
    labels = torch.Tensor(np.zeros((200, 1), dtype=np.float32)).to(device)
    with torch.no_grad():
        bs_total = 0
        for index, (img1, img2, img1_flip, img2_flip, flag) in enumerate(dataloader):
            bs = len(flag)
            img1, img1_flip = img1.to(device), img1_flip.to(device)
            img2, img2_flip = img2.to(device), img2_flip.to(device)
            features11 = net(img1)
            features12 = net(img1_flip)
            features21 = net(img2)
            features22 = net(img2_flip)
            features11_total[bs_total:bs_total + bs] = features11
            features12_total[bs_total:bs_total + bs] = features12
            features21_total[bs_total:bs_total + bs] = features21
            features22_total[bs_total:bs_total + bs] = features22
            labels[bs_total:bs_total + bs] = flag
            bs_total += bs
        assert bs_total == 200, print('SCF pairs should be 200!')
    labels = labels.cpu().numpy()

    for cal_type in ['concat']:  # cal_type: concat/sum/normal
        scores = tensor_pair_cosine_distance(features11_total, features12_total, features21_total, features22_total,
                                             type=cal_type)
        fpr, tpr, _ = roc_curve(labels, scores)  # false positive rate / true positive rate
        roc_auc = auc(fpr, tpr)
        print(roc_auc)

def lfw_verification(net, feature_dim, device, resolution = 'hr', N =1, step=None):
    net.eval()
    if resolution == 'hr':
        dataloader = get_loader('LFWDataset_gz', batch_size=256).dataloader
    elif resolution == 'lr':
        dataloader = get_loader('LFWDataset_gz_lr', batch_size=256, N=N).dataloader
    # 6000 Image Pairs ( 3000 genuine and 3000 impostor matches)
    features11_total = torch.Tensor(np.zeros((6000, feature_dim), dtype=np.float32)).to(device)
    features12_total, features21_total, features22_total = torch.zeros_like(features11_total), \
                                                           torch.zeros_like(features11_total), \
                                                           torch.zeros_like(features11_total)
    labels = torch.Tensor(np.zeros((6000, 1), dtype=np.float32)).to(device)
    with torch.no_grad():
        bs_total = 0
        for index, (img1, img2, img1_flip, img2_flip, targets) in enumerate(dataloader):
            bs = len(targets)
            img1, img1_flip = img1.to(device), img1_flip.to(device)
            img2, img2_flip = img2.to(device), img2_flip.to(device)
            features11 = net(img1)
            features12 = net(img1_flip)
            features21 = net(img2)
            features22 = net(img2_flip)
            features11_total[bs_total:bs_total + bs] = features11
            features12_total[bs_total:bs_total + bs] = features12
            features21_total[bs_total:bs_total + bs] = features21
            features22_total[bs_total:bs_total + bs] = features22
            labels[bs_total:bs_total + bs] = targets
            bs_total += bs
        assert bs_total == 6000, print('LFW pairs should be 6,000!')
    labels = labels.cpu().numpy()
    for cal_type in ['concat']:  # cal_type: concat/sum/normal
        scores = tensor_pair_cosine_distance(features11_total, features12_total, features21_total, features22_total, type=cal_type)
        fpr, tpr, _ = roc_curve(labels, scores) # false positive rate / true positive rate
        # if step == args.iterations:
        #     np.savez('cnn_roc_{}.npz'.format(args.loss_type), name1=fpr, name2=tpr)
        #     message = 'The fpr and tpr is saved to cnn_roc_{}.npz'.format(args.loss_type)
        #     save_log(message, args.log_path)
        roc_auc = auc(fpr, tpr)
        accuracy = []
        thd = []
        folds = KFold(n=6000, n_folds=10, shuffle=False) # 1 for test, 9 for train
        thresholds = np.linspace(-10000, 10000, 10000 + 1)
        thresholds = thresholds / 10000
        predicts = np.hstack((scores, labels))
        for idx, (train, test) in enumerate(folds):
            best_thresh = find_best_threshold(thresholds, predicts[train]) # find the best threshold to say if these two persons are same or not
            accuracy.append(eval_acc(best_thresh, predicts[test]))
            thd.append(best_thresh)
        mean_acc, std = np.mean(accuracy), np.std(accuracy)  # have 10 accuracies and then find the mean and std
        if step is not None:
            message = 'LFWACC={:.4f} std={:.4f} auc={:.4f} type:{} at {}iter.'.format(mean_acc, std, roc_auc, cal_type, step)
        else:
            message = 'LFWACC={:.4f} std={:.4f} auc={:.4f} type:{} at testing.'.format(mean_acc, std, roc_auc, cal_type)
        if step is None:
            np.savez('/home/tangjiawei/PycharmProjects/FYP_Face_Verification/saved/alexnet_hr/resnet_roc_{}.npz'.format(N), name1=fpr, name2=tpr)
        print(message)



if __name__ == '__main__':

    # setup training dataset and device
    dataloader = get_loader('webface', 256)
    class_num = dataloader.num_class
    device = torch.device(1)
    print(device)

    #setup hr_alexnet
    hr_alexnet = models.alexnet(pretrained=True)
    hr_alexnet.classifier[6] = nn.Linear(4096, class_num)
    hr_alexnet.load_state_dict(torch.load('/home/tangjiawei/project/fyp/saved/ALEX_NET_HR_11_April.pkl')) #model_root2 is Alexnet prtrained in HR Webface dataset
    hr_features_net = hr_alexnet.features
    classifier = hr_alexnet.classifier
    hr_alexnet.to(device)
    classifier.to(device)
    hr_features_net.to(device)

    # setup lr_alexnet
    lr_alexnet = models.alexnet(pretrained=True)
    lr_alexnet.classifier[6] = nn.Linear(4096, class_num)
    lr_alexnet.to(device)
    lr_alexnet.load_state_dict(torch.load('/home/tangjiawei/project/fyp/saved/ALEX_NET_HR_11_April.pkl')) #model_root_lr is Alexnet prtrained in with feature loss
    lr_features_net = lr_alexnet.features

    # # setup test model
    # lr_alexnet = models.alexnet(pretrained=True)
    # lr_alexnet.classifier[6] = nn.Linear(4096, class_num)
    # lr_alexnet.to(device)
    # lr_alexnet.load_state_dict(torch.load(
    #     '/home/tangjiawei/project/fyp/saved/ALEX_NET_HR_11_April.pkl'))  # model_root_lr is Alexnet prtrained in with feature loss
    # lr_features_net = lr_alexnet.features

    #setup criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    criterion_lr = torch.nn.MSELoss()
    optimizer = optim.Adam(lr_features_net.parameters(),
                           lr=args.lr,
                           weight_decay=0.0005,
                           betas=(0.9, 0.999),
                           amsgrad=True)
    scheduler = StepLR(optimizer, step_size=20001, gamma=0.1)

    # # validation
    # for i in range(16):
    #     lfw_verification(lr_alexnet, class_num, device, 'lr', i+1)


    # ------------------------
    # Start Training and Validating
    # ------------------------
    iterations = 20000
    pbar = tqdm(range(1, iterations + 1), ncols=0)
    for steps_num in pbar:
        lr_features_net.train()

        hr_features_net.eval()
        hr_alexnet.eval()

        print('Epoch {}/{}'.format(steps_num + 1, iterations))
        print('-' * 10)
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        optimizer.zero_grad()

        hr_face, lr_face, target = dataloader.next()
        hr_face = hr_face.to(device)
        lr_face = lr_face.to(device)
        target = target.to(device)
        hr_features = hr_features_net(hr_face)
        lr_features = lr_alexnet.features(lr_face)
        # features_diff = hr_features - lr_features
        # features_diff = features_diff.view(features_diff.size(0), 256 * 6 * 6)
        # output = classifier(features_diff)
        # _, pred_target = torch.max(output, 1)
        # loss = criterion(output, target.view(-1))
        loss = criterion_lr(hr_features, lr_features)
        # # write by Guan zhong, transform to loss with requires_grad directly
        # loss = np.linalg.norm(hr_features.cpu().detach().numpy() - lr_features.cpu().detach().numpy())
        # #loss = np.array(loss).reshape(1, 1)
        # # loss = ToTensor()(loss)
        # loss = torch.tensor(loss, requires_grad=True)


        # loss = MyLoss(hr_features, lr_features)
        loss.to(device)

        loss.backward()
        optimizer.step()
        if steps_num == 10:
            torch.save(lr_alexnet.state_dict(), '/home/tangjiawei/PycharmProjects/RadimoicDeepFeatureExtraction/model/{}_lr_alexnet_df{}.pth'.format(steps_num, downsampling_factor))
            print("HR image result of LR spherenet")
            logging.info("HR image result of LR spherenet")
            lfw_verification(lr_alexnet, class_num, device, 'lr', 1)
            print("LR image result of LR spherenet")
            logging.info("LR image result of LR spherenet")
            lfw_verification(lr_alexnet, class_num, device, 'lr', 9)
        if steps_num == 100:
            torch.save(lr_alexnet.state_dict(),
                       '/home/tangjiawei/PycharmProjects/RadimoicDeepFeatureExtraction/model/{}_lr_alexnet_df{}.pth'.format(
                           steps_num, downsampling_factor))
            print("HR image result of LR spherenet")
            logging.info("HR image result of LR spherenet")
            lfw_verification(lr_alexnet, class_num, device, 'lr', 1)
            print("LR image result of LR spherenet")
            logging.info("LR image result of LR spherenet")
            lfw_verification(lr_alexnet, class_num, device, 'lr', 9)
        if steps_num == 1000:
            torch.save(lr_alexnet.state_dict(),
                       '/home/tangjiawei/PycharmProjects/RadimoicDeepFeatureExtraction/model/{}_lr_alexnet_df{}.pth'.format(
                           steps_num, downsampling_factor))
            print("HR image result of LR spherenet")
            logging.info("HR image result of LR spherenet")
            lfw_verification(lr_alexnet, class_num, device, 'lr', 1)
            print("LR image result of LR spherenet")
            logging.info("LR image result of LR spherenet")
            lfw_verification(lr_alexnet, class_num, device, 'lr', 9)
        if steps_num == 2000:
            torch.save(lr_alexnet.state_dict(),
                       '/home/tangjiawei/PycharmProjects/RadimoicDeepFeatureExtraction/model/{}_lr_alexnet_df{}.pth'.format(
                           steps_num, downsampling_factor))
            print("HR image result of LR spherenet")
            logging.info("HR image result of LR spherenet")
            lfw_verification(lr_alexnet, class_num, device, 'lr', 1)
            print("LR image result of LR spherenet")
            logging.info("LR image result of LR spherenet")
            lfw_verification(lr_alexnet, class_num, device, 'lr', 9)
        if steps_num == 5000:
            torch.save(lr_alexnet.state_dict(),
                       '/home/tangjiawei/PycharmProjects/RadimoicDeepFeatureExtraction/model/{}_lr_alexnet_df{}.pth'.format(
                           steps_num, downsampling_factor))
            print("HR image result of LR spherenet")
            logging.info("HR image result of LR spherenet")
            lfw_verification(lr_alexnet, class_num, device, 'lr', 1)
            print("LR image result of LR spherenet")
            logging.info("LR image result of LR spherenet")
            lfw_verification(lr_alexnet, class_num, device, 'lr', 9)
        if steps_num == 20000:
            torch.save(lr_alexnet.state_dict(),
                       '/home/tangjiawei/PycharmProjects/RadimoicDeepFeatureExtraction/model/{}_lr_alexnet_df{}.pth'.format(
                           steps_num, downsampling_factor))
            print("HR image result of LR spherenet")
            logging.info("HR image result of LR spherenet")
            lfw_verification(lr_alexnet, class_num, device, 'lr', 1)
            print("LR image result of LR spherenet")
            logging.info("LR image result of LR spherenet")
            lfw_verification(lr_alexnet, class_num, device, 'lr', 9)

    #     if steps_num % 1000 == 0:
    #         # print("HR image result of HR alexnet")
    #         # lfw_verification(hr_alexnet, class_num, device, 'hr', step=steps_num)
    #         # print("LR image result of HR alexnet")
    #         # lfw_verification(hr_alexnet, class_num, device, 'lr', step=steps_num)
    #         print("HR image result of LR alexnet")
    #         lfw_verification(lr_alexnet, class_num, device, 'lr', 1)
    #         print("LR image result of LR alexnet")
    #         lfw_verification(lr_alexnet, class_num, device, 'lr', 9)
    #     # display
    #     description = '{}: {:.4f} , {}: {:.4f}'.format('loss', loss, 'lr', lr)
    #     pbar.set_description(desc=description)
    # torch.save(lr_alexnet.state_dict(), args.model_root_lr_2)




    # face = cv2.imread('/home/jiawei/Documents/polyU/FYP/Couple_mapping_MATLAB/test/HR/251.jpg')
    # face = cv2.resize(face, (224, 224))
    # face = ToTensor()(face)
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # face.to(device)
    # torch.LongTensor(1)
    # alexnet = models.alexnet(pretrained=True)
    # alexnet.features[0] = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
    # alexnet.to(device)
    # alexnet.train()
    # alexnet(face)
    # a = 1




















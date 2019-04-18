import pandas as pd
import cv2
import numpy as np
from matlab_cp2tform import get_similarity_transform_for_cv2
import torch
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from sklearn import preprocessing


def alignment(src_img, src_pts, size=None):
    ref_pts = [[30.2946, 51.6963], [65.5318, 51.5014],
               [48.0252, 71.7366], [33.5493, 92.3655],
               [62.7299, 92.2041]]
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

_lfw_root = '/home/tangjiawei/project/dataset/lfw/'
_lfw_landmarks = '/home/tangjiawei/project/dataset/LFW.csv'
_lfw_pairs = '/home/tangjiawei/project/dataset/lfw_pairs.txt'
_casia_root = '/home/tangjiawei/project/dataset/CASIA-WebFace/'
_casia_landmarks = '/home/tangjiawei/project/dataset/CASIA-maxpy-clean_remove_lfw.csv'

def load_LFWFace():
    source_face = []
    target_face = []
    flag = []
    df = pd.read_csv(_lfw_landmarks, delimiter=",", header=None)
    numpyMatrix = df.values
    landmarks = numpyMatrix[:, 1:]
    with open(_lfw_pairs) as f:
        pairs_lines = f.readlines()[1:]
    for index in range(len(pairs_lines)):
        p = pairs_lines[index].replace('\n', '').split('\t')
        if 3 == len(p):
            sameflag = np.int32(1).reshape(1)
            name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
            name2 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[2]))
        if 4 == len(p):
            sameflag = np.int32(0).reshape(1)
            name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
            name2 = p[2] + '/' + p[2] + '_' + '{:04}.jpg'.format(int(p[3]))
        img1 = alignment(cv2.imread(_lfw_root + name1), landmarks[df.loc[df[0] == name1].index.values[0]])
        img2 = alignment(cv2.imread(_lfw_root + name2), landmarks[df.loc[df[0] == name2].index.values[0]])

        source_face.append(img1)
        target_face.append(img2)
        flag.append(sameflag)

    source_face = np.asarray(source_face, 'uint8')
    target_face = np.asarray(target_face, 'uint8')
    flag = np.asarray(flag, 'uint8')

    return source_face, target_face, flag


def load_WebFace(is_aug=True):
    data = []
    df = pd.read_csv(_casia_landmarks, delimiter=",", header=None)
    numpyMatrix = df.values
    faces_path = numpyMatrix[:, 0]
    landmarks = numpyMatrix[:, 1:]
    class_id = [path.split('/')[0] for path in faces_path]
    unique_class_id = np.unique(class_id)
    le = preprocessing.LabelEncoder()
    le.fit(unique_class_id)
    targets = le.transform(class_id).reshape(-1, 1)
    for index in range(len(faces_path)):
        face = cv2.imread(_casia_root + faces_path[index])
        face = alignment(face, landmarks[index].reshape(-1, 2))
        if is_aug:
            if np.random.random() > 0.5:
                face = cv2.flip(face, 1)
        data.append(face)

    data = np.asarray(data, 'uint8')

    return data, targets


class LFWDataset(torch.utils.data.Dataset):
    def __init__(self):
        super(LFWDataset, self).__init__()
        df = pd.read_csv(_lfw_landmarks, delimiter=",", header=None)
        numpyMatrix = df.values
        self.landmarks = numpyMatrix[:, 1:]
        self.df = df
        with open(_lfw_pairs) as f:
            pairs_lines = f.readlines()[1:]
        self.pairs_lines = pairs_lines

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
        img1_flip = cv2.flip(img1, 1)
        img2_flip = cv2.flip(img2, 1)
        return face_ToTensor(img1), face_ToTensor(img2), \
               face_ToTensor(img1_flip), face_ToTensor(img2_flip), \
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
        # self.transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ])

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
        img1 = cv2.resize(img1, (int(96/self.N), int(112/self.N)))
        img2 = cv2.resize(img2, (int(96/self.N), int(112/self.N)))
        img1 = cv2.resize(img1, (96, 112))
        img2 = cv2.resize(img2, (96, 112))
        img1_flip = cv2.flip(img1, 1)
        img2_flip = cv2.flip(img2, 1)
        return face_ToTensor(img1), face_ToTensor(img2), \
               face_ToTensor(img1_flip), face_ToTensor(img2_flip), \
               torch.LongTensor(sameflag)

    def __len__(self):
        return len(self.pairs_lines)

class WebFaceDataset(torch.utils.data.Dataset):
    def __init__(self, N = 1, size=False, is_aug=True):
        super(WebFaceDataset, self).__init__()
        # Original num. subjects(num. images): 10,575(494,414)
        df = pd.read_csv(_casia_landmarks, delimiter=",", header=None)
        numpyMatrix = df.values
        self.faces_path = numpyMatrix[:, 0]
        self.landmarks = numpyMatrix[:, 1:]
        self.targets, self.num_class = self.get_targets()
        self.is_aug = is_aug
        self.size = size
        self.N = N

    def __getitem__(self, index):
        face = cv2.imread(_casia_root + self.faces_path[index])
        face = alignment(face, self.landmarks[index].reshape(-1,2))
        face_lr = cv2.resize(face, (int(96/self.N), int(112/self.N)))
        face_lr = cv2.resize(face_lr, (96, 112))
        if self.is_aug:
            if np.random.random() > 0.5:
                face = cv2.flip(face, 1)
                face_lr = cv2.flip(face_lr, 1)
        return face_ToTensor(face), face_ToTensor(face_lr), torch.LongTensor(self.targets[index])

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
            dataset = WebFaceDataset(N=N, size=False, is_aug=is_aug)
            num_class = dataset.num_class
        # if name == 'scf':
        #     dataset = SCFDataset()
        #     num_class = None
        #     shuffle = False
        #     drop_last = False
        # if name == 'LFWDataset_gz':
        #     dataset = LFWDataset_gz()
        #     num_class = None
        #     shuffle = False
        #     drop_last = False
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

# class get_loader():
#     def __init__(self,  name, batch_size, is_aug=True, shuffle=True, drop_last=True):
#         if name == 'webface':
#             dataset = WebFaceDataset(is_aug=is_aug)
#             num_class = dataset.num_class
#         elif name == 'lfw':
#             dataset = LFWDataset()
#             num_class = None
#             shuffle = False
#             drop_last = False
#         self.dataloader = DataLoader(dataset=dataset, num_workers=4, batch_size=batch_size,
#                                      pin_memory=True, shuffle=shuffle, drop_last=drop_last)
#         self.num_class = num_class
#         self.train_iter = iter(self.dataloader)
#
#     def next(self):
#         try:
#             data = next(self.train_iter)
#         except:
#             self.train_iter = iter(self.dataloader)
#             data = next(self.train_iter)
#         return data
import torch
import numpy as np
from sklearn.metrics import roc_curve, auc
from loader import get_loader
from util import save_log, KFold, find_best_threshold, eval_acc, tensor_pair_cosine_distance
import csv
import logging

logging.basicConfig(filename='FYP_Resnet_validation.log', level=logging.INFO)
def run(net, feature_dim, device, N =1, step=None):
    net.eval()

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
            message = 'DF={} LFWACC={:.4f} std={:.4f} auc={:.4f} type:{} at {}iter.'.format(N, mean_acc, std, roc_auc, cal_type, step)
            logging.info(message)
        else:
            message = 'DF={} LFWACC={:.4f} std={:.4f} auc={:.4f} type:{} at testing.'.format(N, mean_acc, std, roc_auc, cal_type)
            logging.info(message)
        print(message)
        # if step is None:
        #     np.savez('/home/tangjiawei/PycharmProjects/FYP_Face_Verification/saved/lr_9DF_2000step/cnn_roc_DF9_2000step_{}.npz'.format(N), name1=fpr, name2=tpr)

        # if N != 0:
        #     csvfile = open("/home/tangjiawei/PycharmProjects/FYP_Face_Verification/saved/lr_9DF_30000step/spherenet_lr_result.csv",
        #                    'a+')
        #     writer = csv.writer(csvfile)
        #     result = ['{:.3f}'.format(mean_acc), '{:.3f}'.format(roc_auc)]
        #     writer.writerow(result)
        #     csvfile.close()



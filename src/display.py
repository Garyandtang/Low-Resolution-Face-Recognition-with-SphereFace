from loader import load_LFWFace,load_WebFace
import numpy as np
import cv2
import pandas as pd
from matlab_cp2tform import get_similarity_transform_for_cv2
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from sklearn import preprocessing

_lfw_root = '/home/johnny/Datasets/LFW/lfw/'
_lfw_landmarks = '../data/LFW.csv'
_lfw_pairs = '../data/lfw_pairs.txt'
_casia_root = '/home/johnny/Datasets/CASIA-WebFace/CASIA-WebFace/'
_casia_landmarks = '../data/CASIA-maxpy-clean_remove_lfw.csv'


#spherenet hr pretrained accuracy and auc with different resolution
acc_hr_spherenet = [0.9890000000000001, 0.9890000000000001,0.983, 0.96, 0.9079999999999999,0.86, 0.82,0.784,0.762, 0.75,0.741,0.727, 0.6920000000000001,0.677, 0.6970000000000001,0.6970000000000001]
#auc = [0.9990000000000001, 0.9990000000000001, 0.998, 0.992, 0.9670000000000001, 0.937, 0.899, 0.868, 0.848, 0.8240000000000001, 0.816, 0.799,0.7659999999999999, 0.7390000000000001, 0.764,0.764]
# counter = list(range(1,len(acc)+1))

#spherenet lr trained 10 times accuracy and auc with different resolution
acc_lr_10_spherenet = [0.983, 0.976, 0.9640000000000001, 0.938, 0.903, 0.883, 0.847, 0.838, 0.81, 0.7879999999999999, 0.769, 0.778, 0.748, 0.735, 0.7340000000000001, 0.7340000000000001]
#auc = [0.9990000000000001, 0.998, 0.995, 0.986, 0.9670000000000001, 0.95, 0.927 0.917, 0.897, 0.871, 0.855, 0.855, 0.8220000000000001, 0.8059999999999999, 0.818, 0.818]
# counter = list(range(1,len(acc)+1))

#spherenet lr trained 100 times accuracy and auc with different resolution
acc_lr_100_spherenet = [0.9520000000000001, 0.9540000000000001, 0.943, 0.927, 0.914, 0.9059999999999999, 0.887, 0.8809999999999999, 0.8740000000000001, 0.836, 0.828, 0.807, 0.773, 0.7609999999999999, 0.754, 0.754]
#auc = [0.992, 0.991, 0.987, 0.981, 0.972, 0.966, 0.9440000000000001, 0.9009999999999999, 0.888, 0.841, 0.8390000000000001, 0.8390000000000001]
# counter = list(range(1,len(acc)+1))

#spherenet lr trained 1000 times accuracy and auc with different resolution
acc_lr_1000_spherenet = [0.927, 0.9309999999999999, 0.932, 0.927, 0.922, 0.914, 0.903, 0.9009999999999999, 0.895, 0.856, 0.8290000000000001, 0.816, 0.79, 0.759, 0.7659999999999999, 0.7659999999999999]
#auc = [0.9790000000000001, 0.981, 0.9790000000000001, 0.975, 0.97, 0.9640000000000001, 0.963, 0.9590000000000001, 0.932, 0.91, 0.9, 0.8420000000000001, 0.843, 0.843]

# lr alexnet: accuracy vs resolution
lr_acc_alexnet = [0.8968, 0.9043, 0.9042, 0.9048, 0.8988, 0.8918, 0.8582, 0.8313, 0.7877, 0.7808, 0.7460, 0.7298, 0.7205, 0.7088, 0.7027, 0.7027]

# hr alexnet: accuracy vs resolution
hr_acc_alexnet = [0.9532, 0.9473, 0.9427, 0.9242, 0.8925, 0.8520, 0.7958, 0.7813, 0.7478, 0.7270, 0.7052, 0.6942, 0.6783, 0.6762, 0.6603, 0.6603]

#sperenet lr training steps and lr image accuracy

if __name__ == '__main__':


    # df = pd.read_csv('/home/tangjiawei/PycharmProjects/FYP_Face_Verification/saved/lr_9DF_30000step/spherenet_lr_result.csv', sep=',', header=None)
    # result = df.values.transpose()
    # acc = result[0].tolist()
    # auc = result[1].tolist()
    #
    # # #acc and training step
    # # acc = acc[0:10]
    # # max = len(acc)*500
    # # counter = list(range(0, max, 500))
    # # plt.yticks(np.arange(0.7, 1.0, 0.02))
    # # plt.xticks(np.arange(0, max, 500))
    # # plt.savefig('/home/tangjiawei/PycharmProjects/FYP_Face_Verification/image/lr_trainingstep_acc_9.png')
    # # acc and resolution
    # max = len(acc)+1
    # counter = list(range(1, max))
    # plt.plot(counter, acc_hr_spherenet, 'r', label='HR_Resnet')
    # # plt.plot(counter, hr_acc_alexnet, 'g', label='HR_alexnet')
    # plt.plot(counter, acc_lr_10_spherenet, 'g', label='LR_Resnet_10')
    # plt.plot(counter, acc_lr_100_spherenet, 'b', label='LR_Resnet_100')
    # plt.plot(counter, acc_lr_1000_spherenet, 'gray', label='LR_Resnet_1000')
    # plt.plot(counter, acc, 'y', label='LR_Resnet_30000')
    # plt.yticks(np.arange(0.6, 1.05, 0.05))
    # plt.xticks(np.arange(0, max, 1))
    # plt.legend(loc='lower left')
    # plt.ylabel('Accuracy', fontsize=14)
    # plt.xlabel('Downsampling Factor(N)', fontsize=14)
    # plt.grid(color='gray', which="both", linestyle='--', linewidth=0.5)
    # plt.savefig('/home/tangjiawei/PycharmProjects/FYP_Face_Verification/image/lr_DF9_STEP10_30000_acc_resolution.png')
    # plt.show()
    # print('finished')
    #
    # #对比alex res
    alex_roc = np.load('/home/tangjiawei/PycharmProjects/FYP_Face_Verification/saved/alexnet_hr/resnet_roc_7.npz')
    alex_fpr = alex_roc['name1']
    alex_tpr = alex_roc['name2']
    alex_roc_auc = auc(alex_fpr, alex_tpr)
    lbp_roc = np.load('/home/tangjiawei/PycharmProjects/FYP_Face_Verification/saved/hr_resnet/resnet_roc_7.npz')
    lbp_fpr = lbp_roc['name1']
    lbp_tpr = lbp_roc['name2']

    alex_roc_7 = np.load('/home/tangjiawei/PycharmProjects/RadimoicDeepFeatureExtraction/saved/lr_7DF_20000step/cnn_roc_DF9_2000step_7.npz')
    alex_fpr_7 = alex_roc_7['name1']
    alex_tpr_7 = alex_roc_7['name2']
    alex_roc_auc_7 = auc(alex_fpr_7, alex_tpr_7)

    res_roc_7 = np.load(
        '/home/tangjiawei/PycharmProjects/FYP_Face_Verification/saved/lr_DF7_1000step/cnn_roc_DF7_1000step_7.npz')
    res_fpr_7 = res_roc_7['name1']
    res_tpr_7 = res_roc_7['name2']

    res_roc_auc_7 = auc(res_fpr_7, res_tpr_7)
    lbp_roc_auc = auc(lbp_fpr, lbp_tpr)
    plt.plot(lbp_fpr, lbp_tpr, 'r', label='Resnet [{0:.3f}]'.format(lbp_roc_auc))  # plotting t, a separately
    plt.plot(alex_fpr, alex_tpr, 'g', label='Alexnet [{0:.3f}]'.format(alex_roc_auc))  # plotting t, a separately
    plt.plot(res_fpr_7, res_tpr_7, 'b', label='LR-Resnet [{0:.3f}]'.format(res_roc_auc_7))
    plt.plot(alex_fpr_7, alex_tpr_7, 'y', label='LR-Alexnet 1[{0:.3f}]'.format(alex_roc_auc_7))

    plt.legend(loc='lower right')
    # plt.yscale('log')
    # plt.xscale('log')
    # plt.ylim([0.5, 1])
    # plt.xlim([1e-5, 1])
    plt.xticks(np.arange(0, 1.05, 0.1))
    plt.yticks(np.arange(0, 1.05, 0.1))
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.grid(color='gray', which="both", linestyle='--', linewidth=0.5)
    plt.savefig('/home/tangjiawei/PycharmProjects/FYP_Face_Verification/image/roc_hr_lr_alex_res.png')
    # plt.show()
    print('finished')

    # #画多个 resnet resolution
    #
    # res_roc_1 = np.load('/home/tangjiawei/PycharmProjects/RadimoicDeepFeatureExtraction/saved/lr_5DF_20000step/cnn_roc_DF5_20000step_1.npz')
    # res_fpr_1 = res_roc_1['name1']
    # res_tpr_1 = res_roc_1['name2']
    # res_roc_auc_1 = auc(res_fpr_1, res_tpr_1)
    #
    # res_roc_4 = np.load('/home/tangjiawei/PycharmProjects/RadimoicDeepFeatureExtraction/saved/lr_5DF_20000step/cnn_roc_DF5_20000step_4.npz')
    # res_fpr_4 = res_roc_4['name1']
    # res_tpr_4 = res_roc_4['name2']
    # res_roc_auc_4 = auc(res_fpr_4, res_tpr_4)
    #
    # res_roc_5 = np.load('/home/tangjiawei/PycharmProjects/RadimoicDeepFeatureExtraction/saved/lr_5DF_20000step/cnn_roc_DF5_20000step_5.npz')
    # res_fpr_5 = res_roc_5['name1']
    # res_tpr_5 = res_roc_5['name2']
    # res_roc_auc_5 = auc(res_fpr_5, res_tpr_5)
    #
    # res_roc_7 = np.load('/home/tangjiawei/PycharmProjects/RadimoicDeepFeatureExtraction/saved/lr_5DF_20000step/cnn_roc_DF5_20000step_7.npz')
    # res_fpr_7 = res_roc_7['name1']
    # res_tpr_7 = res_roc_7['name2']
    # res_roc_auc_7 = auc(res_fpr_7, res_tpr_7)
    #
    # res_roc_10 = np.load('/home/tangjiawei/PycharmProjects/RadimoicDeepFeatureExtraction/saved/lr_5DF_20000step/cnn_roc_DF5_20000step_10.npz')
    # res_fpr_10 = res_roc_10['name1']
    # res_tpr_10 = res_roc_10['name2']
    # res_roc_auc_10 = auc(res_fpr_10, res_tpr_10)
    #
    # res_roc_13 = np.load('/home/tangjiawei/PycharmProjects/RadimoicDeepFeatureExtraction/saved/lr_5DF_20000step/cnn_roc_DF5_20000step_13.npz')
    # res_fpr_13 = res_roc_13['name1']
    # res_tpr_13 = res_roc_13['name2']
    # res_roc_auc_13 = auc(res_fpr_13, res_tpr_13)
    #
    # plt.plot(res_fpr_1, res_tpr_1, 'r', label='LR-Alexnet 96x112 [{0:.3f}]'.format(res_roc_auc_1))  # plotting t, a separately
    # plt.plot(res_fpr_4, res_tpr_4, 'b', label='LR-Alexnet 24x28 [{0:.3f}]'.format(res_roc_auc_4))
    # plt.plot(res_fpr_5, res_tpr_5, 'navy', label='LR-Alexnet 19x22 [{0:.3f}]'.format(res_roc_auc_5))
    # plt.plot(res_fpr_7, res_tpr_7, 'y', label='LR-Alexnet 14x16 [{0:.3f}]'.format(res_roc_auc_7))
    # plt.plot(res_fpr_10, res_tpr_10, 'm', label='LR-Alexnet 10x11 [{0:.3f}]'.format(res_roc_auc_10))
    # plt.plot(res_fpr_13, res_tpr_13, 'orange', label='LR-Alexnet 7x8 [{0:.3f}]'.format(res_roc_auc_13))
    # plt.legend(loc='lower right')
    # # plt.yscale('log')
    # # plt.xscale('log')
    # # plt.ylim([0.5, 1])
    # # plt.xlim([1e-5, 1])
    # plt.xticks(np.arange(0, 1.05, 0.1))
    # plt.yticks(np.arange(0, 1.05, 0.1))
    # plt.ylabel('True Positive Rate', fontsize=16)
    # plt.xlabel('False Positive Rate', fontsize=16)
    # plt.grid(color='gray', which="both", linestyle='--', linewidth=0.5)
    # plt.savefig('/home/tangjiawei/PycharmProjects/FYP_Face_Verification/image/roc_lr_alex_DF1_DF13.png')
    # # plt.show()
    # print('finished')

    # # 画多个 res resolution
    #
    # res_roc_1 = np.load(
    #     '/home/tangjiawei/PycharmProjects/FYP_Face_Verification/saved/lr_DF7_1000step/cnn_roc_DF7_1000step_1.npz')
    # res_fpr_1 = res_roc_1['name1']
    # res_tpr_1 = res_roc_1['name2']
    # res_roc_auc_1 = auc(res_fpr_1, res_tpr_1)
    #
    # res_roc_4 = np.load(
    #     '/home/tangjiawei/PycharmProjects/FYP_Face_Verification/saved/lr_DF7_1000step/cnn_roc_DF7_1000step_4.npz')
    # res_fpr_4 = res_roc_4['name1']
    # res_tpr_4 = res_roc_4['name2']
    # res_roc_auc_4 = auc(res_fpr_4, res_tpr_4)
    #
    # res_roc_5 = np.load(
    #     '/home/tangjiawei/PycharmProjects/FYP_Face_Verification/saved/lr_DF7_1000step/cnn_roc_DF7_1000step_5.npz')
    # res_fpr_5 = res_roc_5['name1']
    # res_tpr_5 = res_roc_5['name2']
    # res_roc_auc_5 = auc(res_fpr_5, res_tpr_5)
    #
    # res_roc_7 = np.load(
    #     '/home/tangjiawei/PycharmProjects/FYP_Face_Verification/saved/lr_DF7_1000step/cnn_roc_DF7_1000step_7.npz')
    # res_fpr_7 = res_roc_7['name1']
    # res_tpr_7 = res_roc_7['name2']
    # res_roc_auc_7 = auc(res_fpr_7, res_tpr_7)
    #
    # res_roc_10 = np.load(
    #     '/home/tangjiawei/PycharmProjects/FYP_Face_Verification/saved/lr_DF7_1000step/cnn_roc_DF7_1000step_10.npz')
    # res_fpr_10 = res_roc_10['name1']
    # res_tpr_10 = res_roc_10['name2']
    # res_roc_auc_10 = auc(res_fpr_10, res_tpr_10)
    #
    # res_roc_13 = np.load(
    #     '/home/tangjiawei/PycharmProjects/FYP_Face_Verification/saved/lr_DF7_1000step/cnn_roc_DF7_1000step_13.npz')
    # res_fpr_13 = res_roc_13['name1']
    # res_tpr_13 = res_roc_13['name2']
    # res_roc_auc_13 = auc(res_fpr_13, res_tpr_13)
    #
    # plt.plot(res_fpr_1, res_tpr_1, 'r',
    #          label='LR-Resxnet 96x112 [{0:.3f}]'.format(res_roc_auc_1))  # plotting t, a separately
    # plt.plot(res_fpr_4, res_tpr_4, 'b', label='LR-Resnet 24x28 [{0:.3f}]'.format(res_roc_auc_4))
    # plt.plot(res_fpr_5, res_tpr_5, 'navy', label='LR-Resnet 19x22 [{0:.3f}]'.format(res_roc_auc_5))
    # plt.plot(res_fpr_7, res_tpr_7, 'y', label='LR-Resnet 14x16 [{0:.3f}]'.format(res_roc_auc_7))
    # plt.plot(res_fpr_10, res_tpr_10, 'm', label='LR-Resnet 10x11 [{0:.3f}]'.format(res_roc_auc_10))
    # plt.plot(res_fpr_13, res_tpr_13, 'orange', label='LR-Resnet 7x8 [{0:.3f}]'.format(res_roc_auc_13))
    # plt.legend(loc='lower right')
    # # plt.yscale('log')
    # # plt.xscale('log')
    # # plt.ylim([0.5, 1])
    # # plt.xlim([1e-5, 1])
    # plt.xticks(np.arange(0, 1.05, 0.1))
    # plt.yticks(np.arange(0, 1.05, 0.1))
    # plt.ylabel('True Positive Rate', fontsize=16)
    # plt.xlabel('False Positive Rate', fontsize=16)
    # plt.grid(color='gray', which="both", linestyle='--', linewidth=0.5)
    # plt.savefig('/home/tangjiawei/PycharmProjects/FYP_Face_Verification/image/roc_lr_res_DF1_DF13.png')
    # # plt.show()
    # print('finished')
    # pca_roc = np.load('/home/aaron/projects/FYP_Face_Verification/data/pca_roc.npz')
    # pca_fpr = pca_roc['name1']
    # pca_tpr = pca_roc['name2']
    # pca_roc_auc = auc(pca_fpr, pca_tpr)
    #
    # lbp_roc = np.load('/home/aaron/projects/FYP_Face_Verification/data/lbp_roc.npz')
    # lbp_fpr = lbp_roc['name1']
    # lbp_tpr = lbp_roc['name2']
    # lbp_roc_auc = auc(lbp_fpr, lbp_tpr)
    #
    # cnn_roc_softmax = np.load('/home/aaron/projects/FYP_Face_Verification/data/cnn_roc_softmax.npz')
    # cnn_softmax_fpr = cnn_roc_softmax['name1']
    # cnn_softmax_tpr = cnn_roc_softmax['name2']
    # softmax_roc_auc = auc(cnn_softmax_fpr, cnn_softmax_tpr)
    #
    # cnn_roc_asoftmax = np.load('/home/aaron/projects/FYP_Face_Verification/data/cnn_roc_asoftmax.npz')
    # cnn_asoftmax_fpr = cnn_roc_asoftmax['name1']
    # cnn_asoftmax_tpr = cnn_roc_asoftmax['name2']
    # asoftmax_roc_auc = auc(cnn_asoftmax_fpr, cnn_asoftmax_tpr)
    #
    # cnn_roc_amsoftmax = np.load('/home/aaron/projects/FYP_Face_Verification/data/cnn_roc_amsoftmax.npz')
    # cnn_amsoftmax_fpr = cnn_roc_amsoftmax['name1']
    # cnn_amsoftmax_tpr = cnn_roc_amsoftmax['name2']
    # amsoftmax_roc_auc = auc(cnn_amsoftmax_fpr, cnn_amsoftmax_tpr)
    #
    # plt.plot(pca_fpr, pca_tpr, 'r', label='EigenFaces [{0:.3f}]'.format(pca_roc_auc))  # plotting t, a separately
    # plt.plot(lbp_fpr, lbp_tpr, 'b', label='Unweighted Uniform LBP [{0:.3f}]'.format(lbp_roc_auc))  # plotting t, b separately
    # plt.plot(cnn_softmax_fpr, cnn_softmax_tpr, 'g', label='Softmax [{0:.3f}]'.format(softmax_roc_auc))  # plotting t, b separately
    # plt.plot(cnn_amsoftmax_fpr, cnn_amsoftmax_tpr, 'y',
    #          label='A-Softmax [{0:.3f}]'.format(asoftmax_roc_auc))  # plotting t, b separately
    # plt.plot(cnn_asoftmax_fpr, cnn_asoftmax_tpr, 'm',
    #          label='AM-Softmax [{0:.3f}]'.format(amsoftmax_roc_auc))  # plotting t, b separately
    # # plt.plot(cnn_asoftmax_fpr, cnn_asoftmax_tpr, 'y', label='A-Softmax [{}]'.format(asoftmax_roc_auc))  # plotting t, b separately
    # # plt.plot(cnn_amsoftmax_fpr, cnn_amsoftmax_tpr, 'm', label='AM-Softmax [{}]'.format(amsoftmax_roc_auc))  # plotting t, b separately
    # plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    # plt.legend(loc='lower right')
    # # plt.yscale('log')
    # # plt.xscale('log')
    # plt.ylim([0, 1.05])
    # plt.xlim([0, 1.05])
    # plt.xticks(np.arange(0, 1.05, 0.1))
    # plt.yticks(np.arange(0, 1.05, 0.1))
    # plt.ylabel('True Positive Rate',fontsize=16)
    # plt.xlabel('False Positive Rate',fontsize=16)
    # plt.grid(color='gray', linestyle='--', linewidth=0.5)
    # plt.savefig('/home/aaron/projects/FYP_Face_Verification/src/roc_curve_all.png')
    # plt.show()
    #
    #
    #
    # plt.plot(cnn_softmax_fpr, cnn_softmax_tpr, 'g', label='Softmax [{0:.3f}]'.format(softmax_roc_auc))  # plotting t, b separately
    # plt.plot(cnn_amsoftmax_fpr, cnn_amsoftmax_tpr, 'y',
    #          label='A-Softmax [{0:.3f}]'.format(asoftmax_roc_auc))  # plotting t, b separately
    # plt.plot(cnn_asoftmax_fpr, cnn_asoftmax_tpr, 'm',
    #          label='AM-Softmax [{0:.3f}]'.format(amsoftmax_roc_auc))  # plotting t, b separately
    # # plt.plot(cnn_asoftmax_fpr, cnn_asoftmax_tpr, 'y', label='A-Softmax [{}]'.format(asoftmax_roc_auc))  # plotting t, b separately
    # # plt.plot(cnn_amsoftmax_fpr, cnn_amsoftmax_tpr, 'm', label='AM-Softmax [{}]'.format(amsoftmax_roc_auc))  # plotting t, b separately
    # # plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    # plt.legend(loc='lower right')
    # # plt.yscale('log')
    # plt.xscale('log')
    # plt.ylim([0.8, 1])
    # plt.xlim([1e-5, 1])
    # # plt.xticks(np.arange(0, 1.05, 0.1))
    # # plt.yticks(np.arange(0, 1.05, 0.1))
    # plt.ylabel('True Positive Rate',fontsize=16)
    # plt.xlabel('False Positive Rate',fontsize=16)
    # plt.grid(color='gray', which="both", linestyle='--', linewidth=0.5)
    # plt.savefig('/home/aaron/projects/FYP_Face_Verification/src/roc_curve_deep.png')
    # plt.show()
    print('finished')




from numpy import *
import numpy as np
from sklearn.cluster import KMeans
import os

def cla_mnsit_cul(picPath, actPath, savePath, className):
    temp = savePath + os.sep + 'picture'
    if not os.path.exists(temp):
        os.mkdir(temp)
    temp = savePath + os.sep + 'act'
    if not os.path.exists(temp):
        os.mkdir(temp)
    all_act = np.load(actPath + os.sep + str(className) + '_act.npy')
    all_label = np.load(actPath + os.sep + str(className) + '_label.npy')
    all_index = np.empty((len(all_act)), dtype='uint8')
    clt = KMeans(n_clusters=20, n_init=200)
    clt.fit(all_act)
    getClaPath = picPath + os.sep + str(className)
    saveClaPath = savePath + os.sep + 'picture' + os.sep + str(className)
    path1 = savePath + os.sep + 'act' + os.sep + str(className) + '_act_after_clu.npy'
    path2 = savePath + os.sep + 'act' + os.sep + str(className) + '_label_after_clu.npy'
    path3 = savePath + os.sep + 'act' + os.sep + str(className) + '_index_after_clu.npy'
    if not os.path.exists(saveClaPath):
        os.mkdir(saveClaPath)
    for i in range(len(clt.labels_)):
        getPicPath = getClaPath + os.sep + all_label[i]
        saveFeaPath = saveClaPath + os.sep + str(clt.labels_[i])
        all_index[i] = clt.labels_[i]
        if not os.path.exists(saveFeaPath):
            os.mkdir(saveFeaPath)
        savePicPath = saveFeaPath + os.sep + all_label[i]
        img1 = open(getPicPath, "rb")
        img2 = open(savePicPath, "wb")
        img2.write(img1.read())
        img1.close()
        img2.close()
    np.save(path1, all_act)
    np.save(path2, all_label)
    np.save(path3, all_index)


def cla_CIFAR_cul(picPath, actPath, savePath, className):
    if not os.path.exists(savePath):
        os.mkdir(savePath)
    temp = savePath + os.sep + 'picture'
    if not os.path.exists(temp):
        os.mkdir(temp)
    temp = savePath + os.sep + 'act'
    if not os.path.exists(temp):
        os.mkdir(temp)
    all_act = np.load(actPath + os.sep + str(className) + '_act.npy')
    all_label = np.load(actPath + os.sep + str(className) + '_label.npy')
    all_index = np.empty((len(all_act)), dtype='uint8')
    clt = KMeans(n_clusters=20, n_init=200)
    clt.fit(all_act)
    getClaPath = picPath + os.sep + str(className)
    saveClaPath = savePath + os.sep + 'picture' + os.sep + str(className)
    path1 = savePath + os.sep + 'act' + os.sep + str(className) + '_act_after_clu.npy'
    path2 = savePath + os.sep + 'act' + os.sep + str(className) + '_label_after_clu.npy'
    path3 = savePath + os.sep + 'act' + os.sep + str(className) + '_index_after_clu.npy'
    if not os.path.exists(saveClaPath):
        os.mkdir(saveClaPath)
    for i in range(len(clt.labels_)):
        getPicPath = getClaPath + os.sep + all_label[i]
        saveFeaPath = saveClaPath + os.sep + str(clt.labels_[i])
        all_index[i] = clt.labels_[i]
        if not os.path.exists(saveFeaPath):
            os.mkdir(saveFeaPath)
        savePicPath = saveFeaPath + os.sep + all_label[i]
        img1 = open(getPicPath, "rb")
        img2 = open(savePicPath, "wb")
        img2.write(img1.read())
        img1.close()
        img2.close()
    np.save(path1, all_act)
    np.save(path2, all_label)
    np.save(path3, all_index)

if __name__ == '__main__':
    pass









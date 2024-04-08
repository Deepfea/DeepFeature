import os
import numpy as np
from numpy import mat
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import math
import scipy.spatial.distance as dis
from tqdm import tqdm


def unit_cav(cav):
    u_cav = np.empty(len(cav), dtype='float32')
    y = math.sqrt(sum(cav * cav))
    for i in range(len(cav)):
        u_cav[i] = cav[i]/y
    return u_cav
def load_robust_fea_act(robust_fea_path, candidate_fea_path, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for class_num in range(10):
        print('class_num: ' + str(class_num))
        robust_fea_name = np.load(os.path.join(robust_fea_path, str(class_num) + '_robust_fea.npy'))
        candidate_fea_name = np.load(os.path.join(candidate_fea_path, str(class_num) + '_fea_name.npy'))
        candidate_fea_act = np.load(os.path.join(candidate_fea_path, str(class_num) + '_act.npy'))
        fea_act = []
        print(robust_fea_name)
        for robust_fea_num in range(len(robust_fea_name)):
            act = []
            for act_num in range(len(candidate_fea_act)):
                if int(candidate_fea_name[act_num]) == robust_fea_name[robust_fea_num]:
                    u_act = candidate_fea_act[act_num]
                    act.append(u_act)
            act = np.array(act)
            fea_act.append(act)
            print(np.array(act).shape)
        fea_act = np.array(fea_act)
        np.save(os.path.join(save_path, str(class_num) + '_fea_act.npy'), fea_act)
        np.save(os.path.join(save_path, str(class_num) + '_fea_name.npy'), robust_fea_name)

def find_middle(act, vector_num, rate):
    temp = []
    for act_num in range(len(act)):
        temp.append(act[act_num][vector_num])
    temp = sorted(temp)
    l = len(temp)
    middle_value = temp[int(l / rate)]
    return middle_value

def find_average(act, vector_num):
    average_value = np.average(act[:, vector_num])
    return average_value

def cal_vector(save_path, period, rate):
    period_save_path = os.path.join(save_path, period)
    if not os.path.exists(period_save_path):
        os.makedirs(period_save_path)
    all_vector = []
    for class_num in range(10):
        print('class_num:' + str(class_num))
        class_vector = []
        fea_act = np.load(os.path.join(save_path, str(class_num) + '_fea_act.npy'), allow_pickle=True)
        for fea_num in range(len(fea_act)):
            print('fea_num: ' + str(fea_num))
            act = fea_act[fea_num]
            act = np.array(act)
            activate_vector = np.zeros((len(act[0])), dtype='float32')
            for vector_num in range(len(activate_vector)):
                activate_vector[vector_num] = find_middle(act, vector_num, rate)
            fea_vector = []
            for act_num in tqdm(range(len(act))):
                temp_vector = []
                for temp_i in range(len(activate_vector)):
                    if act[act_num][temp_i] >= activate_vector[temp_i]:
                        temp_vector.append(1)
                    else:
                        temp_vector.append(0)
                fea_vector.append(temp_vector)
            class_vector.append(fea_vector)
        all_vector.append(class_vector)
    all_vector = np.array(all_vector)
    if period == 'in_dis':
        np.save(os.path.join(save_path, period, 'all_vector.npy'), all_vector)
    elif period == 'ou_dis':
        np.save(os.path.join(save_path, period, 'all_vector.npy'), all_vector)

def in_jac_dis(vector):
    dis_list = []
    for vector_i in tqdm(range(len(vector))):
        for vector_j in range(len(vector)):
            if vector_j <= vector_i:
                continue
            pair = []
            pair.append(vector[vector_i])
            pair.append(vector[vector_j])
            matV = mat(pair)
            x = dis.pdist(matV, metric='jaccard')[0]
            dis_list.append(x)
    return dis_list

def in_dis(save_path):
    in_save_path = os.path.join(save_path, 'in_dis')
    all_vector = np.load(os.path.join(in_save_path, 'all_vector.npy'), allow_pickle=True)
    for class_num in range(10):
        if class_num < 5:
            continue
        print('class_num:' + str(class_num))
        class_vector = all_vector[class_num]
        class_in_dis = []
        for fea_num in range(len(class_vector)):
            print('fea_num:' + str(fea_num))
            fea_vector = class_vector[fea_num]
            fea_in_dis = in_jac_dis(fea_vector)
            class_in_dis.append(fea_in_dis)
        class_in_dis = np.array(class_in_dis)
        np.save(os.path.join(in_save_path, str(class_num) + '_in_dis.npy'), class_in_dis)

def ou_jac_dis(fea_vector, class_vector, num):
    dis_list = []
    for fea_num in range(len(class_vector)):
        if fea_num == num:
            continue
        print('compare feature number: ' + str(fea_num))
        temp_vector = class_vector[fea_num]
        for vector_i in tqdm(range(len(temp_vector))):
            for vector_j in range(len(fea_vector)):
                pair = []
                pair.append(temp_vector[vector_i])
                pair.append(fea_vector[vector_j])
                matV = mat(pair)
                x = dis.pdist(matV, metric='jaccard')[0]
                dis_list.append(x)
    return dis_list

def ou_dis(save_path):
    ou_save_path = os.path.join(save_path, 'ou_dis')
    all_vector = np.load(os.path.join(ou_save_path, 'all_vector.npy'), allow_pickle=True)
    for class_num in range(10):
        print('class_num:' + str(class_num))
        class_vector = all_vector[class_num]
        class_ou_dis = []
        for fea_num in range(len(class_vector)):
            print('fea_num:' + str(fea_num))
            fea_vector = class_vector[fea_num]
            fea_ou_dis = ou_jac_dis(fea_vector, class_vector, fea_num)
            class_ou_dis.append(fea_ou_dis)
        class_ou_dis = np.array(class_ou_dis)
        np.save(os.path.join(ou_save_path, str(class_num) + '_ou_dis.npy'), class_ou_dis)


if __name__ == '__main__':
    pass

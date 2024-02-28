import os
import numpy as np
from sklearn import linear_model
import math

def unit_cav(cav):
    u_cav = np.empty(len(cav), dtype='float32')
    y = math.sqrt(sum(cav * cav))
    for i in range(len(cav)):
        u_cav[i] = cav[i]/y
    return u_cav

def cal_and_save(x, z, path, class1):
    if not os.path.exists(path):
        os.mkdir(path)
    negLabel = -1
    cav_list = []
    cav_label = []
    label_name = list(set(z))
    fea_num = len(label_name)
    for i in range(fea_num):
        temp_x = np.copy(x)
        temp_z = np.copy(z)
        num = 0
        for j in range(0, len(x)):
            if label_name[i] != z[j]:
                temp_z[j] = negLabel
            else:
                num += 1
        if num < 300:
            print(str(i), ' is ', str(num), ',continue!')
            continue
        # clf = linear_model.SGDClassifier(alpha=0.0001, tol=1e-3, max_iter=5000) # k=3 is OK
        clf = linear_model.SGDClassifier(alpha=0.01, tol=1e-3, max_iter=1000)
        print(temp_x.shape)
        print(temp_z.shape)
        clf.fit(temp_x, temp_z)
        print(list(set(temp_z)))
        cav = [c for c in clf.coef_]
        cav = np.array(cav[0])
        # print(cav)
        cav = unit_cav(cav)
        # print(cav)
        cav_list.append(cav)
        cav_label.append(i)
    cav_arr = np.array(cav_list)
    label_arr = np.array(cav_label)
    if not os.path.exists(path):
        os.mkdir(path)
    cav_path = os.path.join(path, str(class1) + '_cav.npy')
    label_path = os.path.join(path, str(class1) + '_label.npy')
    label_name_path = os.path.join(path, str(class1) + '_label_name.npy')
    print(cav_arr.shape)
    print(label_arr)
    print(label_name)
    np.save(cav_path, cav_arr)
    np.save(label_path, label_arr)
    np.save(label_name_path, np.array(label_name))

if __name__ == '__main__':
    class_list = ['cat', 'dog']
    dataset = 'dataset_2'
    model_name = 'vgg_19'
    act_path = 'E:/dataset/dataset_2/fea_select/candidate'
    cav_path = 'E:/dataset/dataset_2/fea_select/cav'
    for i in range(2):
        path_1 = act_path + os.sep + str(class_list[i]) + '_act.npy'
        path_2 = act_path + os.sep + str(class_list[i]) + '_fea_name.npy'
        x = np.load(path_1)
        z = np.load(path_2)
        # print(x)
        # print(z)
        cal_and_save(x, z, cav_path, i)






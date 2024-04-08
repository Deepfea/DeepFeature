import os
import numpy as np
from sklearn import linear_model
import math

def unit_cav(cav):
    u_cav = np.empty(len(cav), dtype='float32')
    y = math.sqrt(sum(cav * cav))
    for i in range(len(cav)):
        u_cav[i] = cav[i]/y
    print(u_cav)
    return u_cav

def cal_and_save_cav(act_path, cav_path):
    if not os.path.exists(cav_path):
        os.mkdir(cav_path)
    max_iter_list = [5000]
    alpha_list = [0.0001]
    for max_iter1 in max_iter_list:
        print(max_iter1)
        for alpha1 in alpha_list:
            print(alpha1)
            save_cav_path = os.path.join(cav_path, str(max_iter1) + '_' + str(alpha1))
            if not os.path.exists(save_cav_path):
                os.mkdir(save_cav_path)
            for class_num in range(10):
                x = np.load(os.path.join(act_path, str(class_num) + '_act.npy'))
                y = np.load(os.path.join(act_path, str(class_num) + '_fea_name.npy'))
                y_list = []
                for y_num in range(len(y)):
                    y_list.append(int(y[y_num]))
                y = np.array(y_list)
                neg_label = -1
                cav_list = []
                cav_label = []
                fea_list = np.unique(y)
                for i in range(len(fea_list)):
                    temp_x = np.copy(x)
                    temp_z = np.copy(y)
                    num = 0
                    for j in range(len(x)):
                        if fea_list[i] != y[j]:
                            temp_z[j] = neg_label
                        else:
                            num += 1
                    if num < 300:
                        print(str(i), ' is ', str(num), ',continue!')
                        continue
                    clf = linear_model.SGDClassifier(alpha=alpha1, tol=1e-3, max_iter=max_iter1)
                    clf.fit(temp_x, temp_z)
                    cav = [c for c in clf.coef_]
                    cav = np.array(cav[0])
                    cav = unit_cav(cav)
                    cav_list.append(cav)
                    cav_label.append(fea_list[i])
                cav_arr = np.array(cav_list)
                label_arr = np.array(cav_label)
                cav_save_path = os.path.join(save_cav_path, str(class_num) + '_cav.npy')
                label_path = os.path.join(save_cav_path, str(class_num) + '_fea_name.npy')
                np.save(cav_save_path, cav_arr)
                np.save(label_path, label_arr)


if __name__ == '__main__':
    pass






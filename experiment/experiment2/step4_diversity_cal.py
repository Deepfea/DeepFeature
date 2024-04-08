import math
import os
import random
import numpy as np
from keras.models import load_model
from keras import Model
import tensorflow as tf
import itertools
from tqdm import tqdm
import re
tf.compat.v1.disable_eager_execution()

def unit_act(act):
    u_cav = np.empty(len(act), dtype='float32')
    y = math.sqrt(sum(act * act))
    for i in range(len(act)):
        u_cav[i] = act[i]/y
    return u_cav

def cal_dis(act_1, act_2):
    temp_dis = 0.0
    for index in range(len(act_1)):
        temp_dis = temp_dis + (act_1[index] - act_2[index]) * (act_1[index] - act_2[index])
    temp_dis = math.sqrt(temp_dis)
    return temp_dis


def cal_entropy(w):
    # print(w)
    if 0 in w:
        return 0
    if len(w) == 1:
        return 0
    w = np.array(w)
    w_sum = np.sum(w)
    h = 0.0
    for i in range(len(w)):
        temp = w[i] / w_sum
        h = h + (-1) * temp * math.log(temp)
    h = w_sum * h
    return h

def ini(dis, pic_name, class_num):
    char = '_'
    find = []
    not_find = []
    for temp_num in range(len(dis)):
        truth_class_name = re.split(re.escape(char), pic_name[temp_num])[0]
        if class_num != int(truth_class_name):
            not_find.append(temp_num)
    r_index = -1
    c_index = -1
    max_value = 0
    for r in range(len(dis)):
        for c in range(len(dis)):
            if c <= r:
                continue
            if (r not in not_find) or (c not in not_find):
                continue
            if dis[r][c] > max_value:
                max_value = dis[r][c]
                r_index = r
                c_index = c
    if r_index != -1:
        find.append(r_index)
        temp_index = not_find.index(r_index)
        del not_find[temp_index]
    if c_index != -1:
        find.append(c_index)
        temp_index = not_find.index(c_index)
        del not_find[temp_index]
    return find, not_find



def not_find_list(find_arr, dis_arr):
    not_find = []
    for dis_arr_num in range(len(dis_arr)):
        if dis_arr_num not in find_arr:
            not_find.append(dis_arr_num)
    return not_find


def cal_min_dis(find_arr, not_find_arr, dis_arr):
    temp_x = -1
    temp_x_index = -1
    temp_y = -1
    temp_y_index = -1
    min_dis = 100
    for temp_num1 in range(len(find_arr)):
        for temp_num2 in range(len(not_find_arr)):
            if dis_arr[find_arr[temp_num1]][not_find_arr[temp_num2]] < min_dis:
                temp_x = find_arr[temp_num1]
                temp_x_index = temp_num1
                temp_y = not_find_arr[temp_num2]
                temp_y_index = temp_num2
                min_dis = dis_arr[find_arr[temp_num1]][not_find_arr[temp_num2]]
    return temp_x, temp_x_index, temp_y, temp_y_index, min_dis

def cal_tree(comb, dis_arr):
    comb = list(comb)
    find = []
    find.append(comb[0])
    not_find = comb.copy()
    del not_find[0]
    find_pair = []
    w = []
    while len(not_find) != 0:
        temp_x, temp_x_index, temp_y, temp_y_index, min_dis = cal_min_dis(find, not_find, dis_arr)
        find.append(temp_y)
        temp_list = []
        temp_list.append(temp_x)
        temp_list.append(temp_y)
        find_pair.append(temp_list)
        w.append(min_dis)
        del not_find[temp_y_index]
    return find_pair, w

def cal_tree1(comb, add, dis_arr):
    temp_dis = 10000
    for comb_num in range(len(comb)):
        if dis_arr[comb[comb_num]][add] < temp_dis:
            temp_dis = dis_arr[comb[comb_num]][add]
    return temp_dis

def permutations(num):
    comb = []
    for i in range(num):
        comb.append(i)
    return comb

def cal_div(adv_list, load_path, save_path, rate, acc):
    for adv_name in adv_list:
        print(adv_name)
        rate_path = os.path.join(save_path, adv_name, str(acc))
        if not os.path.exists(rate_path):
            os.makedirs(rate_path)
        for class_num in range(10):
            print(rate)
            print(class_num)
            class_fea_name_path = os.path.join(load_path, adv_name, str(class_num) + '_fea_name.npy')
            class_fea_name = np.load(class_fea_name_path)

            class_fea_acc_path = os.path.join(load_path, adv_name, str(class_num) + '_fea_acc.npy')
            class_fea_acc = np.load(class_fea_acc_path)

            class_fea_pic_name_path = os.path.join(load_path, adv_name, str(class_num) + '_fea_pic_name.npy')
            class_fea_pic_name = np.load(class_fea_pic_name_path, allow_pickle=True)

            class_act_path = os.path.join(save_path, adv_name, str(class_num) + '_fea_pic_act.npy')
            class_act = np.load(class_act_path, allow_pickle=True)

            class_dis_path = os.path.join(save_path, adv_name, str(class_num) + '_fea_dis.npy')
            class_dis = np.load(class_dis_path, allow_pickle=True)

            fea_top_diversity_pic_name = []
            fea_top_diversity_comb = []
            fea_top_diversity_comb_pair = []
            fea_top_diversity_w = []
            fea_top_diversity_value = []
            fea_name = []
            for fea_num in range(len(class_act)):
                print(class_fea_name[fea_num])
                if class_fea_acc[fea_num] > acc:
                    print("强区域,跳过")
                    continue
                find, not_find = ini(class_dis[fea_num], class_fea_pic_name[fea_num], class_num)
                if class_fea_name[fea_num] == -1:
                    print("无区域")
                    find = find + not_find
                    if len(find) == 1:
                        find_pair = []
                        w = []
                        h = 0
                    else:
                        find_pair, w = cal_tree(find, class_dis[fea_num])
                        print(w)
                        h = cal_entropy(w)
                else:
                    print("弱区域")
                    print(math.ceil(len(class_dis[fea_num]) * rate))
                    if math.ceil(len(class_dis[fea_num]) * rate) == 1:
                        find_pair = []
                        w = []
                        h = 0
                    elif math.ceil(len(class_dis[fea_num]) * rate) == 2:
                        find_pair, w = cal_tree(find, class_dis[fea_num])
                        print(w)
                        h = cal_entropy(w)
                    else:
                        find_pair, w = cal_tree(find, class_dis[fea_num])
                        while len(find) != math.ceil(len(class_dis[fea_num]) * rate):
                            print(len(find))
                            temp_h = 0
                            temp_num = -1
                            if len(find) < 80:
                                for not_find_num in range(len(not_find)):
                                    temp_find = find.copy()
                                    temp_find.append(not_find[not_find_num])
                                    find_pair, w = cal_tree(temp_find, class_dis[fea_num])
                                    h = cal_entropy(w)
                                    if h > temp_h:
                                        temp_h = h
                                        temp_num = not_find_num
                            else:
                                while_num = 80
                                while while_num > 0:
                                    random_number = random.randint(0, len(not_find)-1)
                                    temp_find = find.copy()
                                    temp_find.append(not_find[random_number])
                                    find_pair, w = cal_tree(temp_find, class_dis[fea_num])
                                    h = cal_entropy(w)
                                    if h > temp_h:
                                        temp_h = h
                                        temp_num = random_number
                                    while_num = while_num - 1
                            find.append(not_find[temp_num])
                            del not_find[temp_num]
                        find_pair, w = cal_tree(find, class_dis[fea_num])
                        h = cal_entropy(w)

                top_diversity_comb = find
                top_diversity_comb_pair = find_pair
                top_diversity_w = w
                top_diversity_value = h
                top_diversity_pic_name = []
                for temp_i in range(len(top_diversity_comb)):
                    top_diversity_pic_name.append(class_fea_pic_name[fea_num][top_diversity_comb[temp_i]])

                fea_top_diversity_comb.append(top_diversity_comb)
                fea_top_diversity_comb_pair.append(top_diversity_comb_pair)
                fea_top_diversity_w.append(top_diversity_w)
                fea_top_diversity_value.append(top_diversity_value)
                fea_name.append(class_fea_name[fea_num])
                fea_top_diversity_pic_name.append(top_diversity_pic_name)

            fea_top_diversity_comb = np.array(fea_top_diversity_comb)
            print(fea_top_diversity_comb)
            np.save(os.path.join(rate_path, str(class_num) + '_fea_top_diversity_comb.npy'), fea_top_diversity_comb)

            fea_top_diversity_comb_pair = np.array(fea_top_diversity_comb_pair)
            print(fea_top_diversity_comb_pair)
            np.save(os.path.join(rate_path, str(class_num) + '_fea_top_diversity_comb_pair.npy'), fea_top_diversity_comb_pair)

            fea_top_diversity_w = np.array(fea_top_diversity_w)
            print(fea_top_diversity_w)
            np.save(os.path.join(rate_path, str(class_num) + '_fea_top_diversity_w.npy'), fea_top_diversity_w)

            fea_top_diversity_value = np.array(fea_top_diversity_value)
            print(fea_top_diversity_value)
            np.save(os.path.join(rate_path, str(class_num) + '_fea_top_diversity_value.npy'), fea_top_diversity_value)

            fea_name = np.array(fea_name)
            print(fea_name)
            np.save(os.path.join(rate_path, str(class_num) + '_fea_name.npy'), fea_name)

            fea_top_diversity_pic_name = np.array(fea_top_diversity_pic_name)
            print(fea_top_diversity_pic_name)
            np.save(os.path.join(rate_path, str(class_num) + '_fea_top_diversity_pic_name.npy'), fea_top_diversity_pic_name)



if __name__ == '__main__':
    pass


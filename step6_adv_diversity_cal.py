import math
import os
import numpy as np
from keras.models import load_model
from keras import Model
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

def sel_adv(class_list, adv_path, adv_fea_path, adv_div_path, fea_num):
    if not os.path.exists(adv_div_path):
        os.mkdir(adv_div_path)
    for class_num in range(len(class_list)):
        fea_pic_num = np.load(os.path.join(adv_fea_path, str(class_num) + '_fea_pic_num.npy'))
        adv_x = np.load(os.path.join(adv_path, 'adv_npy', str(class_num) + '_adv_x.npy'))
        adv_fea_index = np.load(os.path.join(adv_fea_path, str(class_num) + '_adv_fea_index.npy'))
        adv_fea_score = np.load(os.path.join(adv_fea_path, str(class_num) + '_adv_fea_score.npy'))
        sorted_indices = np.argsort(-fea_pic_num)
        sel_fea_index_arr = []
        for i in range(fea_num):
            sel_fea_index_arr.append(sorted_indices[i])
        sel_fea_index_arr = np.array(sel_fea_index_arr)
        print(adv_fea_index.shape)
        for sel_fea_index_num in range(len(sel_fea_index_arr)):
            print(sel_fea_index_arr[sel_fea_index_num])
            retrain_x = []
            retrain_y = []
            scores = []
            for adv_fea_index_num in range(len(adv_fea_index)):
                if adv_fea_index[adv_fea_index_num] == sel_fea_index_arr[sel_fea_index_num]:
                    retrain_x.append(adv_x[adv_fea_index_num])
                    retrain_y.append(class_num)
                    scores.append(adv_fea_score[adv_fea_index_num])
            retrain_x = np.array(retrain_x)
            retrain_y = np.array(retrain_y)
            scores = np.array(scores)
            print(retrain_x[0])
            print(retrain_y[0])
            print(scores[0])
            print(retrain_x.shape)
            print(retrain_y.shape)
            print(scores.shape)
            np.save(os.path.join(adv_div_path, str(class_num) + '_' + str(sel_fea_index_num) + '_retrain_x.npy'), retrain_x)
            np.save(os.path.join(adv_div_path, str(class_num) + '_' + str(sel_fea_index_num) + '_retrain_y.npy'), retrain_y)
            np.save(os.path.join(adv_div_path, str(class_num) + '_' + str(sel_fea_index_num) + '_scores.npy'), scores)

def cal_act(class_list, fea_num, adv_div_path, model_path, model_name):
    for class_num in range(len(class_list)):
        for fea_num_index in range(fea_num):
            retrain_x = np.load(os.path.join(adv_div_path, str(class_num) + '_' + str(fea_num_index) + '_retrain_x.npy'))
            g_model = tf.Graph()
            g_session = tf.Session(graph=g_model)
            with g_session.as_default():
                with g_model.as_default():
                    model = load_model(model_path)
                    if model_name == 'vgg_19':  # 33, 35, Conv; 39, 40, Dense;
                        i = Model(inputs=model.layers[0].output, outputs=model.layers[35].output)
                        temp_1 = i.predict(retrain_x).reshape(len(retrain_x), -1, i.output.shape[-1])
                        temp_1 = np.mean(temp_1, axis=1)
                    if model_name == 'resnet_50':  # 169, 173, Conv;
                        i = Model(inputs=model.layers[0].output, outputs=model.layers[173].output)
                        temp_1 = i.predict(retrain_x).reshape(len(retrain_x), -1, i.output.shape[-1])
                        temp_1 = np.mean(temp_1, axis=1)
                    print(temp_1[0])
            np.save(os.path.join(adv_div_path, str(class_num) + '_' + str(fea_num_index) + '_act.npy'), temp_1)

def cal_dis(act_1, act_2):
    temp_dis = 0.0
    for index in range(len(act_1)):
        temp_dis = temp_dis + (act_1[index] - act_2[index]) * (act_1[index] - act_2[index])
    temp_dis = math.sqrt(temp_dis)
    return temp_dis

def cal_graph(acts):
    dis = np.zeros((len(acts), len(acts)), dtype='float64')
    for i in range(len(acts)):
        for j in range(len(acts)):
            if i > j:
                continue
            elif i == j:
                dis[j][i] = 0
            else:
                dis[i][j] = cal_dis(acts[i], acts[j])
                dis[j][i] = dis[i][j]
    return dis


def cal_entropy(w):
    if len(w) == 1:
        return 0
    w = np.array(w)
    w_sum = np.sum(w)
    # print('w_sum: ' + str(w_sum))
    h = 0.0
    for i in range(len(w)):
        temp = w[i] / w_sum
        h = h + (-1) * temp * math.log(temp)
    # print('h: ' + str(h))
    # h = w_sum * h
    h = w_sum / (1.0 + h)
    # print('final: ' + str(h))
    return h


def not_find(find_arr, dis_arr):
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

def cal_tree(vertex_arr, dis_arr):
    find_arr = vertex_arr
    not_find_arr = not_find(find_arr, dis_arr)
    find_pair_arr = []
    w = []
    h = -1
    while len(not_find_arr) != 0:
        temp_x, temp_x_index, temp_y, temp_y_index, min_dis = cal_min_dis(find_arr, not_find_arr, dis_arr)
        temp_w = w.copy()
        temp_w.append(min_dis)
        temp_h = cal_entropy(temp_w)
        if temp_h > h:
            find_arr.append(temp_y)
            temp_list = []
            temp_list.append(temp_x)
            temp_list.append(temp_y)
            find_pair_arr.append(temp_list)
            w.append(min_dis)
            h = temp_h
        del not_find_arr[temp_y_index]
    return find_arr, find_pair_arr, w, h

def cal_div(class_list, fea_num, adv_div_path):
    for class_num in range(len(class_list)):
        for fea_num_index in range(fea_num):
            acts = np.load(os.path.join(adv_div_path, str(class_num) + '_' + str(fea_num_index) + '_act.npy'))
            scores = np.load(os.path.join(adv_div_path, str(class_num) + '_' + str(fea_num_index) + '_scores.npy'))
            ini_index = np.argmax(scores)
            dis_arr = cal_graph(acts)
            vertex_arr = []
            vertex_arr.append(ini_index)
            max_value = np.max(dis_arr)
            dis_arr = dis_arr / max_value
            print(dis_arr)
            for temp_i in range(len(dis_arr)):
                for temp_j in range(len(dis_arr)):
                    if temp_i == temp_j:
                        dis_arr[temp_i][temp_j] = 1000
            find_arr, find_pair_arr, w, h = cal_tree(vertex_arr, dis_arr)
            print(len(vertex_arr))
            print(len(find_arr))
            print(find_arr)
            # print(len(x))
            break
        break


if __name__ == '__main__':
    class_list = ['cat', 'dog']
    model_name = 'vgg_19'
    adv_fea_path = 'E:/dataset/dataset_2/fea_select/adv_fea'
    adv_path = 'E:/dataset/dataset_2/fea_select/adv'
    adv_div_path = 'E:/dataset/dataset_2/fea_select/adv_div'
    model_path = 'E:/pycharmproject/pythonProject/pythonProject/DeepRF/model/vgg_19/vgg_19.h5'
    fea_num = 3
    # sel_adv(class_list, adv_path, adv_fea_path, adv_div_path, model_path, fea_num)
    # cal_act(class_list, fea_num, adv_div_path, model_path, model_name)
    cal_div(class_list, fea_num, adv_div_path)

import os
import tensorflow as tf
import keras.backend as backend
import foolbox as fb
import numpy as np
import skimage.io as io
import keras
from keras.models import load_model

def save_adv_npy(model_path, load_path, save_path, adv_list):
    m = load_model(model_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for adv_name in adv_list:
        print(adv_name + ':')
        save_adv_path = os.path.join(save_path, adv_name)
        if not os.path.exists(save_adv_path):
            os.makedirs(save_adv_path)
        adv_path = os.path.join(load_path, adv_name)
        for class_num in range(10):
            print(str(class_num) + ':')
            adv_arr_path = os.path.join(adv_path, str(class_num) + '_adv_x.npy')
            adv_name_path = os.path.join(adv_path, str(class_num) + '_adv_name.npy')
            adv_x = np.load(adv_arr_path)
            adv_name1 = np.load(adv_name_path)
            print(adv_x.shape)
            print(adv_name1.shape)
            adv_y = m.predict(adv_x)
            adv_y = np.argmax(adv_y, axis=1)
            temp_x = []
            temp_name = []
            for adv_num in range(len(adv_y)):
                if adv_y[adv_num] != class_num:
                    temp_x.append(adv_x[adv_num])
                    temp_name.append(adv_name1[adv_num])
            save_adv_arr_path = os.path.join(save_adv_path, str(class_num) + '_adv_x.npy')
            save_adv_name_path = os.path.join(save_adv_path, str(class_num) + '_adv_name.npy')
            temp_x = np.array(temp_x)
            temp_name = np.array(temp_name)
            print(temp_x.shape)
            print(temp_name.shape)
            np.save(save_adv_arr_path, temp_x)
            np.save(save_adv_name_path, temp_name)

def save_remain_npy(dataset_path, save_path, adv_list):
    for adv_name in adv_list:
        print(adv_name + ':')
        for class_num in range(10):
            print(str(class_num) + ':')
            remain_x = []
            remain_name = []
            not_remain_x = []
            x_path = os.path.join(dataset_path, str(class_num) + '_test_x.npy')
            name_path = os.path.join(dataset_path, str(class_num) + '_test_name.npy')
            all_x = np.load(x_path)
            all_name = np.load(name_path)
            print('all: ' + str(len(all_x)))
            adv_name_path = os.path.join(save_path, adv_name, str(class_num) + '_adv_name.npy')
            adv_x_name = np.load(adv_name_path)
            print('adv: ' + str(len(adv_x_name)))
            for all_name_num in range(len(all_name)):
                img_base, img_txt = os.path.splitext(all_name[all_name_num])
                if img_base not in adv_x_name:
                    remain_x.append(all_x[all_name_num])
                    remain_name.append(all_name[all_name_num])

            remain_x = np.array(remain_x)

            remain_name = np.array(remain_name)
            print('remain: ' + str(len(remain_name)))
            save_remain_x_path = os.path.join(save_path, adv_name, str(class_num) + '_remain_x.npy')
            save_remain_name_path = os.path.join(save_path, adv_name, str(class_num) + '_remain_name.npy')
            np.save(save_remain_x_path, remain_x)
            np.save(save_remain_name_path, remain_name)

def merge_test_data(model_path, save_path, adv_list):
    m = load_model(model_path)
    for adv_name in adv_list:
        print(adv_name + ' :')
        for class_num in range(10):
            print(str(class_num) + ':')
            adv_x_path = os.path.join(save_path, adv_name, str(class_num) + '_adv_x.npy')
            adv_name_path = os.path.join(save_path, adv_name, str(class_num) + '_adv_name.npy')
            remain_x_path = os.path.join(save_path, adv_name, str(class_num) + '_remain_x.npy')
            remain_name_path = os.path.join(save_path, adv_name, str(class_num) + '_remain_name.npy')
            adv_x = np.load(adv_x_path)
            remain_x = np.load(remain_x_path)
            all_x = np.concatenate((adv_x, remain_x), axis=0)
            adv_name1 = np.load(adv_name_path)
            remain_name = np.load(remain_name_path)
            all_name = np.concatenate((adv_name1, remain_name), axis=0)
            np.save(os.path.join(save_path, adv_name, str(class_num) + '_all_x.npy'), all_x)
            np.save(os.path.join(save_path, adv_name, str(class_num) + '_all_name.npy'), all_name)
            result = m.predict(all_x)
            result = np.argmax(result, axis=1)
            count = np.count_nonzero(result == class_num)
            print('正确个数：' + str(count))
if __name__ == '__main__':
    pass

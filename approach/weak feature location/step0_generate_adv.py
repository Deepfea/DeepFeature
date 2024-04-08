import os
import tensorflow as tf
import keras.backend as backend
import foolbox as fb
import numpy as np
import skimage.io as io
import keras
from keras.models import load_model
import os

def gen_adv(adv_name, img_arr_x, img_arr_y, img_arr_name, model_path, adv_npy_path, class_num):
    model = load_model(model_path)
    bounds = (0, 1)
    fool_model = fb.models.KerasModel(model, bounds, preprocessing=(0, 1))
    for arr_num in range(len(img_arr_x)):
        print('Start: ', str(arr_num))
        temp_x = []
        temp_y = []
        temp_x.append(img_arr_x[arr_num])
        temp_y.append(img_arr_y[arr_num])
        temp_x = np.array(temp_x)
        temp_y = np.array(temp_y)
        if adv_name == 'cw':
            attack = fb.attacks.CarliniWagnerL2Attack(fool_model)
            adv = attack(temp_x, temp_y)
        elif adv_name == 'fgsm':
            attack = fb.attacks.FGSM(fool_model)
            adv = attack(temp_x, temp_y)
            # adv = attack(temp_x, temp_y, [0.01, 0.1], max_epsilon=0)
        elif adv_name == 'bim':
            attack = fb.attacks.BIM(fool_model)
            adv = attack(temp_x, temp_y)
        elif adv_name == 'jsma':
            attack = fb.attacks.SaliencyMapAttack(fool_model)
            # adv = attack(temp_x, temp_y, max_iter=2000, fast=True, theta=0.3, max_perturbations_per_pixel=20)
            adv = attack(temp_x, temp_y, max_perturbations_per_pixel=100)
        has_nan = np.any(np.isnan(adv))
        if has_nan == True:
            print("can't find.")
            continue
        img_name = img_arr_name[arr_num]
        img_base, img_txt = os.path.splitext(img_name)
        print(img_base)
        np.save(os.path.join(adv_npy_path, img_base + '_adv_x.npy'), adv)
        np.save(os.path.join(adv_npy_path, img_base + '_adv_y.npy'), temp_y)
        print(adv.shape)
        print(temp_y.shape)
    backend.clear_session()

def get_adv_npy(save_path, adv_list, model_path):
    if not os.path.exists(load_path):
        os.mkdir(load_path)
    for adv_name in adv_list:
        adv_name_path = os.path.join(save_path, adv_name)
        if not os.path.exists(adv_name_path):
            os.mkdir(adv_name_path)
        adv_npy_path = os.path.join(adv_name_path, 'npy')
        if not os.path.exists(adv_npy_path):
            os.mkdir(adv_npy_path)
        for class_num in range(10):
            save_class_path = os.path.join(adv_npy_path, str(class_num))
            if not os.path.exists(save_class_path):
                os.mkdir(save_class_path)
            npy_x = np.load(os.path.join(save_path, str(class_num) + '_test_x.npy'))
            npy_y = np.load(os.path.join(save_path, str(class_num) + '_test_y.npy'))
            npy_name = np.load(os.path.join(save_path, str(class_num) + '_test_name.npy'))
            print('Class: ', str(class_num))
            print(npy_x.shape)
            gen_adv(adv_name, npy_x, npy_y, npy_name, model_path, save_class_path, class_num)

def save_adv_pic(save_path, adv_list):
    for adv_name in adv_list:
        print(adv_name + ':')
        adv_npy_path = os.path.join(save_path, adv_name, 'npy')
        adv_pic_path = os.path.join(save_path, adv_name, 'pic')
        if not os.path.exists(adv_pic_path):
            os.makedirs(adv_pic_path)
        for class_num in range(10):
            adv_class_npy_path = os.path.join(adv_npy_path, str(class_num))
            adv_class_pic_path = os.path.join(adv_pic_path, str(class_num))
            if not os.path.exists(adv_class_pic_path):
                os.makedirs(adv_class_pic_path)
            adv_name_path = os.path.join(save_path, str(class_num) + '_test_name.npy')
            adv_name_npy = np.load(adv_name_path)
            adv_arr = []
            adv_pic_name = []
            for adv_num in range(len(adv_name_npy)):
                img_base, img_txt = os.path.splitext(adv_name_npy[adv_num])
                # print(img_base)
                adv_pic_npy_path = os.path.join(adv_class_npy_path, img_base + '_adv_x.npy')
                if not os.path.exists(adv_pic_npy_path):
                    print(img_base + ' : 对抗样本不存在!')
                    continue
                adv_pic_npy = np.load(adv_pic_npy_path)[0]
                adv_arr.append(adv_pic_npy)
                adv_pic_name.append(img_base)
                adv_pic_x_path = os.path.join(adv_class_pic_path, adv_name_npy[adv_num])
                temp_arr = (adv_pic_npy * 255).astype('uint8')
                io.imsave(adv_pic_x_path, temp_arr)
            adv_arr = np.array(adv_arr)
            adv_arr_path = os.path.join(save_path, adv_name, str(class_num) + '_adv_x.npy')
            np.save(adv_arr_path, adv_arr)
            adv_pic_name = np.array(adv_pic_name)
            adv_name_path = os.path.join(save_path, adv_name, str(class_num) + '_adv_name.npy')
            np.save(adv_name_path, adv_pic_name)
            print(adv_arr.shape)
            print(adv_pic_name.shape)

def save_error_test_data(load_path, save_path, model_path):
    for class_num in range(10):
        npy_x = np.load(os.path.join(load_path, str(class_num) + '_test_x.npy'))
        npy_y = np.load(os.path.join(load_path, str(class_num) + '_test_y.npy'))
        npy_name = np.load(os.path.join(load_path, str(class_num) + '_test_name.npy'))
        # print(npy_name)
        model = load_model(model_path)
        predict_y = model.predict(npy_x)
        predict_y = np.argmax(predict_y, axis=1)
        print(predict_y.shape)
        save_x = []
        save_y = []
        save_name = []
        for pic_num in range(len(predict_y)):
            if predict_y[pic_num] == npy_y[pic_num]:
                save_x.append(npy_x[pic_num])
                save_y.append(npy_y[pic_num])
                save_name.append(npy_name[pic_num])
        save_x = np.array(save_x)
        print(save_x.shape)
        save_y = np.array(save_y)
        save_name = np.array(save_name)
        np.save(os.path.join(save_path, str(class_num) + '_test_x.npy'), save_x)
        np.save(os.path.join(save_path, str(class_num) + '_test_y.npy'), save_y)
        np.save(os.path.join(save_path, str(class_num) + '_test_name.npy'), save_name)

if __name__ == '__main__':
    pass








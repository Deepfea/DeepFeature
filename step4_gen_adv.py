import os
import tensorflow as tf
import keras.backend as backend
import foolbox as fb
import numpy as np
import skimage.io as io
import keras
from keras.models import load_model
import os
def save_ori_npy(npy_path, adv_path, model_path):
    if not os.path.exists(adv_path):
        os.mkdir(adv_path)
    ori_npy_path = os.path.join(adv_path, 'ori_npy')
    train_x = np.load(os.path.join(npy_path, 'train_x.npy'))
    train_y = np.load(os.path.join(npy_path, 'train_y.npy'))
    if not os.path.exists(ori_npy_path):
        os.mkdir(ori_npy_path)
    model = load_model(model_path)
    predict_y = model.predict(train_x)
    predict_y = np.argmax(predict_y, axis=1)
    ori_pic_arr = []
    ori_label_arr = []
    num = 0
    for i in range(len(predict_y)):
        if int(train_y[i]) == int(predict_y[i]):
            ori_pic_arr.append(train_x[i])
            ori_label_arr.append(predict_y[i])
            num += 1
    ori_pic_arr = np.array(ori_pic_arr)
    ori_label_arr = np.array(ori_label_arr)
    print(ori_label_arr.shape)
    print(ori_pic_arr.shape)
    np.save(os.path.join(ori_npy_path, 'ori_x.npy'), ori_pic_arr)
    np.save(os.path.join(ori_npy_path, 'ori_y.npy'), ori_label_arr)

def save_ori_pic(adv_path):
    ori_pic_path = os.path.join(adv_path, 'ori_pic')
    if not os.path.exists(ori_pic_path):
        os.mkdir(ori_pic_path)
    ori_pic_arr = np.load(os.path.join(adv_path, 'ori_npy', 'ori_x.npy'))
    ori_label_arr = np.load(os.path.join(adv_path, 'ori_npy', 'ori_y.npy'))
    label_list = np.unique(ori_label_arr)
    for i in range(len(label_list)):
        num = 0
        ori_class_path = os.path.join(ori_pic_path, str(i))
        if not os.path.exists(ori_class_path):
            os.mkdir(ori_class_path)
        for j in range(len(ori_label_arr)):
            if(int(ori_label_arr[j])!=int(label_list[i])):
                continue
            img = (ori_pic_arr[j]*255).astype('uint8')
            io.imsave(os.path.join(ori_class_path, str(i) + '_' + str(num) + '.jpg'), img)
            num += 1
def get_adv_npy(adv_path, adv_name, model_path):
    adv_npy_path = os.path.join(adv_path, 'adv_npy')
    if not os.path.exists(adv_npy_path):
        os.mkdir(adv_npy_path)
    model = load_model(model_path)
    bounds = (0, 1)
    fool_model = fb.models.KerasModel(model, bounds, preprocessing=(0, 1))
    if adv_name == 'cw':
        attack = fb.attacks.CarliniWagnerL2Attack(fool_model)
    elif adv_name == 'fgsm':
        attack = fb.attacks.FGSM(fool_model)
    elif adv_name == 'bim':
        attack = fb.attacks.L1BasicIterativeAttack(fool_model)
    elif adv_name == 'jsma':
        attack = fb.attacks.SaliencyMapAttack(fool_model)
    else:
        print('对抗生成名字错误，不生成图片！！！')
    ori_x = np.load(os.path.join(adv_path, 'ori_npy', 'ori_x.npy'))
    ori_y = np.load(os.path.join(adv_path, 'ori_npy', 'ori_y.npy'))
    class_list = np.unique(ori_y)
    for class_name in class_list:
        adv_class_path = os.path.join(adv_npy_path, str(class_name))
        if not os.path.exists(adv_class_path):
            os.mkdir(adv_class_path)
    label_num = 30
    for num in range(len(ori_x)):
        if num <= 673:
            continue
        if ori_y[num] == ori_y[0]:
            print(num)
            continue
        temp_arr = []
        temp_arr.append(ori_x[num])
        temp_arr = np.array(temp_arr)
        temp_label = []
        temp_label.append(ori_y[num])
        temp_label = np.array(temp_label)
        if adv_name == 'fgsm':
            adv = attack(temp_arr, temp_label, [0.01, 0.1], max_epsilon=0)
        elif adv_name == 'cw':
            adv = attack(temp_arr, temp_label)
        elif adv_name == 'bim':
            attack = attack(temp_arr, temp_label)
        elif adv_name == 'jsma':
            attack = attack(temp_arr, temp_label, max_iter=2000, fast=True, theta=0.3, max_perturbations_per_pixel=20)
        predict_y = model.predict(adv)
        print(predict_y)
        predict_y = np.argmax(predict_y, axis=1)
        print(str(ori_y[num]) + '_' + str(num) + '.jpg')
        print("原始标签：" + str(ori_y[num]))
        print("预测标签：" + str(predict_y))
        adv_pic_path = os.path.join(adv_npy_path, str(ori_y[num]), str(ori_y[num]) + '_' + str(label_num) + '.npy')
        np.save(adv_pic_path, adv)
        label_num += 1

def save_adv_npy_and_pic(adv_path, class_list):
    npy_save_path = os.path.join(adv_path, 'adv_npy')
    all_x = []
    all_y = []
    all_name = []
    for i in range(len(class_list)):
        adv_x = []
        adv_y = []
        adv_file_name = []
        npy_load_path = os.path.join(npy_save_path, str(i))
        file = os.listdir(npy_load_path)
        for j in range(len(file)):
            print(os.path.join(npy_load_path, str(i) + '_' + str(j) + '.npy'))
            temp_x = np.load(os.path.join(npy_load_path, str(i) + '_' + str(j) + '.npy'))[0]
            temp_y = i
            temp_name = str(i) + '_' + str(j) + '.jpg'
            adv_x.append(temp_x)
            adv_y.append(temp_y)
            adv_file_name.append(temp_name)
            all_x.append(temp_x)
            all_y.append(temp_y)
            all_name.append(temp_name)
        adv_x = np.array(adv_x)
        adv_y = np.array(adv_y)
        adv_file_name = np.array(adv_file_name)
        print(adv_x.shape)
        print(adv_y.shape)
        print(adv_file_name.shape)
        np.save(os.path.join(npy_save_path, str(i) + '_adv_x.npy'), adv_x)
        np.save(os.path.join(npy_save_path, str(i) + '_adv_y.npy'), adv_y)
        np.save(os.path.join(npy_save_path, str(i) + '_adv_name.npy'), adv_file_name)
    all_x = np.array(all_x)
    all_y = np.array(all_y)
    all_name = np.array(all_name)
    print(all_x.shape)
    print(all_y.shape)
    print(all_name.shape)
    np.save(os.path.join(npy_save_path, 'adv_x.npy'), all_x)
    np.save(os.path.join(npy_save_path, 'adv_y.npy'), all_y)
    np.save(os.path.join(npy_save_path, 'adv_name.npy'), all_name)

    pic_save_path = os.path.join(adv_path, 'adv_pic')
    if not os.path.exists(pic_save_path):
        os.mkdir(pic_save_path)
    for i in range(len(all_name)):
        temp_x = all_x[i]
        temp_y = all_y[i]
        file_name = all_name[i]
        adv_class_path = os.path.join(pic_save_path, str(temp_y))
        if not os.path.exists(adv_class_path):
            os.mkdir(adv_class_path)
        temp_x = (temp_x * 255).astype('uint8')
        io.imsave(os.path.join(adv_class_path, file_name), temp_x)

if __name__ == '__main__':
    adv_name = 'cw'
    class_list = ['cat', 'dog']
    model_name = 'vgg_19'
    model_path = 'E:/pycharmproject/pythonProject/pythonProject/DeepRF/model/vgg_19/vgg_19.h5'
    score_path = 'E:/dataset/dataset_2/fea_select/score'
    npy_path = 'E:/dataset/dataset_2/model_train_and_test/npy'
    adv_path = 'E:/dataset/dataset_2/fea_select/adv'
    # save_ori_npy(npy_path, adv_path, model_path)
    # save_ori_pic(adv_path)
    # get_adv_npy(adv_path, adv_name, model_path)
    # save_adv_npy_and_pic(adv_path, class_list)





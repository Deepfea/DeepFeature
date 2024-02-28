import os
import numpy as np
import keras
from keras import backend as K
from keras.models import load_model
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

def load_cav(path, i):
    x_path = path + os.sep + str(i) + '_cav.npy'
    y_path = path + os.sep + str(i) + '_label.npy'
    x = np.load(x_path)
    y = np.load(y_path)
    return x, y

def get_gradients(model_path, img_arr, label_arr, input_i, type1):
    label_arr = keras.utils.to_categorical(label_arr, 2)
    g_model = tf.Graph()
    g_session = tf.Session(graph=g_model)
    with g_session.as_default():
        with g_model.as_default():
            model = load_model(model_path)
            grads = K.gradients(model.total_loss, model.layers[input_i].output)
            input_tensors = [model.inputs[0], model.sample_weights[0], model.targets[0], K.learning_phase()]
            get_gradients = K.function(inputs=input_tensors, outputs=grads)
            te = get_gradients([img_arr, np.ones(img_arr.shape[0]), label_arr, 0])[0]
    if type1 == 'conv':
        te = te.reshape(len(te), -1, model.layers[input_i].output.shape[-1])
        te = np.mean(te, axis=1)
    elif type1 == 'dense':
        te = te.reshape(len(te), model.layers[input_i].output.shape[-1])
    print(te.shape)
    return te

def get_adv_gradients(class_list, model_name, model_path, adv_path, adv_fea_path):
    for class_num in range(len(class_list)):
        adv_x_path = os.path.join(adv_path, 'adv_npy', str(class_num) + '_adv_x.npy')
        adv_y_path = os.path.join(adv_path, 'adv_npy', str(class_num) + '_adv_y.npy')
        adv_x = np.load(adv_x_path)
        adv_y = np.load(adv_y_path)
        if model_name == 'resnet_50':
            print(model_name)
            #     graArr = get_gradients(modelPath, imgArr, labelArr, 173, 'conv')
        elif model_name == 'vgg_19':
            one_time_img_num = 50
            print(len(adv_x) / one_time_img_num)
            for group_num in range(int(len(adv_x) / one_time_img_num) + 1):
                if group_num != int(len(adv_x) / one_time_img_num):
                    temp_x = np.empty((one_time_img_num, adv_x.shape[1], adv_x.shape[2], adv_x.shape[3]),
                                      dtype='float32')
                    temp_y = np.empty(one_time_img_num, dtype='float32')
                    for temp_i in range(one_time_img_num):
                        t = group_num * one_time_img_num + temp_i
                        print(t)
                        temp_x[temp_i] = adv_x[t]
                        temp_y[temp_i] = adv_y[t]
                else:
                    if len(adv_x) % one_time_img_num == 0:
                        continue
                    temp_x = np.empty(
                        (len(adv_x) % one_time_img_num, adv_x.shape[1], adv_x.shape[2], adv_x.shape[3]),
                        dtype='float32')
                    temp_y = np.empty(len(adv_y) % one_time_img_num, dtype='float32')
                    for temp_i in range(len(adv_y) % one_time_img_num):
                        t = group_num * one_time_img_num + temp_i
                        print(t)
                        temp_x[temp_i] = adv_x[t]
                        temp_y[temp_i] = adv_y[t]
                gra_arr = get_gradients(model_path, temp_x, temp_y, 35, 'conv')
                if group_num == 0:
                    temp_arr = gra_arr
                else:
                    temp_arr = np.concatenate((temp_arr, gra_arr), axis=0)
        print(temp_arr.shape)
        print(temp_arr[0])
        print(temp_arr[1])
        np.save(os.path.join(adv_fea_path, str(class_num) + '_gradient.npy'), temp_arr)

def cal_score(class_list, cav_path, adv_fea_path):
    for class_num in range(len(class_list)):
        class_cav_path = os.path.join(cav_path, str(class_num) + '_cav.npy')
        fea_name_path = os.path.join(cav_path, str(class_num) + '_label_name.npy')
        adv_gradients_path = os.path.join(adv_fea_path, str(class_num) + '_gradient.npy')
        class_cav = np.load(class_cav_path)
        fea_name = np.load(fea_name_path)
        adv_gradients = np.load(adv_gradients_path)
        gradients_score_arr = np.empty(len(adv_gradients), dtype='float32')
        gradients_index_arr = np.empty(len(adv_gradients), dtype='int')
        for gradients_num in range(len(gradients_score_arr)):
            gradients_score_arr[gradients_num] = -1.0
            gradients_index_arr[gradients_num] = -1
        for gradients_num in range(len(adv_gradients)):
            for fea_num in range(len(class_cav)):
                temp_score = np.dot(class_cav[fea_num], adv_gradients[gradients_num].T)
                if temp_score > gradients_score_arr[gradients_num]:
                    gradients_index_arr[gradients_num] = fea_num
                    gradients_score_arr[gradients_num] = temp_score
        gradients_name_arr = []
        for gradients_num in range(len(adv_gradients)):
            if gradients_index_arr[gradients_num] == -1:
                gradients_name_arr[gradients_num] = 'none'
            else:
                gradients_name_arr.append(fea_name[gradients_index_arr[gradients_num]])
        gradients_name_arr = np.array(gradients_name_arr)
        print(gradients_score_arr)
        print(gradients_index_arr)
        print(gradients_name_arr)
        np.save(os.path.join(adv_fea_path, str(class_num) + '_adv_fea_score.npy'), gradients_score_arr)
        np.save(os.path.join(adv_fea_path, str(class_num) + '_adv_fea_index.npy'), gradients_index_arr)
        np.save(os.path.join(adv_fea_path, str(class_num) + '_adv_fea_name.npy'), gradients_name_arr)

def cal_fea_pic_num(class_list, cav_path, adv_fea_path):
    for class_num in range(len(class_list)):
        fea_name_path = os.path.join(cav_path, str(class_num) + '_label_name.npy')
        fea_name = np.load(fea_name_path)
        fea_pic_num = np.zeros(len(fea_name), dtype='int')
        adv_fea_index = np.load(os.path.join(adv_fea_path, str(class_num) + '_adv_fea_index.npy'))
        for pic_num in range(len(adv_fea_index)):
            fea_pic_num[adv_fea_index[pic_num]] += 1
        np.save(os.path.join(adv_fea_path, str(class_num) + '_fea_pic_num.npy'), fea_pic_num)
        print(class_list[class_num] + ":")
        print(fea_pic_num)


if __name__ == '__main__':
    class_list = ['cat', 'dog']
    model_name = 'vgg_19'
    cav_path = 'E:/dataset/dataset_2/fea_select/cav'
    adv_fea_path = 'E:/dataset/dataset_2/fea_select/adv_fea'
    adv_path = 'E:/dataset/dataset_2/fea_select/adv'
    if not os.path.exists(adv_fea_path):
        os.mkdir(adv_fea_path)
    model_path = 'E:/pycharmproject/pythonProject/pythonProject/DeepRF/model/vgg_19/vgg_19.h5'
    # get_adv_gradients(class_list, model_name, model_path, adv_path, adv_fea_path)
    # cal_score(class_list, cav_path, adv_fea_path)
    # cal_fea_pic_num(class_list, cav_path, adv_fea_path)


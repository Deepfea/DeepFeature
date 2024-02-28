import os
import numpy as np
import keras
from keras import backend as K
from keras.models import load_model, Model
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

def cal_gradient(model_path, model_name, class_path, save_path, cav, num, class_name):
    if model_name == 'resnet_50':
        print(model_name)
        #     graArr = get_gradients(modelPath, imgArr, labelArr, 173, 'conv')
    elif model_name == 'vgg_19':
        train_x = np.load(os.path.join(class_path, class_name + '_train_x.npy'))
        train_y = np.load(os.path.join(class_path, class_name + '_train_y.npy'))
        one_time_img_num = 50
        print(len(train_y)/one_time_img_num)
        for group_num in range(int(len(train_y)/one_time_img_num)+1):
            if group_num != int(len(train_y)/one_time_img_num):
                temp_x = np.empty((one_time_img_num, train_x.shape[1], train_x.shape[2], train_x.shape[3]),
                                  dtype='float32')
                temp_y = np.empty(one_time_img_num, dtype='float32')
                for temp_i in range(one_time_img_num):
                    t = group_num * one_time_img_num + temp_i
                    print(t)
                    temp_x[temp_i] = train_x[t]
                    temp_y[temp_i] = train_y[t]
            else:
                if len(train_y) % one_time_img_num == 0:
                    continue
                temp_x = np.empty((len(train_y) % one_time_img_num, train_x.shape[1], train_x.shape[2], train_x.shape[3]),
                                  dtype='float32')
                temp_y = np.empty(len(train_y) % one_time_img_num, dtype='float32')
                for temp_i in range(len(train_y) % one_time_img_num):
                    t = group_num * one_time_img_num + temp_i
                    print(t)
                    temp_x[temp_i] = train_x[t]
                    temp_y[temp_i] = train_y[t]
            gra_arr = get_gradients(model_path, temp_x, temp_y, 35, 'conv')
            if group_num == 0:
                temp_arr = gra_arr
            else:
                temp_arr = np.concatenate((temp_arr, gra_arr), axis=0)
    print(temp_arr.shape)
    print(temp_arr[0])
    print(temp_arr[1])
    print(cav.shape)
    np.save(os.path.join(save_path, str(num) + '_gradient.npy'), temp_arr)
    np.save(os.path.join(save_path, str(num) + '_cav.npy'), cav)

def cal_score(class_num, score_path):
    cav_arr = np.load(os.path.join(score_path, str(class_num) + '_cav.npy'))
    gradient_arr = np.load(os.path.join(score_path, str(class_num) + '_gradient.npy'))
    scores = np.zeros(len(cav_arr), dtype='float32')
    for cav_num in range(len(cav_arr)):
        for gradient_num in range(len(gradient_arr)):
            temp = np.dot(cav_arr[cav_num], gradient_arr[gradient_num].T)
            scores[cav_num] += temp
        scores[cav_num] /= float(len(gradient_arr))
        print(scores[cav_num])
        print(str(class_num) + ': ' + str(cav_num) + ' : finish!')
    np.save(os.path.join(score_path, str(i) + '_scores.npy'), scores)

if __name__ == '__main__':
    class_list = ['cat', 'dog']
    model_name = 'vgg_19'
    cav_path = 'E:/dataset/dataset_2/fea_select/cav'
    score_path = 'E:/dataset/dataset_2/fea_select/score'
    if not os.path.exists(score_path):
        os.mkdir(score_path)
    npy_path = 'E:/dataset/dataset_2/model_train_and_test/npy/train'
    model_path = 'E:/pycharmproject/pythonProject/pythonProject/DeepRF/model/vgg_19/vgg_19.h5'
    if not os.path.exists(score_path):
        os.mkdir(score_path)
    # for i in range(len(class_list)):
    #     x, y = load_cav(cav_path, i)
    #     print('cav of class ' + str(i) + ' are loaded!')
    #     cal_gradient(model_path, model_name, npy_path, score_path, x, i, class_list[i])

    for i in range(len(class_list)):
        cal_score(i, score_path)
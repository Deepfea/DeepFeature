import os
import numpy as np
import keras
from keras import backend as K
from keras.models import load_model, Model
import tensorflow as tf
import math
tf.compat.v1.disable_eager_execution()

def load_cav(path, i):
    x_path = path + os.sep + str(i) + '_cav.npy'
    y_path = path + os.sep + str(i) + '_label.npy'
    x = np.load(x_path)
    y = np.load(y_path)
    return x, y


def get_gradients(model_path, img_arr, label_arr, input_i, type1):
    label_arr = keras.utils.to_categorical(label_arr, 10)
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
    return te



def cal_gradient(model_path, model_name, class_path, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for class_num in range(10):
        train_x = np.load(os.path.join(class_path, str(class_num) + '_train_x.npy'))
        train_y = np.load(os.path.join(class_path, str(class_num) + '_train_y.npy'))
        if model_name == 'lenet1':
            temp_arr = get_gradients(model_path, train_x, train_y, 3, 'conv')
        elif model_name == 'lenet5':
            temp_arr = get_gradients(model_path, train_x, train_y, 3, 'conv')
        print(temp_arr.shape)
        np.save(os.path.join(save_path, str(class_num) + '_gradient.npy'), temp_arr)

def double_act(act):
    if act[0] < 1e-20:
        act = act * 1e+20
    return act

def unit_act(act):
    u_cav = np.empty(len(act), dtype='float64')
    act = double_act(act)
    y = math.sqrt(sum(act * act))
    if y == 0:
        print('!!!!!!')
        print(act)
    for i in range(len(act)):
        u_cav[i] = act[i]/y
    return u_cav

def cal_score(cav_path, score_path):
    max_iter_list = [5000]
    alpha_list = [0.0001]
    for max_iter1 in max_iter_list:
        print(max_iter1)
        for alpha1 in alpha_list:
            print(alpha1)
            save_path = os.path.join(score_path, str(max_iter1) + '_' + str(alpha1))
            load_path = os.path.join(cav_path, str(max_iter1) + '_' + str(alpha1))
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            for class_num in range(10):
                print(class_num)
                class_cav = np.load(os.path.join(load_path, str(class_num) + '_cav.npy'))
                cav_name = np.load(os.path.join(load_path, str(class_num) + '_fea_name.npy'))
                gradient_arr = np.load(os.path.join(score_path, str(class_num) + '_gradient.npy'))
                scores = np.zeros(len(class_cav), dtype='float32')
                for cav_num in range(len(class_cav)):
                    # print(class_cav[cav_num])
                    num = 0
                    for gradient_num in range(len(gradient_arr)):
                        if np.sum(gradient_arr[gradient_num]) == 0:
                            num += 1
                            continue
                        unit_gradient_arr = unit_act(gradient_arr[gradient_num])
                        temp = np.dot(class_cav[cav_num], unit_gradient_arr.T)
                        scores[cav_num] += temp
                    scores[cav_num] /= float(len(gradient_arr) - num)
                    if scores[cav_num] > 0:
                        print(str(class_num) + ': ' + str(cav_name[cav_num]) + ' : finish!')

def val_score(cav_path, score_path):
    for class_num in range(10):
        print(class_num)
        cav_load_path = os.path.join(cav_path, str(class_num) + '_cav.npy')
        gradient_arr = np.load(os.path.join(score_path, str(class_num) + '_gradient.npy'))
        cav = np.load(cav_load_path)
        scores = np.zeros(len(cav), dtype='float32')
        for cav_num in range(len(cav)):
            num = 0
            for gradient_num in range(len(gradient_arr)):
                if np.sum(gradient_arr[gradient_num]) == 0:
                    num += 1
                    continue
                unit_gradient_arr = unit_act(gradient_arr[gradient_num])
                temp = np.dot(cav[cav_num], unit_gradient_arr.T)
                scores[cav_num] += temp
            scores[cav_num] /= float(len(gradient_arr) - num)
        np.save(os.path.join(score_path, str(class_num) + '_scores.npy'), scores)
        print(scores)


if __name__ == '__main__':
    pass
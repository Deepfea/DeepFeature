import os
import numpy as np
import keras
from keras import backend as K
from keras.models import load_model
import tensorflow as tf
import math
tf.compat.v1.disable_eager_execution()

def double_act(act):
    if (0 < act[0] < 1e-20) or (-1e-20 < act[0] < 0):
        act = act * 1e+20
    if act[0] == 0:
        for temp_i in range(len(act)):
            act[temp_i] = 1
    return act

def unit_act(act):
    act = double_act(act)
    u_cav = np.empty(len(act), dtype='float64')
    y = math.sqrt(sum(act * act))
    if y == 0:
        print('!!!!!')
    if y != 0:
        for i in range(len(act)):
            u_cav[i] = act[i]/y
    return u_cav

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
    print(te.shape)
    return te

def get_test_gradients(adv_list, model_name, model_path, save_path):
    for adv_name in adv_list:
        for class_num in range(10):
            adv_x_path = os.path.join(save_path, adv_name, str(class_num) + '_all_x.npy')
            adv_y_path = os.path.join(save_path, adv_name, str(class_num) + '_all_y.npy')
            adv_x = np.load(adv_x_path)
            adv_y = np.load(adv_y_path)
            if model_name == 'vgg16':
                temp_arr = get_gradients(model_path, adv_x, adv_y, 17, 'conv')
            elif model_name == 'resnet20':
                temp_arr = get_gradients(model_path, adv_x, adv_y, 68, 'conv')
            print(temp_arr.shape)
            print(temp_arr[0])
            print(temp_arr[1])
            np.save(os.path.join(save_path, adv_name, str(class_num) + '_gradient.npy'), temp_arr)

def cal_score(adv_list, cav_path, save_path):
    for adv_name in adv_list:
        for class_num in range(10):
            class_cav_path = os.path.join(cav_path, str(class_num) + '_robust_fea_cav.npy')
            fea_name_path = os.path.join(cav_path, str(class_num) + '_robust_fea.npy')
            adv_gradients_path = os.path.join(save_path, adv_name, 'prediction', str(class_num) + '_gradient.npy')
            class_cav = np.load(class_cav_path)
            fea_name = np.load(fea_name_path)
            adv_gradients = np.load(adv_gradients_path)
            adv_score_arr = np.empty(len(adv_gradients), dtype='float64')
            adv_index_arr = np.empty(len(adv_gradients), dtype='int')
            for gradients_num in range(len(adv_score_arr)):
                adv_score_arr[gradients_num] = 0
                adv_index_arr[gradients_num] = -1
            for gradients_num in range(len(adv_gradients)):
                unit_gradient = unit_act(adv_gradients[gradients_num])
                for fea_num in range(len(class_cav)):
                    temp_score = np.dot(class_cav[fea_num], unit_gradient.T)
                    if temp_score > adv_score_arr[gradients_num]:
                        adv_index_arr[gradients_num] = fea_num
                        adv_score_arr[gradients_num] = temp_score
            gradients_name_arr = []
            for gradients_num in range(len(adv_gradients)):
                if adv_index_arr[gradients_num] == -1:
                    gradients_name_arr.append(-1)
                    print('有负的')
                else:
                    gradients_name_arr.append(fea_name[adv_index_arr[gradients_num]])
            gradients_name_arr = np.array(gradients_name_arr)
            print(adv_score_arr)
            print(adv_index_arr)
            print(gradients_name_arr)
            np.save(os.path.join(save_path, adv_name, 'prediction', str(class_num) + '_fea_score.npy'), adv_score_arr)
            np.save(os.path.join(save_path, adv_name, 'prediction', str(class_num) + '_fea_index.npy'), adv_index_arr)
            np.save(os.path.join(save_path, adv_name, 'prediction', str(class_num) + '_fea_name.npy'), gradients_name_arr)


def get_predict_result(adv_list, model_path, test_data_path, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    m = load_model(model_path)
    for adv_name in adv_list:
        adv_save_path = os.path.join(save_path, adv_name)
        if not os.path.exists(adv_save_path):
            os.makedirs(adv_save_path)
        for class_num in range(10):
            test_data = np.load(os.path.join(test_data_path, adv_name, str(class_num) + '_all_x.npy'))
            data_name = np.load(os.path.join(test_data_path, adv_name, str(class_num) + '_all_name.npy'))
            result = m.predict(test_data)
            result = np.argmax(result, axis=1)
            np.save(os.path.join(adv_save_path, str(class_num) + '_all_y.npy'), result)
            np.save(os.path.join(adv_save_path, str(class_num) + '_all_x.npy'), test_data)
            np.save(os.path.join(adv_save_path, str(class_num) + '_all_name.npy'), data_name)
            print(test_data.shape)
            print(len(result))
            print(result)

def classify_class(adv_list, save_path):
    for adv_name in adv_list:
        predict_x = []
        predict_y = []
        predict_gradient = []
        predict_name = []
        for i in range(10):
            predict_x.append([])
            predict_y.append([])
            predict_gradient.append([])
            predict_name.append([])
        for class_num in range(10):
            all_x = np.load(os.path.join(save_path, adv_name, str(class_num) + '_all_x.npy'))
            all_y = np.load(os.path.join(save_path, adv_name, str(class_num) + '_all_y.npy'))
            all_name = np.load(os.path.join(save_path, adv_name, str(class_num) + '_all_name.npy'))
            all_gradient = np.load(os.path.join(save_path, adv_name, str(class_num) + '_gradient.npy'))
            for num in range(len(all_x)):
                class_name = all_y[num]
                predict_x[class_name].append(all_x[num])
                predict_y[class_name].append(class_name)
                predict_gradient[class_name].append(all_gradient[num])
                predict_name[class_name].append(all_name[num])
        save_predict_path = os.path.join(save_path, adv_name, 'prediction')
        if not os.path.exists(save_predict_path):
            os.makedirs(save_predict_path)
        for class_num in range(10):
            print(class_num)
            temp_x = predict_x[class_num]
            temp_y = predict_y[class_num]
            temp_gradient = predict_gradient[class_num]
            temp_name = predict_name[class_num]
            temp_x = np.array(temp_x)
            np.save(os.path.join(save_predict_path, str(class_num) + '_x.npy'), temp_x)
            temp_y = np.array(temp_y)
            np.save(os.path.join(save_predict_path, str(class_num) + '_y.npy'), temp_y)
            temp_gradient = np.array(temp_gradient)
            np.save(os.path.join(save_predict_path, str(class_num) + '_gradient.npy'), temp_gradient)
            temp_name = np.array(temp_name)
            np.save(os.path.join(save_predict_path, str(class_num) + '_name.npy'), temp_name)
            print(len(temp_name))

if __name__ == '__main__':
    pass


import dataset_1.lenet1.approach.step2_train_and_test_model as step2_train_and_test_model
import os
import re
import numpy as np
import keras
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from keras.models import load_model
tf.compat.v1.disable_eager_execution()

def evaluate_ori_aca_and_acac(adv_list, load_path, model_path):
    print(model_path)
    for adv_name in adv_list:
        print(adv_name + ':')
        for class_num in range(10):
            temp_x = np.load(os.path.join(load_path, adv_name, str(class_num) + '_all_x.npy'))
            temp_y = np.zeros(len(temp_x), dtype='int')
            for temp_num in range(len(temp_y)):
                temp_y[temp_num] = class_num
            if class_num == 0:
                test_x = temp_x
                test_y = temp_y
            else:
                test_x = np.concatenate((test_x, temp_x), axis=0)
                test_y = np.concatenate((test_y, temp_y), axis=0)

        temp_model = load_model(model_path)

        confidence_result = temp_model.predict(test_x)
        class_result = np.argmax(confidence_result, axis=1)
        aca_value = 0.0
        for temp_num in range(len(class_result)):
            if int(class_result[temp_num]) == int(test_y[temp_num]):
                aca_value += 1
        aca_value = aca_value / float(len(class_result))

        acac_value = 0.0
        for temp_num in range(len(class_result)):
            temp_value = np.sum(confidence_result[temp_num]) - confidence_result[temp_num][int(test_y[temp_num])]
            temp_value = temp_value / 9.0
            acac_value += temp_value
        acac_value = acac_value / float(len(class_result))
        print('aca_value:' + str(aca_value))
        print('acac_value:' + str(acac_value))

def evaluate_aca_and_acac(model_name, adv_list, load_path, retrained_model_path,rate):
    for adv_name in adv_list:
        print(adv_name + ':')
        load_test_data = os.path.join(load_path, adv_name)
        for class_num in range(10):
            temp_x = np.load(os.path.join(load_test_data, str(class_num) + '_all_x.npy'))
            temp_y = np.zeros(len(temp_x), dtype='int')
            for temp_num in range(len(temp_y)):
                temp_y[temp_num] = class_num

            if class_num == 0:
                test_x = temp_x
                test_y = temp_y
            else:
                test_x = np.concatenate((test_x, temp_x), axis=0)
                test_y = np.concatenate((test_y, temp_y), axis=0)
        for temp_rate in rate:
            model_path = os.path.join(retrained_model_path, adv_name, str(temp_rate), model_name + '.h5')
            temp_model = load_model(model_path)

            confidence_result = temp_model.predict(test_x)
            class_result = np.argmax(confidence_result, axis=1)
            aca_value = 0.0
            for temp_num in range(len(class_result)):
                if int(class_result[temp_num]) == int(test_y[temp_num]):
                    aca_value += 1
            aca_value = aca_value / float(len(class_result))

            acac_value = 0.0
            for temp_num in range(len(class_result)):
                temp_value = np.sum(confidence_result[temp_num]) - confidence_result[temp_num][int(test_y[temp_num])]
                temp_value = temp_value / 9.0
                acac_value += temp_value
            acac_value = acac_value / float(len(class_result))
            print(model_name + '(' + str(temp_rate) + '):')
            print('aca_value:' + str(aca_value))
            print('acac_value:' + str(acac_value))

def evaluate_ave_value(adv_list, retrained_model_path, rate):
    for adv_name in adv_list:
        print(adv_name)
        for temp_rate in rate:
            print(temp_rate)
            score = 0

            for class_num in range(10):
                # print(class_num)
                class_score_path = os.path.join(retrained_model_path, adv_name, str(temp_rate),  str(class_num) + '_select_score.npy')
                class_score = np.load(class_score_path)
                if len(class_score) == 0:
                    ave_value = 0
                else:
                    ave_value = np.average(class_score)
                # print(ave_value)
                score += ave_value
            score = score / 10.0
            print(score)

if __name__ == '__main__':
    dataset_name = 'dataset_3'
    model_name = 'vgg19'
    adv_list = ['cw', 'bim', 'jsma', 'fgsm']
    rate = [0.05, 0.15, 0.25]
    index_name = 'DeepGini'

    ori_model_path = os.path.join('/media/usr/external/home/usr/project/deepfeature_data', dataset_name, model_name,
                              'model', model_name + '.h5')
    retrained_model_path = os.path.join('/media/usr/external/home/usr/project/deepfeature_data', dataset_name, model_name, 'experiment/experiment_2', index_name, '1_select_data')

    load_adv_path = os.path.join('/media/usr/external/home/usr/project/deepfeature_data/', dataset_name, model_name,
                                 'approach/3_test_data')
    rate = [0.05, 0.15, 0.25]

    # evaluate_ori_aca_and_acac(adv_list, load_adv_path, ori_model_path)

    evaluate_aca_and_acac(model_name, adv_list, load_adv_path, retrained_model_path, rate)

    evaluate_ave_value(adv_list, retrained_model_path, rate)
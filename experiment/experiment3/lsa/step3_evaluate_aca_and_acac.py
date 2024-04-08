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

def evaluate_coverage(adv_list, retrained_model_path, rate):
    for adv_name in adv_list:
        print(adv_name)
        for temp_rate in rate:
            print(temp_rate)
            cov_rate = 0
            for class_num in range(10):
                bucket_path = os.path.join(retrained_model_path, adv_name, str(temp_rate),  str(class_num) + '_select_bucket.npy')
                class_buckets = np.load(bucket_path)
                class_buckets = np.unique(class_buckets)
                cov_rate += len(class_buckets)
            cov_rate = cov_rate / 10000.0
            print(cov_rate)

if __name__ == '__main__':
    pass
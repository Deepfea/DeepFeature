import numpy as np
from keras.models import load_model, Model
import os
from tqdm import tqdm
import math

def gen_sadl_layers(model_name, model_path):
    model = load_model(model_path)
    if model_name == 'lenet1':
        input = model.layers[0].output
        layers = [model.layers[3].output]
        layers = list(zip(1 * ['conv'], layers))
    elif model_name == 'lenet5':
        input = model.layers[0].output
        layers = [model.layers[3].output]
        layers = list(zip(1 * ['conv'], layers))
    elif model_name == 'resnet20':
        input = model.layers[0].output
        layers = [model.layers[68].output]
        layers = list(zip(1 * ['conv'], layers))
    elif model_name == 'vgg16':
        input = model.layers[0].output
        layers = [model.layers[17].output]
        layers = list(zip(1 * ['conv'], layers))
    elif model_name == 'resnet50':
        input = model.layers[0].output
        layers = [model.layers[173].output]
        layers = list(zip(1 * ['conv'], layers))
    elif model_name == 'vgg19':
        input = model.layers[0].output
        layers = [model.layers[35].output]
        layers = list(zip(1 * ['conv'], layers))
    return input, layers


def gen_model(layers, input):
    model = []
    index = []
    for name, layer in layers:
        m = Model(inputs=input, outputs=layer)
        model.append(m)
        index.append(name)
    models = list(zip(index, model))
    return models

def gen_neuron_activate(models, x, std, period='train'):
    neuron_activate = []
    mask = []
    for index, model in models:
        if index == 'conv':
            temp = model.predict(x).reshape(len(x), -1, model.output.shape[-1])
            temp = np.mean(temp, axis=1)
        if index == 'dense':
            temp = model.predict(x).reshape(len(x), model.output.shape[-1])
        neuron_activate.append(temp)
        mask.append(np.array(np.std(temp, axis=0)) > std)
    neuron_activate = np.concatenate(neuron_activate, axis=1)
    mask = np.concatenate(mask, axis=0)
    # print(mask)
    if period == 'train':
        return neuron_activate, mask
    else:
        return neuron_activate

def com_DSA(model_name, model_path, train_path, test_path, adv_path, save_path, adv_list, type_name):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    input, layers = gen_sadl_layers(model_name, model_path)
    models = gen_model(layers, input)

    for adv_name in adv_list:
        print(adv_name + ':')
        save_adv_path = os.path.join(save_path, adv_name)
        if not os.path.exists(save_adv_path):
            os.makedirs(save_adv_path)
        all_train_act = []
        all_test_act = []
        for class_num in range(10):
            x_train = np.load(os.path.join(train_path, str(class_num) + '_train_x.npy'))
            train_neuron_activate, mask = gen_neuron_activate(models, x_train, 0.05, 'train')
            all_train_act.append(train_neuron_activate)
            if type_name == 'test':
                x_test = np.load(os.path.join(test_path, str(class_num) + '_test_x.npy'))
            else:
                x_test = np.load(os.path.join(adv_path, adv_name, str(class_num) + '_all_x.npy'))
            test_neuron_activate = gen_neuron_activate(models, x_test, 0.05, 'test')
            all_test_act.append(test_neuron_activate)
        all_train_act = np.array(all_train_act)
        all_test_act = np.array(all_test_act)
        all_test_score = []
        for class_num in range(10):
            print('class: ' + str(class_num))
            class_test_score = []
            class_test_act = all_test_act[class_num]
            for num in tqdm(range(len(class_test_act))):

                class_train_act = all_train_act[class_num]
                dis_a = float(100000000)
                temp_a = class_train_act[0]
                for temp_i in range(len(class_train_act)):
                    temp_dis = float(((class_test_act[num]-class_train_act[temp_i])**2.0).sum())
                    if temp_dis < dis_a:
                        dis_a = temp_dis
                        temp_a = class_train_act[temp_i]

                dis_b = float(100000000)
                temp_b = class_train_act[0]
                for temp_class_num in range(10):
                    if temp_class_num == class_num:
                        continue
                    other_class_train_act = all_train_act[temp_class_num]
                    for temp_i in range(len(other_class_train_act)):
                        temp_dis = float(((class_test_act[num] - other_class_train_act[temp_i]) ** 2.0).sum())
                        if temp_dis < dis_b:
                            dis_b = temp_dis
                            temp_b = other_class_train_act[temp_i]

                dis = (dis_a / dis_b) ** 0.5
                class_test_score.append(dis)
            all_test_score.append(class_test_score)
        all_test_score = np.array(all_test_score)
        if type_name == 'test':
            np.save(os.path.join(save_adv_path, 'all_test_score.npy'), all_test_score)
        else:
            np.save(os.path.join(save_adv_path, 'all_adv_score.npy'), all_test_score)

def select_adv_index(load_score_path, load_deepfeature_path, save_index_path, adv_list, rate):
    if not os.path.exists(save_index_path):
        os.makedirs(save_index_path)
    for adv_name in adv_list:
        print(adv_name)
        save_adv_path = os.path.join(save_index_path, adv_name)
        if not os.path.exists(save_adv_path):
            os.makedirs(save_adv_path)
        test_score = np.load(os.path.join(load_score_path, adv_name, 'all_test_score.npy'), allow_pickle=True)
        adv_score = np.load(os.path.join(load_score_path, adv_name, 'all_adv_score.npy'), allow_pickle=True)
        max_score = []
        min_score = []
        for class_num in range(10):
            temp_max_score = -100
            temp_min_score = 100
            class_test_score = test_score[class_num]
            class_adv_score = adv_score[class_num]
            if np.max(class_test_score) > temp_max_score:
                temp_max_score = np.max(class_test_score)
            if np.max(class_adv_score) > temp_max_score:
                temp_max_score = np.max(class_adv_score)
            if np.min(class_test_score) < temp_min_score:
                temp_min_score = np.min(class_test_score)
            if np.min(class_adv_score) < temp_min_score:
                temp_min_score = np.min(class_adv_score)
            max_score.append(temp_max_score)
            min_score.append(temp_min_score)
        for class_num in range(10):
            seg_num = 1000
            class_test_buckets = np.zeros(int(seg_num), dtype='int')
            seg_value = (max_score[class_num] - min_score[class_num]) / float(seg_num)
            for value in test_score[class_num]:
                seg_name = math.floor((value - min_score[class_num]) / seg_value)
                if seg_name == seg_num:
                    seg_name -= 1
                class_test_buckets[seg_name] = 1
            class_adv_buckets = []
            for value_num in range(len(adv_score[class_num])):
                value = adv_score[class_num][value_num]
                seg_name = math.floor((value - min_score[class_num]) / seg_value)
                if seg_name == seg_num:
                    seg_name -= 1
                class_adv_buckets.append(seg_name)
            class_adv_buckets = np.array(class_adv_buckets)
            # print(class_adv_buckets)
            sort = class_adv_buckets.argsort()
            can_adv_index = []
            for sort_num in range(len(sort)):
                temp_bucket = class_adv_buckets[sort[sort_num]]
                if class_test_buckets[temp_bucket] == 0:
                    can_adv_index.append(sort[sort_num])
            can_adv_index = np.array(can_adv_index)
            print(class_num)
            for temp_rate in rate:
                save_rate_path = os.path.join(save_adv_path, str(temp_rate))
                if not os.path.exists(save_rate_path):
                    os.makedirs(save_rate_path)
                load_deepfeature_retrain_path = os.path.join(load_deepfeature_path, adv_name, str(temp_rate), str(class_num) + '_y.npy')
                deepfeature_retrain = np.load(load_deepfeature_retrain_path)
                select_num = len(deepfeature_retrain)
                select_index = can_adv_index[:select_num]
                np.save(os.path.join(save_rate_path, str(class_num) + '_select_index.npy'), select_index)

if __name__ == '__main__':
    pass





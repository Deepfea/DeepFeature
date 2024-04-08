import numpy as np
from keras.models import load_model, Model
import os
from tqdm import tqdm
import math
from scipy import stats

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

def reduce_act(all_train_act, all_test_act, th):
    final_train_act = []
    final_test_act = []
    for class_num in range(10):
        class_act = all_train_act[class_num]
        class_act = np.array(class_act)
        neuron_act = np.transpose(class_act)
        flag = np.zeros(len(neuron_act), dtype='int')
        for neuron_num in range(len(neuron_act)):
            temp_x = np.min(neuron_act[neuron_num])
            print('neuron_num(min value):' + str(neuron_num))
            print(temp_x)
            if temp_x > th:
                flag[neuron_num] = 1

        train_act = []
        test_act = []
        class_train_act = all_train_act[class_num]
        class_train_act = np.array(class_train_act)
        class_test_act = all_test_act[class_num]
        class_test_act = np.array(class_test_act)
        train_neuron_act = np.transpose(class_train_act)
        test_neuron_act = np.transpose(class_test_act)
        for neuron_num in range(len(train_neuron_act)):
            if flag[neuron_num] == 1:
                train_act.append(train_neuron_act[neuron_num])
                test_act.append(test_neuron_act[neuron_num])
        train_act = np.array(train_act)
        test_act = np.array(test_act)
        train_act = np.transpose(train_act)
        test_act = np.transpose(test_act)
        print(train_act.shape)
        print(test_act.shape)
        final_train_act.append(train_act)
        final_test_act.append(test_act)
    return final_train_act, final_test_act

def com_LSA(model_name, model_path, train_path, test_path, adv_path, save_path, adv_list, type_name, th):
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
            train_neuron_activate, mask = gen_neuron_activate(models, x_train, th, 'train')
            all_train_act.append(train_neuron_activate)
            if type_name == 'test':
                x_test = np.load(os.path.join(test_path, str(class_num) + '_test_x.npy'))
            else:
                x_test = np.load(os.path.join(adv_path, adv_name, str(class_num) + '_all_x.npy'))
            test_neuron_activate = gen_neuron_activate(models, x_test, th, 'test')
            all_test_act.append(test_neuron_activate)
        all_train_act1, all_test_act1 = reduce_act(all_train_act, all_test_act, th)

        all_test_score = []
        for class_num in range(10):
            class_score = []
            train = all_train_act1[class_num]
            train = np.transpose(train)
            kde = stats.gaussian_kde(train, bw_method='scott')
            test_activate = all_test_act1[class_num]
            for test_neuron_activate in tqdm(test_activate):
                temp = np.empty((len(test_neuron_activate), 1), dtype='float32')
                for num in range(len(test_neuron_activate)):
                    temp[num][0] = test_neuron_activate[num]
                score = kde.logpdf(temp)
                # print(score[0])
                class_score.append(score[0])
            all_test_score.append(class_score)
        all_test_score = np.array(all_test_score)
        if type_name == 'test':
            np.save(os.path.join(save_adv_path, 'all_test_score.npy'), all_test_score)
        else:
            np.save(os.path.join(save_adv_path, 'all_adv_score.npy'), all_test_score)


if __name__ == '__main__':
    pass





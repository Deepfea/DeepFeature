import math
import numpy as np
import keras
from keras.callbacks import ModelCheckpoint
from keras.models import load_model, Model
from keras.layers import Input, Flatten, Conv2D, MaxPooling2D, Dense, Activation
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.optimizers import RMSprop, SGD
import os
from tqdm import tqdm
from scipy import stats
import tensorflow as tf

def vgg19():
    input_tensor = Input(shape=[224, 224, 3])
    # input_tensor = Input(shape=[32, 32, 3])
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(input_tensor)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(10, name='before_softmax')(x)
    x = Activation('softmax', name='predictions')(x)
    # x = Dense(1000, activation='softmax', name='predictions')(x)
    model = keras.Model(input_tensor, x, name='vgg19')
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

def vgg16():
    input_tensor = Input(shape=[32, 32, 3])
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(input_tensor)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(10, name='before_softmax')(x)
    x = Activation('softmax', name='predictions')(x)
    model = keras.Model(input_tensor, x, name='vgg16')
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

def resnet_layer(inputs,num_filters=16,kernel_size=3,strides=1,
                 activation='relu', batch_normalization=True, conv_first=True):

    conv = Conv2D(num_filters, kernel_size=kernel_size, strides=strides,
                  padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))
    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

def resnet_v1(input_shape, depth, num_classes=10):
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x, num_filters=num_filters, strides=strides)
            y = resnet_layer(inputs=y, num_filters=num_filters, activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x, num_filters=num_filters, kernel_size=1, strides=strides, activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2
    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    x = Dense(10, name='before_softmax')(y)
    outputs = Activation('softmax', name='predictions')(x)
    # outputs = Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(y)
    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


def Lenet1():
    nb_classes = 10
    # convolution kernel size
    kernel_size = (5, 5)
    input_tensor = Input(shape=[28, 28, 1])
    # block1
    x = Conv2D(4, kernel_size, activation='relu', padding='same', name='block1_conv1')(input_tensor)
    x = MaxPooling2D(pool_size=(2, 2), name='block1_pool1')(x)
    # block2
    x = Conv2D(12, kernel_size, activation='relu', padding='same', name='block2_conv1')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='block2_pool1')(x)
    x = Flatten(name='flatten')(x)
    x = Dense(nb_classes, name='before_softmax')(x)
    x = Activation('softmax', name='predictions')(x)
    model = keras.Model(input_tensor, x)
    return model

def Lenet5():
    nb_classes = 10
    # convolution kernel size
    kernel_size = (5, 5)
    input_tensor = Input(shape=[28, 28, 1])
    # block1
    x = Conv2D(6, kernel_size, activation='relu', padding='same', name='block1_conv1')(input_tensor)
    x = MaxPooling2D(pool_size=(2, 2), name='block1_pool1')(x)
    # block2
    x = Conv2D(16, kernel_size, activation='relu', padding='same', name='block2_conv1')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='block2_pool1')(x)
    x = Flatten(name='flatten')(x)
    x = Dense(120, activation='relu', name='fc1')(x)
    x = Dense(84, activation='relu', name='fc2')(x)
    x = Dense(nb_classes, name='before_softmax')(x)
    x = Activation('softmax', name='predictions')(x)
    model = keras.Model(input_tensor, x)
    return model

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


def save_train_data(save_path, load_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    load_period_path = os.path.join(load_path, 'train', 'npy')
    save_period_path = os.path.join(save_path, '0_train_data')
    if not os.path.exists(save_period_path):
        os.makedirs(save_period_path)
    for class_name in range(10):
        x = np.load(os.path.join(load_period_path, str(class_name) + '_train_x.npy'))
        y = np.load(os.path.join(load_period_path, str(class_name) + '_train_y.npy'))
        np.save(os.path.join(save_period_path, str(class_name) + '_train_x.npy'), x)
        np.save(os.path.join(save_period_path, str(class_name) + '_train_y.npy'), y)

def save_test_data(save_path, load_path, adv_list):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for adv_name in adv_list:
        save_adv_path = os.path.join(save_path, adv_name)
        if not os.path.exists(save_adv_path):
            os.makedirs(save_adv_path)
        for class_num in range(10):
            load_class_x_path = os.path.join(load_path, adv_name, str(class_num) + '_all_x.npy')
            x = np.load(load_class_x_path)
            y = np.zeros(len(x), dtype='int')
            for temp_i in range(len(y)):
                y[temp_i] = class_num
            load_name_path = os.path.join(load_path, adv_name, str(class_num) + '_all_name.npy')
            name = np.load(load_name_path)
            np.save(os.path.join(save_adv_path, str(class_num) + '_test_x.npy'), x)
            print(x.shape)
            np.save(os.path.join(save_adv_path, str(class_num) + '_test_y.npy'), y)
            print(y.shape)
            np.save(os.path.join(save_adv_path, str(class_num) + '_test_name.npy'), name)
            print(name.shape)

def com_DSA(model_name, model_path, train_path, test_path, save_path, adv_list):
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
            x_test = np.load(os.path.join(test_path, adv_name, str(class_num) + '_test_x.npy'))
            test_neuron_activate = gen_neuron_activate(models, x_test, 0.05, 'test')
            all_test_act.append(test_neuron_activate)
        all_train_act = np.array(all_train_act)
        all_test_act = np.array(all_test_act)
        np.save(os.path.join(save_adv_path, 'all_train_act.npy'), all_train_act)
        np.save(os.path.join(save_adv_path, 'all_test_act.npy'), all_test_act)
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
        np.save(os.path.join(save_adv_path, 'all_test_score.npy'), all_test_score)

def show_value(path, adv_list):
    for adv_name in adv_list:
        value = np.load(os.path.join(path, adv_name, 'all_test_score.npy'), allow_pickle=True)
        for i in range(10):
            class_value = value[i]
            print(class_value)

def select_data(load_train_path, load_test_path, load_score_path, save_path, adv_list, temp_rate):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for adv_name in adv_list:
        load_adv_test_path = os.path.join(load_test_path, adv_name)
        save_adv_path = os.path.join(save_path, adv_name)
        if not os.path.exists(save_adv_path):
            os.makedirs(save_adv_path)
        save_rate_path = os.path.join(save_adv_path, str(temp_rate))
        if not os.path.exists(save_rate_path):
            os.makedirs(save_rate_path)
        all_test_score = np.load(os.path.join(load_score_path, adv_name, 'all_test_score.npy'), allow_pickle=True)
        add_retrain_x = []
        add_retrain_y = []
        for class_num in range(10):
            print('class_num(add):' + str(class_num))
            class_score = all_test_score[class_num]
            class_score = np.array(class_score)
            # sort = class_score.argsort()
            sort = class_score.argsort()[::-1]
            num = math.ceil(len(class_score) * temp_rate)
            select_index = sort[:num]
            test_x = np.load(os.path.join(load_adv_test_path, str(class_num) + '_test_x.npy'))
            for temp_i in range(len(select_index)):
                add_retrain_x.append(test_x[select_index[temp_i]])
                add_retrain_y.append(class_num)
                # print(class_score[select_index[temp_i]])
        add_retrain_x = np.array(add_retrain_x)
        print(add_retrain_x.shape)
        add_retrain_y = np.array(add_retrain_y)
        print(add_retrain_y.shape)
        np.save(os.path.join(save_rate_path, 'add_retrain_x.npy'), add_retrain_x)
        np.save(os.path.join(save_rate_path, 'add_retrain_y.npy'), add_retrain_y)

        for class_num in range(10):
            print('class_num(train):' + str(class_num))
            class_train_x = np.load(os.path.join(load_train_path, str(class_num) + '_train_x.npy'))
            class_train_y = np.load(os.path.join(load_train_path, str(class_num) + '_train_y.npy'))
            if class_num == 0:
                all_train_x = class_train_x
                all_train_y = class_train_y
            else:
                all_train_x = np.concatenate((all_train_x, class_train_x), axis=0)
                all_train_y = np.concatenate((all_train_y, class_train_y), axis=0)
        print(all_train_x.shape)
        print(all_train_y.shape)
        all_train_x = np.concatenate((all_train_x, add_retrain_x), axis=0)
        all_train_y = np.concatenate((all_train_y, add_retrain_y), axis=0)
        print(all_train_x.shape)
        print(all_train_y.shape)
        np.save(os.path.join(save_rate_path, 'train_x.npy'), all_train_x)
        np.save(os.path.join(save_rate_path, 'train_y.npy'), all_train_y)

        for class_num in range(10):
            print('class_num(test):' + str(class_num))
            class_test_x = np.load(os.path.join(load_adv_test_path, str(class_num) + '_test_x.npy'))
            class_test_y = np.load(os.path.join(load_adv_test_path, str(class_num) + '_test_y.npy'))
            if class_num == 0:
                all_test_x = class_test_x
                all_test_y = class_test_y
            else:
                all_test_x = np.concatenate((all_test_x, class_test_x), axis=0)
                all_test_y = np.concatenate((all_test_y, class_test_y), axis=0)
        print(all_test_x.shape)
        print(all_test_y.shape)
        np.save(os.path.join(save_rate_path, 'test_x.npy'), all_test_x)
        np.save(os.path.join(save_rate_path, 'test_y.npy'), all_test_y)

def save_validate_data(load_path, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in range(10):
        temp_x = np.load(os.path.join(load_path, str(i) + '_test_x.npy'))
        temp_y = np.load(os.path.join(load_path, str(i) + '_test_y.npy'))
        if i == 0:
            val_x = temp_x
            val_y = temp_y
        else:
            val_x = np.concatenate((val_x, temp_x), axis=0)
            val_y = np.concatenate((val_y, temp_y), axis=0)
    np.save(os.path.join(save_path, 'val_x.npy'), val_x)
    np.save(os.path.join(save_path, 'val_y.npy'), val_y)

def retrain(train_x, train_y, val_x, val_y, model_save_path, model_name):
    if model_name == 'lenet1':
        g_model = tf.Graph()
        g_session = tf.Session(graph=g_model)
        with g_session.as_default():
            with g_model.as_default():
                model = Lenet1()
                model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
                checkpoint = ModelCheckpoint(filepath=model_save_path,
                                             monitor='val_accuracy', mode='auto', save_best_only='True')
                model.fit(train_x, train_y, epochs=32, batch_size=128, validation_data=(val_x, val_y),
                          callbacks=[checkpoint])
    elif model_name == 'lenet5':
        g_model = tf.Graph()
        g_session = tf.Session(graph=g_model)
        with g_session.as_default():
            with g_model.as_default():
                model = Lenet5()
                model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
                checkpoint = ModelCheckpoint(filepath=model_save_path,
                                             monitor='val_accuracy', mode='auto', save_best_only='True')
                model.fit(train_x, train_y, epochs=32, batch_size=128, validation_data=(val_x, val_y),
                          callbacks=[checkpoint])
    elif model_name == 'resnet20':
        g_model = tf.Graph()
        g_session = tf.Session(graph=g_model)
        with g_session.as_default():
            with g_model.as_default():
                batch_size = 256  # orig paper trained all networks with batch_size=128
                epochs = 32
                version = 1
                depth = 20
                model_type = 'ResNet%dv%d' % (depth, version)
                print(model_type)
                input_shape = train_x.shape[1:]
                print(input_shape)
                model = resnet_v1(input_shape=input_shape, depth=depth)
                model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr_schedule(0)), metrics=['accuracy'])
                # Prepare callbacks for model saving and for learning rate adjustment.
                checkpoint = ModelCheckpoint(filepath=model_save_path, monitor='val_accuracy', verbose=1, save_best_only=True)
                lr_scheduler = LearningRateScheduler(lr_schedule)
                lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
                callbacks = [checkpoint, lr_reducer, lr_scheduler]
                model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, validation_data=(val_x, val_y), shuffle=True, callbacks=callbacks)
    elif model_name == 'vgg16':
        g_model = tf.Graph()
        g_session = tf.Session(graph=g_model)
        with g_session.as_default():
            with g_model.as_default():
                batch_size = 256  # orig paper trained all networks with batch_size=128
                epochs = 32
                model = vgg16()
                checkpoint = ModelCheckpoint(filepath=model_save_path,
                                             monitor='val_accuracy', mode='auto', save_best_only='True')
                model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, validation_data=(val_x, val_y), callbacks=[checkpoint])
    elif model_name == 'resnet50':
        g_model = tf.Graph()
        g_session = tf.Session(graph=g_model)
        with g_session.as_default():
            with g_model.as_default():
                batch_size = 64  # orig paper trained all networks with batch_size=128
                epochs = 32
                version = 1
                depth = 50
                model_type = 'ResNet%dv%d' % (depth, version)
                print(model_type)
                input_shape = train_x.shape[1:]
                print(input_shape)
                model = resnet_v1(input_shape=input_shape, depth=depth)
                model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr_schedule(0)), metrics=['accuracy'])
                # # Prepare callbacks for model saving and for learning rate adjustment.
                checkpoint = ModelCheckpoint(filepath=model_save_path, monitor='val_accuracy', verbose=1, save_best_only=True)
                lr_scheduler = LearningRateScheduler(lr_schedule)
                lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
                callbacks = [checkpoint, lr_reducer, lr_scheduler]
                model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, validation_data=(val_x, val_y),
                              shuffle=True, callbacks=callbacks)
    elif model_name == 'vgg19':
        g_model = tf.Graph()
        g_session = tf.Session(graph=g_model)
        with g_session.as_default():
            with g_model.as_default():
                model = vgg19()
                checkpoint = ModelCheckpoint(filepath=model_save_path,
                                             monitor='val_accuracy', mode='auto', save_best_only='True')
                model.fit(train_x, train_y, epochs=32, batch_size=64, validation_data=(val_x, val_y), callbacks=[checkpoint])

def retrain_model(load_train_path, load_val_path, load_adv_path, save_path, adv_list, temp_rate, model_name):
    for adv_name in adv_list:
        print(adv_name)
        print(temp_rate)
        add_x = []
        add_y = []
        for class_num in range(10):
            x_adv = np.load(os.path.join(load_adv_path, adv_name, str(class_num) + '_all_x.npy'))
            y_adv = np.zeros(len(x_adv), dtype='int')
            for x_adv_num in range(len(x_adv)):
                y_adv[x_adv_num] = class_num
            index = np.load(os.path.join(save_path, adv_name, str(temp_rate), str(class_num) + '_select_index.npy'))
            for index_num in range(len(index)):
                add_x.append(x_adv[index[index_num]])
                add_y.append(y_adv[index[index_num]])

            temp_train_x = np.load(os.path.join(load_train_path, str(class_num) + '_train_x.npy'))
            temp_train_y = np.load(os.path.join(load_train_path, str(class_num) + '_train_y.npy'))
            temp_val_x = np.load(os.path.join(load_val_path, str(class_num) + '_test_x.npy'))
            temp_val_y = np.load(os.path.join(load_val_path, str(class_num) + '_test_y.npy'))
            if class_num == 0:
                train_x = temp_train_x
                train_y = temp_train_y
                val_x = temp_val_x
                val_y = temp_val_y
            else:
                train_x = np.concatenate((train_x, temp_train_x), axis=0)
                train_y = np.concatenate((train_y, temp_train_y), axis=0)
                val_x = np.concatenate((val_x, temp_val_x), axis=0)
                val_y = np.concatenate((val_y, temp_val_y), axis=0)
        add_x = np.array(add_x)
        add_y = np.array(add_y)
        train_x = np.concatenate((train_x, add_x), axis=0)
        train_y = np.concatenate((train_y, add_y), axis=0)
        train_y = keras.utils.to_categorical(train_y, 10)
        val_y = keras.utils.to_categorical(val_y, 10)
        save_model_path = os.path.join(save_path, adv_name, str(temp_rate), model_name + '.h5')
        print(train_x.shape)
        print(train_y.shape)
        print(val_x.shape)
        print(val_y.shape)
        retrain(train_x, train_y, val_x, val_y, save_model_path, model_name)

if __name__ == '__main__':
    pass






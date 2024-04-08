import dataset_1.lenet1.approach.step2_train_and_test_model as step2_train_and_test_model
import os
import re
import numpy as np
import keras
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
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

tf.compat.v1.disable_eager_execution()

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

def select_pic_name(adv_list, load_path_1, load_path_2, save_path, temp_rate):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    char = '_'
    for adv_name in adv_list:
        print(adv_name)
        print(temp_rate)
        save_adv_path = os.path.join(save_path, adv_name)
        if not os.path.exists(save_adv_path):
            os.makedirs(save_adv_path)
        save_rate_path = os.path.join(save_adv_path, str(temp_rate))
        load_rate_path = os.path.join(load_path_2, adv_name, str(temp_rate))
        if not os.path.exists(save_rate_path):
            os.makedirs(save_rate_path)
        for class_num in range(10):
            retrain_x = []
            retrain_y = []
            retrain_name = []
            print('class_num :' + str(class_num))
            class_fea_name = np.load(os.path.join(load_rate_path, str(class_num) + '_fea_name.npy'), allow_pickle=True)
            print(class_fea_name)
            class_comb = np.load(os.path.join(load_rate_path, str(class_num) + '_fea_top_diversity_comb.npy'), allow_pickle=True)
            all_x = np.load(os.path.join(load_path_1, adv_name, str(class_num) + '_fea_pic_x.npy'), allow_pickle=True)
            all_fea_name = np.load(os.path.join(load_path_1, adv_name, str(class_num) + '_fea_name.npy'), allow_pickle=True)
            all_pic_name = np.load(os.path.join(load_path_1, adv_name, str(class_num) + '_fea_pic_name.npy'), allow_pickle=True)
            for fea_num in range(len(class_fea_name)):
                index = np.where(all_fea_name == class_fea_name[fea_num])[0][0]
                fea_comb = class_comb[fea_num]
                # print(fea_comb)
                for x_num in range(len(fea_comb)):
                    retrain_x.append(all_x[index][fea_comb[x_num]])
                    truth_class_name = re.split(re.escape(char), all_pic_name[index][fea_comb[x_num]])[0]
                    retrain_y.append(int(truth_class_name))
                    retrain_name.append(all_pic_name[index][fea_comb[x_num]])
            retrain_x = np.array(retrain_x)
            print(retrain_x.shape)
            np.save(os.path.join(save_rate_path, str(class_num) + '_x.npy'), retrain_x)
            retrain_y = np.array(retrain_y)
            np.save(os.path.join(save_rate_path, str(class_num) + '_y.npy'), retrain_y)
            retrain_name = np.array(retrain_name)
            np.save(os.path.join(save_rate_path, str(class_num) + '_name.npy'), retrain_name)

def np_concatenate(train_data_path, test_data_path, load_rate_path):
    for class_num in range(10):
        temp_train_x = np.load(os.path.join(train_data_path, str(class_num) + '_train_x.npy'))
        temp_train_y = np.load(os.path.join(train_data_path, str(class_num) + '_train_y.npy'))
        temp_val_x = np.load(os.path.join(test_data_path, str(class_num) + '_test_x.npy'))
        temp_val_y = np.load(os.path.join(test_data_path, str(class_num) + '_test_y.npy'))
        temp_add_x = np.load(os.path.join(load_rate_path, str(class_num) + '_x.npy'))
        temp_add_y = np.load(os.path.join(load_rate_path, str(class_num) + '_y.npy'))
        if class_num == 0:
            train_x = temp_train_x
            train_y = temp_train_y
            val_x = temp_val_x
            val_y = temp_val_y
            add_x = temp_add_x
            add_y = temp_add_y
        else:
            train_x = np.concatenate((train_x, temp_train_x), axis=0)
            train_y = np.concatenate((train_y, temp_train_y), axis=0)
            val_x = np.concatenate((val_x, temp_val_x), axis=0)
            val_y = np.concatenate((val_y, temp_val_y), axis=0)
            add_x = np.concatenate((add_x, temp_add_x), axis=0)
            add_y = np.concatenate((add_y, temp_add_y), axis=0)
    train_x = np.concatenate((train_x, add_x), axis=0)
    train_y = np.concatenate((train_y, add_y), axis=0)
    print(train_x.shape)
    print(val_x.shape)
    return train_x, train_y, val_x, val_y
def retrain(adv_list, dataset_path, load_path, temp_rate, model_name):
    for adv_name in adv_list:
        print(adv_name + ':')
        load_rate_path = os.path.join(load_path, adv_name, str(temp_rate))
        train_data_path = os.path.join(dataset_path, 'train', 'npy')
        test_data_path = os.path.join(dataset_path, 'test', 'npy')
        if model_name == 'lenet1':
            for train_num in range(1):
                g_model = tf.Graph()
                g_session = tf.Session(graph=g_model)
                with g_session.as_default():
                    with g_model.as_default():
                        model_save_path = os.path.join(load_rate_path, str(train_num) + '_' + model_name + '.h5')
                        train_x, train_y, val_x, val_y = np_concatenate(train_data_path, test_data_path, load_rate_path)
                        train_y = keras.utils.to_categorical(train_y, 10)
                        val_y = keras.utils.to_categorical(val_y, 10)
                        model = Lenet1()
                        model.compile(loss='categorical_crossentropy', optimizer='adam',
                                                  metrics=['accuracy'])
                        checkpoint = ModelCheckpoint(filepath=model_save_path,
                                                                 monitor='val_accuracy', mode='auto',
                                                                 save_best_only='True')
                        model.fit(train_x, train_y, epochs=32, batch_size=128,
                                              validation_data=(val_x, val_y), verbose=2,
                                              callbacks=[checkpoint])
        elif model_name == 'lenet5':
            for train_num in range(1):
                g_model = tf.Graph()
                g_session = tf.Session(graph=g_model)
                with g_session.as_default():
                    with g_model.as_default():
                        model_save_path = os.path.join(load_rate_path, str(train_num) + '_' + model_name + '.h5')
                        train_x, train_y, val_x, val_y = np_concatenate(train_data_path, test_data_path, load_rate_path)
                        train_y = keras.utils.to_categorical(train_y, 10)
                        val_y = keras.utils.to_categorical(val_y, 10)
                        model = Lenet5()
                        model.compile(loss='categorical_crossentropy', optimizer='adam',
                                      metrics=['accuracy'])
                        checkpoint = ModelCheckpoint(filepath=model_save_path,
                                                     monitor='val_accuracy', mode='auto',
                                                     save_best_only='True')
                        model.fit(train_x, train_y, epochs=32, batch_size=128,
                                  validation_data=(val_x, val_y),
                                  callbacks=[checkpoint])
        elif model_name == 'resnet20':
            for train_num in range(1):
                g_model = tf.Graph()
                g_session = tf.Session(graph=g_model)
                with g_session.as_default():
                    with g_model.as_default():
                        model_save_path = os.path.join(load_rate_path, str(train_num) + '_' + model_name + '.h5')
                        train_x, train_y, val_x, val_y = np_concatenate(train_data_path, test_data_path, load_rate_path)
                        train_y = keras.utils.to_categorical(train_y, 10)
                        val_y = keras.utils.to_categorical(val_y, 10)
                        version = 1
                        depth = 20
                        model_type = 'ResNet%dv%d' % (depth, version)
                        print(model_type)
                        input_shape = train_x.shape[1:]
                        print(input_shape)
                        model = resnet_v1(input_shape=input_shape, depth=depth)
                        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr_schedule(0)),
                                      metrics=['accuracy'])
                        # Prepare callbacks for model saving and for learning rate adjustment.
                        checkpoint = ModelCheckpoint(filepath=model_save_path, monitor='val_accuracy',
                                                     verbose=1, save_best_only=True)
                        lr_scheduler = LearningRateScheduler(lr_schedule)
                        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5,
                                                       min_lr=0.5e-6)
                        callbacks = [checkpoint, lr_reducer, lr_scheduler]
                        model.fit(train_x, train_y, batch_size=128, epochs=32,
                                  validation_data=(val_x, val_y), shuffle=True, callbacks=callbacks)
        elif model_name == 'vgg16':
            for train_num in range(1):
                g_model = tf.Graph()
                g_session = tf.Session(graph=g_model)
                with g_session.as_default():
                    with g_model.as_default():
                        model_save_path = os.path.join(load_rate_path, str(train_num) + '_' + model_name + '.h5')
                        train_x, train_y, val_x, val_y = np_concatenate(train_data_path, test_data_path, load_rate_path)
                        model = vgg16()
                        checkpoint = ModelCheckpoint(filepath=model_save_path,
                                                                 monitor='val_accuracy', mode='auto',
                                                                 save_best_only='True')
                        model.fit(train_x, train_y, epochs=32, batch_size=128,
                                              validation_data=(val_x, val_y), callbacks=[checkpoint])
        elif model_name == 'resnet50':
            for train_num in range(1):
                g_model = tf.Graph()
                g_session = tf.Session(graph=g_model)
                with g_session.as_default():
                    with g_model.as_default():
                        model_save_path = os.path.join(load_rate_path, str(train_num) + '_' + model_name + '.h5')
                        train_x, train_y, val_x, val_y = np_concatenate(train_data_path, test_data_path, load_rate_path)
                        train_y = keras.utils.to_categorical(train_y, 10)
                        val_y = keras.utils.to_categorical(val_y, 10)
                        version = 1
                        depth = 50
                        model_type = 'ResNet%dv%d' % (depth, version)
                        print(model_type)
                        input_shape = train_x.shape[1:]
                        print(input_shape)
                        model = resnet_v1(input_shape=input_shape, depth=depth)
                        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr_schedule(0)),
                                      metrics=['accuracy'])
                        # # Prepare callbacks for model saving and for learning rate adjustment.
                        checkpoint = ModelCheckpoint(filepath=model_save_path, monitor='val_accuracy',
                                                     verbose=1, save_best_only=True)
                        lr_scheduler = LearningRateScheduler(lr_schedule)
                        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5,
                                                       min_lr=0.5e-6)
                        callbacks = [checkpoint, lr_reducer, lr_scheduler]
                        model.fit(train_x, train_y, batch_size=64, epochs=20,
                                  validation_data=(val_x, val_y),
                                  shuffle=True, callbacks=callbacks)
        elif model_name == 'vgg19':
            for train_num in range(1):
                g_model = tf.Graph()
                g_session = tf.Session(graph=g_model)
                with g_session.as_default():
                    with g_model.as_default():
                        model_save_path = os.path.join(load_rate_path, str(train_num) + '_' + model_name + '.h5')
                        train_x, train_y, val_x, val_y = np_concatenate(train_data_path, test_data_path, load_rate_path)
                        train_y = keras.utils.to_categorical(train_y, 10)
                        val_y = keras.utils.to_categorical(val_y, 10)
                        model = vgg19()
                        checkpoint = ModelCheckpoint(filepath=model_save_path,
                                                     monitor='val_accuracy', mode='auto',
                                                     save_best_only='True')
                        model.fit(train_x, train_y, epochs=20, batch_size=64,
                                  validation_data=(val_x, val_y), callbacks=[checkpoint])


if __name__ == '__main__':
    pass

